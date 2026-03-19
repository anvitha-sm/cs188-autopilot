import collections
import heapq
import math
import os
import threading
import time
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import (
    ObjectDetector,
    ObjectDetectorOptions,
    RunningMode,
)

from controller import Supervisor, Mouse

GESTURE_MODEL_PATH = "models/gesture_recognizer.task"
OBJ_MODEL_PATH = "models/efficientdet_lite0.tflite"
GESTURE_MIN_SCORE = 0.6
GESTURE_GAP_FRAMES = 8
SPEECH_ENERGY_THRESHOLD = 200
SPEECH_PAUSE_THRESHOLD = 0.5
SPEED = 5.0
SPEED_STEP = 1.0
SPEED_MIN = 1.0
SPEED_MAX = 13.0
MAX_STEER = 2.0
WP_RADIUS = 3.5
WP_SPACING = 2.0

LANE_CHANGE_LOOKAHEAD = WP_RADIUS * 1.8
JUNC_RADIUS = 5.0
LANE_CHANGE_DURATION = 0.25
BODY_OFFSET = 0.0
ROAD_WIDTH = 21.5
NUM_LANES = 4
LANE_WIDTH = ROAD_WIDTH / NUM_LANES
OUR_LANE_OFFSETS = [
    ROAD_WIDTH / 2 - LANE_WIDTH / 2,
    ROAD_WIDTH / 2 - LANE_WIDTH * 1.5,
]
NUM_OUR_LANES = len(OUR_LANE_OFFSETS)

CAM_CANDIDATES = ["front_camera", "camera", "Camera", "FrontCamera", "camera(1)"]
ROI_TOP_FRAC = 0.60
OBJ_SEE_FRAC = 0.30
OBJ_CLOSE_FRAC = 0.50
OBJ_MIN_SCORE = 0.30
MISS_NEEDED = 8
BOTH_BLOCKED_NEEDED = 5
HUMAN_OVERRIDE_COOLDOWN = 50
TURN_OVERRIDE_WINDOW = 6.0

_fist_event = threading.Event()
_gesture_lock = threading.Lock()
_gesture_queue = collections.deque(maxlen=4)
_speed_lock = threading.Lock()
_turn_lock = threading.Lock()
_speech_turn_cmd = None
_det_lock = threading.Lock()
_det_hit = [False, False]

def angle_diff(a, b):
    d = a - b
    while d > math.pi: d -= 2 * math.pi
    while d < -math.pi: d += 2 * math.pi
    return d

def hdist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)

def road_canonical_dir(road):
    dx = road.ex - road.sx
    dy = road.ey - road.sy
    l = math.hypot(dx, dy)
    return (dx / l, dy / l) if l > 1e-9 else (1.0, 0.0)

def road_right_normal(road):
    # AI-generated: right-hand normal for keeping the vehicle on the correct side of the road
    fdx, fdy = road_canonical_dir(road)
    return fdy, -fdx

def apply_lateral(cx, cy, road, canonical_sl):
    rx, ry = road_right_normal(road)
    return cx + rx * canonical_sl, cy + ry * canonical_sl

def _first_wp_ahead(wps, cx, cy, heading, min_fwd):
    for wi, (wx, wy) in enumerate(wps):
        fwd = (wx - cx) * math.cos(heading) + (wy - cy) * math.sin(heading)
        if fwd >= min_fwd:
            return wi
    return max(0, len(wps) - 1)

def _skip_behind_wps(wps, cx, cy, heading):
    for wi, (wx, wy) in enumerate(wps):
        fwd = (wx - cx) * math.cos(heading) + (wy - cy) * math.sin(heading)
        if fwd > 0:
            return wi
    return max(0, len(wps) - 1)

class Junction: # just x and y coordinates of junction with its id, used for routing
    def __init__(self, jid, x, y):
        self.id = jid
        self.x = x
        self.y = y

class StraightRoad: # segment on road network
    curved = False
    arc_cx = arc_cy = radius = 0.0
    def __init__(self, rid, sjid, ejid, sx, sy, ex, ey):
        self.id = rid
        self.sjid = sjid
        self.ejid = ejid
        self.sx, self.sy = sx, sy
        self.ex, self.ey = ex, ey

    def other_end(self, jid):
        return self.ejid if jid == self.sjid else self.sjid

    def center_point(self, t):
        return (
            self.sx + (self.ex - self.sx) * t,
            self.sy + (self.ey - self.sy) * t,
        )

    def waypoints(self, from_jid, canonical_sl):
        # AI-generated: evenly spaced waypoint generation along segment,
        # offset into the correct lane on the right side of the road
        t0, t1 = (0.1, 0.9) if from_jid == self.sjid else (0.9, 0.1)
        seg_len = math.hypot(self.ex - self.sx, self.ey - self.sy)
        n = max(2, int(seg_len / WP_SPACING))
        wps = []
        for i in range(1, n + 1):
            t = t0 + (t1 - t0) * i / n
            cx, cy = self.center_point(t)
            wps.append(apply_lateral(cx, cy, self, canonical_sl))
        return wps


class CurvedRoad: # segment on road network
    curved = True

    def __init__(self, rid, sjid, ejid, sx, sy, ex, ey, arc_cx, arc_cy, radius):
        self.id = rid
        self.sjid = sjid
        self.ejid = ejid
        self.sx, self.sy = sx, sy
        self.ex, self.ey = ex, ey
        self.arc_cx = arc_cx
        self.arc_cy = arc_cy
        self.radius = radius

    def other_end(self, jid):
        return self.ejid if jid == self.sjid else self.sjid

    def waypoints(self, from_jid, canonical_sl):
        # AI-generated: arc waypoint generation, lateral offset applied
        # after sampling — not quite perpendicular to the tangent but close enough
        if from_jid == self.sjid:
            a0 = math.atan2(self.sy - self.arc_cy, self.sx - self.arc_cx)
            a1 = math.atan2(self.ey - self.arc_cy, self.ex - self.arc_cx)
        else:
            a0 = math.atan2(self.ey - self.arc_cy, self.ex - self.arc_cx)
            a1 = math.atan2(self.sy - self.arc_cy, self.sx - self.arc_cx)

        da = angle_diff(a1, a0)
        arc_len = abs(da) * self.radius
        n = max(3, int(arc_len / WP_SPACING))
        wps = []
        for i in range(1, n + 1):
            t = 0.1 + (i / n) * 0.8
            angle = a0 + da * t
            cx = self.arc_cx + self.radius * math.cos(angle)
            cy = self.arc_cy + self.radius * math.sin(angle)
            wps.append(apply_lateral(cx, cy, self, canonical_sl))
        return wps

junctions = {}
roads = {}
graph = {}
road_between = {}
def build_graph(car_supervisor):
    root = car_supervisor.getRoot()
    kids = root.getField("children")

    for i in range(kids.getCount()):
        node = kids.getMFNode(i)
        if node.getTypeName() in ("Crossroad", "RoadIntersection"):
            jid = node.getField("id").getSFString()
            p = node.getField("translation").getSFVec3f()
            junctions[jid] = Junction(jid, p[0], p[1])
            graph[jid] = []
    for i in range(kids.getCount()):
        node = kids.getMFNode(i)
        typ = node.getTypeName()
        if typ not in ("StraightRoadSegment", "CurvedRoadSegment"):
            continue

        rid = node.getField("id").getSFString()
        sjid = node.getField("startJunction").getSFString()
        ejid = node.getField("endJunction").getSFString()
        sj, ej = junctions[sjid], junctions[ejid]
        if typ == "CurvedRoadSegment":
            p = node.getField("translation").getSFVec3f()
            radius = node.getField("curvatureRadius").getSFFloat()
            road = CurvedRoad(rid, sjid, ejid,
                              sj.x, sj.y, ej.x, ej.y,
                              p[0], p[1], radius)
        else:
            road = StraightRoad(rid, sjid, ejid, sj.x, sj.y, ej.x, ej.y)

        roads[rid] = road
        road_between[(sjid, ejid)] = road
        road_between[(ejid, sjid)] = road
        if ejid not in graph[sjid]: graph[sjid].append(ejid)
        if sjid not in graph[ejid]: graph[ejid].append(sjid)

def closest_road(px, py): # snap to the road goal
    best, best_d = None, 1e9
    for road in roads.values():
        if road.curved:
            d = abs(hdist(px, py, road.arc_cx, road.arc_cy) - road.radius)
        else:
            dx, dy = road.ex - road.sx, road.ey - road.sy
            l2 = dx * dx + dy * dy
            if l2 < 1e-9:
                continue
            t = max(0.0, min(1.0,
                ((px - road.sx) * dx + (py - road.sy) * dy) / l2))
            d = hdist(px, py, road.sx + dx * t, road.sy + dy * t)
        if d < best_d:
            best_d, best = d, road
    return best

def astar(start, goal, forbidden=None): # plan route from the start position of car to the road goal
    if start == goal:
        return [start]

    frontier = [(0.0, start)]
    came_from = {start: None}
    g_cost = {start: 0.0}

    while frontier:
        _, cur = heapq.heappop(frontier)
        if cur == goal:
            break
        for nb in graph.get(cur, []):
            if cur == start and nb == forbidden:
                continue
            nc = g_cost[cur] + hdist(junctions[cur].x, junctions[cur].y,
                                     junctions[nb].x, junctions[nb].y)
            if nb not in g_cost or nc < g_cost[nb]:
                g_cost[nb] = nc
                came_from[nb] = cur
                heapq.heappush(frontier,
                    (nc + hdist(junctions[nb].x, junctions[nb].y,
                                junctions[goal].x, junctions[goal].y), nb))

    path, node = [], goal
    while node:
        path.append(node)
        node = came_from.get(node)
    path.reverse()
    return path

def turn_label(from_jid, via_jid, to_jid): # which way to turn
    in_dx = junctions[via_jid].x - junctions[from_jid].x
    in_dy = junctions[via_jid].y - junctions[from_jid].y
    out_dx = junctions[to_jid].x - junctions[via_jid].x
    out_dy = junctions[to_jid].y - junctions[via_jid].y
    angle = math.atan2(in_dx * out_dy - in_dy * out_dx,
                       in_dx * out_dx + in_dy * out_dy)
    if abs(angle) < 0.4: return "STRAIGHT"
    if abs(angle) > 2.5: return "U-TURN"
    return "LEFT" if angle > 0 else "RIGHT"

def snap_to_road_lane(click_x, click_y, ref_next_jid, ref_from_jid):
    # AI-generated: projects click onto nearest road center line (straight or arc),
    # uses dot product to determine which side of the road, then picks the cheaper
    # junction endpoint via A* cost comparison
    g_road = closest_road(click_x, click_y)

    if g_road.curved:
        angle = math.atan2(click_y - g_road.arc_cy, click_x - g_road.arc_cx)
        snap_x = g_road.arc_cx + g_road.radius * math.cos(angle)
        snap_y = g_road.arc_cy + g_road.radius * math.sin(angle)
    else:
        dx, dy = g_road.ex - g_road.sx, g_road.ey - g_road.sy
        l2 = dx * dx + dy * dy
        t = max(0.1, min(0.9,
            ((click_x - g_road.sx) * dx + (click_y - g_road.sy) * dy) / l2))
        snap_x = g_road.sx + dx * t
        snap_y = g_road.sy + dy * t

    rx, ry = road_right_normal(g_road)
    side_dot = (click_x - snap_x) * rx + (click_y - snap_y) * ry
    goal_sl = OUR_LANE_OFFSETS[0] if side_dot >= 0 else -OUR_LANE_OFFSETS[0]
    snap_x, snap_y = apply_lateral(snap_x, snap_y, g_road, goal_sl)

    best_jid, best_cost = None, float("inf")
    for cjid in (g_road.sjid, g_road.ejid):
        path = astar(ref_next_jid, cjid, forbidden=ref_from_jid)
        cost = (
            sum(hdist(junctions[path[i]].x, junctions[path[i]].y,
                      junctions[path[i+1]].x, junctions[path[i+1]].y)
                for i in range(len(path) - 1))
            if len(path) >= 2 else 0.0
        )
        cost += hdist(snap_x, snap_y, junctions[cjid].x, junctions[cjid].y)
        if cost < best_cost:
            best_cost, best_jid = cost, cjid

    return snap_x, snap_y, g_road, best_jid

_marker_node = None

def place_marker(car_supervisor, x, y, z): # visualization of goal marker
    global _marker_node
    remove_marker()
    root = car_supervisor.getRoot()
    kids = root.getField("children")
    kids.importMFNodeFromString(-1, f"""
      DEF GOAL_MARKER Transform {{
        translation {x} {y} {z + 1.5}
        children [
          Shape {{
            appearance PBRAppearance {{
              baseColor 1 0.2 0
              emissiveColor 1 0.4 0
              roughness 0.3
            }}
            geometry Sphere {{ radius 1.2 }}
          }}
          PointLight {{ color 1 0.4 0  intensity 2  radius 20 }}
        ]
      }}
    """)
    _marker_node = kids.getMFNode(kids.getCount() - 1)

def remove_marker():
    global _marker_node
    if _marker_node is not None:
        _marker_node.remove()
        _marker_node = None

_last_gesture_name = None
_same_gesture_ct = 0

def _gesture_callback(result, output_image, timestamp_ms): # thread for gesture recognition
    global _last_gesture_name, _same_gesture_ct

    gesture_name = None
    if result.gestures:
        top = result.gestures[0][0]
        if top.score > GESTURE_MIN_SCORE:
            gesture_name = top.category_name

    if gesture_name == "Closed_Fist":
        if not _fist_event.is_set():
            _fist_event.set()
    else:
        if _fist_event.is_set():
            _fist_event.clear()

    if gesture_name in ("Thumb_Up", "Thumb_Down"):
        if gesture_name != _last_gesture_name:
            _same_gesture_ct = 1
            direction = -1 if gesture_name == "Thumb_Up" else +1
            with _gesture_lock:
                _gesture_queue.append(direction)
        else:
            _same_gesture_ct += 1
            if _same_gesture_ct % (GESTURE_GAP_FRAMES * 8) == 0:
                direction = -1 if gesture_name == "Thumb_Up" else +1
                with _gesture_lock:
                    _gesture_queue.append(direction)
    else:
        if _last_gesture_name in ("Thumb_Up", "Thumb_Down"):
            _same_gesture_ct = 0

    _last_gesture_name = gesture_name

def _gesture_thread():
    options = mp_vision.GestureRecognizerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=GESTURE_MODEL_PATH),
        running_mode=mp_vision.RunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        result_callback=_gesture_callback,
    )

    cap = None
    for idx in range(4):
        test = cv2.VideoCapture(idx)
        if test.isOpened():
            ret, _ = test.read()
            if ret:
                cap = test
                break
        test.release()

    if cap is None:
        return

    with mp_vision.GestureRecognizer.create_from_options(options) as recognizer:
        timestamp_ms = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms += 33
            recognizer.recognize_async(mp_image, timestamp_ms)
            time.sleep(0.01)
    cap.release()

def _speech_thread(): # separate thread for speech recognition
    global SPEED, _speech_turn_cmd

    try:
        import speech_recognition as sr
    except ImportError:
        return

    r = sr.Recognizer()
    r.energy_threshold = SPEECH_ENERGY_THRESHOLD
    r.dynamic_energy_threshold = False
    r.pause_threshold = SPEECH_PAUSE_THRESHOLD

    try:
        mic = sr.Microphone()
    except Exception:
        return

    with mic as source:
        r.adjust_for_ambient_noise(source, duration=1)

    while True:
        try:
            with mic as source:
                audio = r.listen(source, timeout=5, phrase_time_limit=3)
            text = r.recognize_google(audio).lower()

            with _speed_lock:
                if "faster" in text:
                    SPEED = min(SPEED_MAX, round(SPEED + SPEED_STEP, 1))
                elif "slower" in text:
                    SPEED = max(SPEED_MIN, round(SPEED - SPEED_STEP, 1))

            with _turn_lock:
                if "left" in text and "right" not in text:
                    _speech_turn_cmd = "left"
                elif "right" in text and "left" not in text:
                    _speech_turn_cmd = "right"

        except (sr.WaitTimeoutError, sr.UnknownValueError):
            pass
        except sr.RequestError:
            time.sleep(2)
        except Exception:
            time.sleep(1)

def _bgr_from_webots(raw_bytes, w, h):
    arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((h, w, 4))
    return arr[:, :, :3]

def _detect_lane_from_camera(bgr):
    # AI-generated: HSV histogram approach for yellow center line and white curb
    # detection, per-column smoothing and side-of-frame heuristics to figure out
    # which lane we're in
    h, w = bgr.shape[:2]
    roi = bgr[int(h * ROI_TOP_FRAC):, :]
    rh, rw = roi.shape[:2]
    scan = roi[rh // 2:, :]
    sh = scan.shape[0]
    scan_hsv = cv2.cvtColor(scan, cv2.COLOR_BGR2HSV)

    yellow_hist = np.sum(
        cv2.inRange(scan_hsv,
                    np.array([15, 80, 80]),
                    np.array([40, 255, 255])),
        axis=0).astype(float)
    white_hist = np.sum(
        cv2.inRange(scan_hsv,
                    np.array([0, 0, 160]),
                    np.array([180, 40, 255])),
        axis=0).astype(float)

    k = np.ones(15) / 15
    yellow_hist = np.convolve(yellow_hist, k, mode="same")
    white_hist = np.convolve(white_hist, k, mode="same")

    cx = rw / 2.0
    yellow_col = None
    left_white_col = None
    right_white_col = None

    if yellow_hist.max() > sh * 5:
        yellow_col = float(np.argmax(yellow_hist))
    if white_hist[:int(cx)].max() > sh * 3:
        left_white_col = float(np.argmax(white_hist[:int(cx)]))
    if white_hist[int(cx):].max() > sh * 3:
        right_white_col = float(np.argmax(white_hist[int(cx):])) + cx

    our_lane = None
    yellow_blocked = False

    if yellow_col is not None:
        if yellow_col / rw < 0.25:
            our_lane = 0
        else:
            our_lane = 1
            yellow_blocked = True

    if our_lane is None:
        if right_white_col is not None and left_white_col is None:
            our_lane = 1
        elif left_white_col is not None and right_white_col is None:
            our_lane = 0
        elif left_white_col is not None and right_white_col is not None:
            our_lane = 0 if abs(right_white_col - cx) > abs(left_white_col - cx) else 1

    return our_lane, yellow_blocked

_obj_ts = 0

def _obj_det_callback(result, output_image, timestamp_ms):
    # AI-generated: bounding box bottom-edge thresholds and lane assignment by
    # horizontal center — also handled some edge cases around reversed travel direction
    global _det_hit
    if result is None:
        return

    h, w = output_image.height, output_image.width
    mid_x = w / 2
    hit = [False, False]

    for det in result.detections:
        score = det.categories[0].score if det.categories else 0.0
        if score < OBJ_MIN_SCORE:
            continue

        bb = det.bounding_box
        box_bottom = (bb.origin_y + bb.height) / h
        box_cx = bb.origin_x + bb.width / 2

        if box_bottom < OBJ_SEE_FRAC:
            continue

        if _travelling_forward:
            z = 0 if box_cx > mid_x else 1
        else:
            z = 0 if box_cx < mid_x else 1

        if box_bottom >= OBJ_CLOSE_FRAC:
            hit[z] = True

    with _det_lock:
        _det_hit[0] = hit[0]
        _det_hit[1] = hit[1]

def _obstacle_detect(bgr):
    global _obj_ts
    _obj_ts += 33
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    _obj_detector.detect_async(mp_img, _obj_ts)
    with _det_lock:
        return [_det_hit[0], _det_hit[1]]

def _update_zone_blocked_state(hit): # figure out if lane is blocked for obstacle avoidance
    for z in range(2):
        if hit[z]:
            _near_miss_ct[z] = 0
            if not _lane_near_blocked[z]:
                _lane_near_blocked[z] = True
        else:
            _near_miss_ct[z] += 1
            if _lane_near_blocked[z] and _near_miss_ct[z] >= MISS_NEEDED:
                _lane_near_blocked[z] = False

def _update_lane_from_camera():
    global _camera_lane, _yellow_blocked
    if not USE_CAMERA:
        return
    raw = cam.getImage()
    if raw is None:
        return
    bgr = _bgr_from_webots(raw, CAM_W, CAM_H)
    lane, yblocked = _detect_lane_from_camera(bgr)
    if lane is not None:
        _camera_lane = lane
        _yellow_blocked = yblocked
    hit = _obstacle_detect(bgr)
    _update_zone_blocked_state(hit)

def _current_canonical_sl(): # orientation of car currently for the car to know which axis to move about
    if _lane_t >= 1.0:
        offset = OUR_LANE_OFFSETS[_cur_lane]
    else:
        s = _lane_t * _lane_t * (3.0 - 2.0 * _lane_t)
        offset = _lane_offset_from + s * (_lane_offset_to - _lane_offset_from)
    return offset if _travelling_forward else -offset

def _measure_actual_canonical_sl(road, px, py, travelling_fwd):
    # AI-generated: measures signed lateral offset from road center line at startup
    # so the car spawns in the right lane rather than defaulting to lane 0
    if not road.curved:
        dx, dy = road.ex - road.sx, road.ey - road.sy
        l2 = dx * dx + dy * dy
        t = ((px - road.sx) * dx + (py - road.sy) * dy) / max(l2, 1e-9)
        proj_x = road.sx + dx * t
        proj_y = road.sy + dy * t
        l = math.sqrt(l2)
        rx, ry = dy / l, -dx / l
        return (px - proj_x) * rx + (py - proj_y) * ry
    else:
        dist = math.hypot(px - road.arc_cx, py - road.arc_cy)
        a0 = math.atan2(road.sy - road.arc_cy, road.sx - road.arc_cx)
        a1 = math.atan2(road.ey - road.arc_cy, road.ex - road.arc_cx)
        da = angle_diff(a1, a0)
        sign = 1 if da > 0 else -1
        return sign * (dist - road.radius)

def _try_lane_change(direction, source="gesture"):
    global _cur_lane, _target_lane, _lane_t
    global _lane_offset_from, _lane_offset_to
    global waypoints, wp_idx

    base_lane = _target_lane
    new_lane = base_lane + direction

    if new_lane < 0 or new_lane >= NUM_OUR_LANES:
        return False

    if _lane_t < 1.0 and new_lane == _target_lane:
        return False

    if source == "avoidance" and _lane_near_blocked[new_lane]:
        return False

    if _lane_t >= 1.0:
        _lane_offset_from = OUR_LANE_OFFSETS[_cur_lane]
    else:
        s = _lane_t * _lane_t * (3.0 - 2.0 * _lane_t)
        _lane_offset_from = _lane_offset_from + (_lane_offset_to - _lane_offset_from) * s

    _lane_offset_to = OUR_LANE_OFFSETS[new_lane]
    _target_lane = new_lane
    _lane_t = 0.0

    return True

_near_miss_ct = [0, 0]
_lane_near_blocked = [False, False]
_dodged = False
_both_blocked_ct = 0

def _handle_obstacle_avoidance():
    global _dodged, _both_blocked_ct

    if _lane_t < 1.0:
        return
    if final_phase:
        return
    if _avoidance_suppressed > 0:
        return

    adj_lane = 1 - _cur_lane

    if _dodged is not False and not _lane_near_blocked[_dodged]:
        _dodged = False
    if not _lane_near_blocked[_cur_lane]:
        _both_blocked_ct = 0
        return

    if _lane_near_blocked[adj_lane]:
        _both_blocked_ct += 1
        if _both_blocked_ct >= BOTH_BLOCKED_NEEDED:
            _fist_event.set()
        return

    _both_blocked_ct = 0

    if _dodged is not False and adj_lane == _dodged and _lane_near_blocked[_dodged]:
        _fist_event.set()
        return
    ok = _try_lane_change(adj_lane - _cur_lane, source="avoidance")
    if not ok:
        _fist_event.set()
    else:
        _dodged = _cur_lane

def _apply_speech_turn_override(arrived_jid, from_jid, goal_jid_ref): # if human command for faster or slower
    global _speech_turn_cmd

    with _turn_lock:
        cmd = _speech_turn_cmd
        _speech_turn_cmd = None

    if cmd is None:
        return None, None

    best_nb = None
    best_angle = None

    for nb in graph.get(arrived_jid, []):
        if nb == from_jid:
            continue
        lbl = turn_label(from_jid, arrived_jid, nb)
        if lbl.upper() == cmd.upper():
            in_dx = junctions[arrived_jid].x - junctions[from_jid].x
            in_dy = junctions[arrived_jid].y - junctions[from_jid].y
            out_dx = junctions[nb].x - junctions[arrived_jid].x
            out_dy = junctions[nb].y - junctions[arrived_jid].y
            a = math.atan2(in_dx * out_dy - in_dy * out_dx,
                           in_dx * out_dx + in_dy * out_dy)
            if best_nb is None or abs(a) > abs(best_angle):
                best_nb = nb
                best_angle = a

    if best_nb is None:
        return None, None

    return best_nb, goal_jid_ref

for _model_path in (GESTURE_MODEL_PATH, OBJ_MODEL_PATH):
    if not os.path.exists(_model_path):
        raise FileNotFoundError(
            f"Model file not found: {_model_path}\n"
            f"Download it manually — see README.md for links."
        )

threading.Thread(target=_gesture_thread, daemon=True).start()
threading.Thread(target=_speech_thread, daemon=True).start()
car = Supervisor()
timestep = int(car.getBasicTimeStep())
car_node = car.getSelf()
GROUND_Z = car_node.getPosition()[2]

cam = None
for _name in CAM_CANDIDATES:
    _dev = car.getDevice(_name)
    if _dev is not None:
        cam = _dev
        break
if cam is None:
    for _i in range(car.getNumberOfDevices()):
        _dev = car.getDeviceByIndex(_i)
        if _dev.getNodeType() == car.CAMERA:
            cam = _dev
            break

if cam is None:
    USE_CAMERA = False
    CAM_W = CAM_H = 0
else:
    cam.enable(timestep)
    USE_CAMERA = True
    CAM_W = cam.getWidth()
    CAM_H = cam.getHeight()

_camera_lane = None
_yellow_blocked = False
_obj_options = ObjectDetectorOptions(
    base_options=mp_python.BaseOptions(model_asset_path=OBJ_MODEL_PATH),
    running_mode=RunningMode.LIVE_STREAM,
    max_results=10,
    score_threshold=OBJ_MIN_SCORE,
    result_callback=_obj_det_callback,
)
_obj_detector = ObjectDetector.create_from_options(_obj_options)

mouse = car.getMouse()
mouse.enable(timestep)
mouse.enable3dPosition()

build_graph(car)
pos = car_node.getPosition()
car_x, car_y = pos[0], pos[1]

cur_road = closest_road(car_x, car_y)
_rot = car_node.getField("rotation").getSFRotation()
car_heading = _rot[3] if _rot[2] > 0 else -_rot[3]

fdx, fdy = road_canonical_dir(cur_road)
road_fwd_heading = math.atan2(fdy, fdx)
_travelling_forward = abs(angle_diff(car_heading, road_fwd_heading)) < math.pi / 2

cur_from_jid = cur_road.sjid if _travelling_forward else cur_road.ejid
cur_next_jid = cur_road.other_end(cur_from_jid)
_startup_canonical_sl = _measure_actual_canonical_sl(
    cur_road, car_x, car_y, _travelling_forward)

travel_sl = _startup_canonical_sl if _travelling_forward else -_startup_canonical_sl
_cur_lane = min(range(NUM_OUR_LANES),
                key=lambda i: abs(travel_sl - OUR_LANE_OFFSETS[i]))

_target_lane = _cur_lane
_lane_t = 1.0
_lane_offset_from = OUR_LANE_OFFSETS[_cur_lane]
_lane_offset_to = OUR_LANE_OFFSETS[_cur_lane]
_avoidance_suppressed = 0
if USE_CAMERA:
    car.step(timestep)
    _update_lane_from_camera()

waypoints = cur_road.waypoints(cur_from_jid, _startup_canonical_sl)
final_phase = False
goal_x = goal_y = None
g_road = None
goal_jid = None
prev_left = False
while car.step(timestep) != -1:
    state = mouse.getState()
    left_down = (state.left == 1)
    if left_down and not prev_left:
        x, y, z = state.x, state.y, state.z
        if not (math.isnan(x) or math.isnan(y) or math.isnan(z)):
            goal_x, goal_y, g_road, goal_jid = snap_to_road_lane(
                x, y, cur_next_jid, cur_from_jid)
            place_marker(car, goal_x, goal_y, GROUND_Z)
            break
    prev_left = left_down

route = astar(cur_next_jid, goal_jid, forbidden=cur_from_jid)
route_idx = 0
wp_idx = _skip_behind_wps(waypoints, car_x, car_y, car_heading)
tx, ty = waypoints[wp_idx] if waypoints else (car_x, car_y)
CAM_POLL_STEPS = 1
step_count = 0

while car.step(timestep) != -1:
    dt = timestep / 1000.0
    step_count += 1
    with _speed_lock:
        current_speed = SPEED
    if step_count % CAM_POLL_STEPS == 0:
        _update_lane_from_camera()
    if _avoidance_suppressed > 0:
        _avoidance_suppressed -= 1

    if _fist_event.is_set():
        state = mouse.getState()
        left_down = (state.left == 1)
        if left_down and not prev_left:
            x, y, z = state.x, state.y, state.z
            if not (math.isnan(x) or math.isnan(y) or math.isnan(z)):
                goal_x, goal_y, g_road, goal_jid = snap_to_road_lane(
                    x, y, cur_next_jid, cur_from_jid)
                place_marker(car, goal_x, goal_y, GROUND_Z)
                route = astar(cur_next_jid, goal_jid, forbidden=cur_from_jid)
                route_idx = 0
                final_phase = False
                _fist_event.clear()
        prev_left = left_down
        continue

    with _gesture_lock:
        pending_directions = list(_gesture_queue)
        _gesture_queue.clear()

    for direction in pending_directions:
        ok = _try_lane_change(direction, source="gesture")
        if ok:
            _avoidance_suppressed = HUMAN_OVERRIDE_COOLDOWN
            break
    _handle_obstacle_avoidance()
    if _lane_t < 1.0:
        _lane_t = min(1.0, _lane_t + dt / LANE_CHANGE_DURATION)
        waypoints = cur_road.waypoints(cur_from_jid, _current_canonical_sl())
        wp_idx = _first_wp_ahead(waypoints, car_x, car_y, car_heading,
                                 LANE_CHANGE_LOOKAHEAD)
        if _lane_t >= 1.0:
            _cur_lane = _target_lane
            _lane_offset_from = OUR_LANE_OFFSETS[_cur_lane]
            _lane_offset_to = OUR_LANE_OFFSETS[_cur_lane]
            waypoints = cur_road.waypoints(cur_from_jid, _current_canonical_sl())
            wp_idx = _first_wp_ahead(waypoints, car_x, car_y, car_heading,
                                     LANE_CHANGE_LOOKAHEAD)

    state = mouse.getState()
    left_down = (state.left == 1)
    if left_down and not prev_left:
        x, y, z = state.x, state.y, state.z
        if not (math.isnan(x) or math.isnan(y) or math.isnan(z)):
            goal_x, goal_y, g_road, goal_jid = snap_to_road_lane(
                x, y, cur_next_jid, cur_from_jid)
            place_marker(car, goal_x, goal_y, GROUND_Z)
            route = astar(cur_next_jid, goal_jid, forbidden=cur_from_jid)
            route_idx = 0
            final_phase = False
    prev_left = left_down

    if hdist(car_x, car_y, goal_x, goal_y) < JUNC_RADIUS:
        prev_left = False
        while car.step(timestep) != -1:
            state = mouse.getState()
            left_down = (state.left == 1)
            if left_down and not prev_left:
                x, y, z = state.x, state.y, state.z
                if not (math.isnan(x) or math.isnan(y) or math.isnan(z)):
                    goal_x, goal_y, g_road, goal_jid = snap_to_road_lane(
                        x, y, cur_next_jid, cur_from_jid)
                    place_marker(car, goal_x, goal_y, GROUND_Z)
                    route = astar(cur_next_jid, goal_jid, forbidden=cur_from_jid)
                    route_idx = 0
                    final_phase = False
                    break
            prev_left = left_down

    while (wp_idx < len(waypoints) and
           hdist(car_x, car_y, *waypoints[wp_idx]) < WP_RADIUS):
        wp_idx += 1
    if wp_idx >= len(waypoints):
        arrived = cur_next_jid

        if final_phase or arrived == goal_jid or route_idx >= len(route) - 1:
            if not final_phase:
                final_phase = True

                dist_to_sj = hdist(car_x, car_y,
                                   junctions[g_road.sjid].x, junctions[g_road.sjid].y)
                dist_to_ej = hdist(car_x, car_y,
                                   junctions[g_road.ejid].x, junctions[g_road.ejid].y)
                arrived_end = g_road.sjid if dist_to_sj < dist_to_ej else g_road.ejid
                _travelling_forward = (arrived_end == g_road.sjid)
                final_sl = (OUR_LANE_OFFSETS[0] if _travelling_forward
                            else -OUR_LANE_OFFSETS[0])

                if _cur_lane != 0:
                    _lane_offset_from = OUR_LANE_OFFSETS[_cur_lane]
                    _lane_offset_to = OUR_LANE_OFFSETS[0]
                    _target_lane = 0
                    _lane_t = 0.0
                else:
                    _lane_t = 1.0

                waypoints = g_road.waypoints(arrived_end, final_sl)
                wp_idx = _skip_behind_wps(waypoints, car_x, car_y, car_heading)
                cur_from_jid = arrived_end
                cur_next_jid = g_road.other_end(arrived_end)
                cur_road = g_road
        else:
            dist_to_junc = hdist(car_x, car_y,
                                 junctions[arrived].x, junctions[arrived].y)
            time_to_junc = dist_to_junc / max(current_speed, 0.1)

            forced_nb = None
            if time_to_junc <= TURN_OVERRIDE_WINDOW:
                forced_nb, _ = _apply_speech_turn_override(
                    arrived, cur_from_jid, goal_jid)

            if forced_nb is not None:
                next_jid = forced_nb
                next_road = road_between.get((arrived, next_jid))
                if next_road is None:
                    forced_nb = None
                else:
                    route = astar(next_jid, goal_jid, forbidden=arrived)
                    route_idx = 0
                    cur_road = next_road
                    cur_from_jid = arrived
                    cur_next_jid = next_jid
                    _travelling_forward = (cur_from_jid == cur_road.sjid)
                    waypoints = cur_road.waypoints(cur_from_jid, _current_canonical_sl())
                    wp_idx = 0
            if forced_nb is None:
                route_idx += 1
                next_jid = route[route_idx]
                next_road = road_between.get((arrived, next_jid))
                if next_road is None:
                    _fist_event.set()
                    break
                cur_road = next_road
                cur_from_jid = arrived
                cur_next_jid = next_jid
                _travelling_forward = (cur_from_jid == cur_road.sjid)
                waypoints = cur_road.waypoints(cur_from_jid, _current_canonical_sl())
                wp_idx = 0

    if wp_idx < len(waypoints):
        tx, ty = waypoints[wp_idx]

    dx, dy = tx - car_x, ty - car_y
    if math.hypot(dx, dy) > 0.1:
        err = angle_diff(math.atan2(dy, dx), car_heading)
        car_heading += max(-MAX_STEER * dt, min(MAX_STEER * dt, err))

    car_x += current_speed * dt * math.cos(car_heading)
    car_y += current_speed * dt * math.sin(car_heading)

    car_node.getField("translation").setSFVec3f([car_x, car_y, GROUND_Z])
    car_node.getField("rotation").setSFRotation([0.0, 0.0, 1.0,
                                                 car_heading + BODY_OFFSET])
