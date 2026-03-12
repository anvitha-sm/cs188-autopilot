"""
controller.py — Webots autonomous car controller
=================================================

Runs inside Webots as the robot controller for a single car node.  The car
navigates a road network by click-to-goal, with three layers of real-time
human override:

  * Gesture control  — closed fist stops/resumes; thumbs up/down change lane
  * Speech control   — "faster" / "slower" adjust speed; "left" / "right"
                       override the next junction turn
  * Mouse click      — sets or resets the navigation goal at any time

Dependencies
------------
Install Python packages with:
    pip install -r requirements.txt

Webots R2023b or later must be installed from https://cyberbotics.com.
The two ML model files are downloaded automatically on first run:
    gesture_recognizer.task   (MediaPipe hand-gesture recogniser)
    efficientdet_lite0.tflite (MediaPipe object detector)

Entry point
-----------
Set this file as the controller field of your car robot node in Webots.
Webots calls the script directly; there is no if __name__ == "__main__" guard.

Coordinate conventions
----------------------
All positions are in Webots world coordinates (X right, Y forward, Z up).
"canonical_sl" (side-lateral offset) is the signed distance from a road's
centreline measured along the road's right-hand normal:
    positive  -> kerb side  (lane 0)
    negative  -> inner side (lane 1)
"""

# ===========================================================================
# Standard-library imports
# ===========================================================================
import collections
import heapq
import math
import os
import threading
import time
import urllib.request

# ===========================================================================
# Third-party imports
# ===========================================================================
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

# ===========================================================================
# Webots imports  (provided by the Webots installation, not pip)
# ===========================================================================
from controller import Supervisor, Mouse


# ===========================================================================
# Global tuning constants
# ===========================================================================

# --- Model files ------------------------------------------------------------
# MediaPipe gesture recogniser (float16, task bundle).
GESTURE_MODEL_PATH = "gesture_recognizer.task"
GESTURE_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer"
    "/gesture_recognizer/float16/1/gesture_recognizer.task"
)

# MediaPipe EfficientDet Lite object detector (float16, TFLite).
OBJ_MODEL_PATH = "efficientdet_lite0.tflite"
OBJ_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/object_detector"
    "/efficientdet_lite0/float16/1/efficientdet_lite0.tflite"
)

# --- Gesture recognition ----------------------------------------------------
# Minimum confidence score for a gesture to be accepted.
GESTURE_MIN_SCORE = 0.6

# Frames of a different (or absent) gesture required before the same thumb
# gesture can fire again.  Prevents a held thumbs-up from spamming commands.
GESTURE_GAP_FRAMES = 8

# --- Speech recognition -----------------------------------------------------
SPEECH_ENERGY_THRESHOLD = 200   # Microphone sensitivity (lower = more sensitive)
SPEECH_PAUSE_THRESHOLD  = 0.5   # Silence duration (s) that ends a phrase

# --- Speed ------------------------------------------------------------------
SPEED      = 5.0    # Initial speed (m/s)
SPEED_STEP = 1.0    # Amount each "faster" / "slower" command changes speed
SPEED_MIN  = 1.0    # Minimum permitted speed (m/s)
SPEED_MAX  = 13.0   # Maximum permitted speed (m/s)

# --- Steering and waypoints -------------------------------------------------
MAX_STEER  = 2.0    # Maximum heading-change rate (rad/s)
WP_RADIUS  = 3.5    # Distance (m) at which the car advances to the next waypoint
WP_SPACING = 2.0    # Distance (m) between generated waypoints along a road

# Distance ahead (m) to look when picking a waypoint during a lane change.
LANE_CHANGE_LOOKAHEAD = WP_RADIUS * 1.8

# Distance (m) from the goal point at which the car is considered to have arrived.
JUNC_RADIUS = 5.0

# Duration (s) of a lane-change manoeuvre (smoothstep interpolation).
LANE_CHANGE_DURATION = 0.25

# Vertical offset applied to the car body rotation field (Webots convention).
BODY_OFFSET = 0.0

# --- Road geometry ----------------------------------------------------------
ROAD_WIDTH = 21.5   # Total carriageway width (m), both directions combined
NUM_LANES  = 4      # Total number of lanes across the full road width

LANE_WIDTH = ROAD_WIDTH / NUM_LANES

# Lateral offsets from the road centreline for the two lanes we drive in.
# Index 0 = kerb lane (outermost), index 1 = inner lane.
OUR_LANE_OFFSETS = [
    ROAD_WIDTH / 2 - LANE_WIDTH / 2,    # lane 0: kerb side
    ROAD_WIDTH / 2 - LANE_WIDTH * 1.5,  # lane 1: inner side
]
NUM_OUR_LANES = len(OUR_LANE_OFFSETS)

# --- Camera / lane detection ------------------------------------------------
# Candidate device names tried in order when searching for the front camera.
CAM_CANDIDATES = ["front_camera", "camera", "Camera", "FrontCamera", "camera(1)"]

# Fraction of frame height above which the ROI is ignored (sky / horizon).
ROI_TOP_FRAC = 0.60

# --- Obstacle detection -----------------------------------------------------
# Object must occupy at least this fraction of frame height (from bottom) to
# be considered visible at all.
OBJ_SEE_FRAC = 0.30

# Object must reach this fraction to be treated as an imminent collision.
OBJ_CLOSE_FRAC = 0.50

# Minimum detection confidence score.
OBJ_MIN_SCORE = 0.30

# Consecutive frames without a detection before a lane is marked as clear.
MISS_NEEDED = 8

# Consecutive steps both lanes must appear blocked before handing over to
# the operator (guards against brief double-false-positives).
BOTH_BLOCKED_NEEDED = 5

# --- Human-override cooldown ------------------------------------------------
# Simulation steps after any human gesture or speech command during which
# the autonomous obstacle-avoidance logic is suppressed.  At a 32 ms timestep
# this is roughly 1.6 seconds -- long enough for the lane change to complete
# and the car to clear the obstacle before avoidance re-evaluates.
HUMAN_OVERRIDE_COOLDOWN = 50

# Seconds before reaching a junction within which a speech turn command
# is accepted as a turn override.
TURN_OVERRIDE_WINDOW = 6.0


# ===========================================================================
# Shared state protected by locks
# ===========================================================================

# Gesture subsystem
_fist_event    = threading.Event()          # Set while the car should be stopped
_gesture_lock  = threading.Lock()
_gesture_queue = collections.deque(maxlen=4)  # Pending lane-change directions

# Speed (modified by speech thread, read by main loop)
_speed_lock = threading.Lock()

# Speech turn override
_turn_lock       = threading.Lock()
_speech_turn_cmd = None   # "left" | "right" | None

# Obstacle detection results (written by MediaPipe callback, read by main loop)
_det_lock = threading.Lock()
_det_hit  = [False, False]  # [lane-0 blocked, lane-1 blocked]


# ===========================================================================
# Model download helper
# ===========================================================================

def _ensure_model(path, url):
    """
    Downloads the ML model file at url to path if it does not already exist.
    Prints progress messages so the operator knows what is happening on first
    run.
    """
    if not os.path.exists(path):
        print(f"[MODEL] Downloading {os.path.basename(path)} ...")
        urllib.request.urlretrieve(url, path)
        print(f"[MODEL] Saved to {path}")


# ===========================================================================
# Geometry helpers
# ===========================================================================

def angle_diff(a, b):
    """Returns the signed angular difference (a - b), normalised to (-pi, pi]."""
    d = a - b
    while d >  math.pi: d -= 2 * math.pi
    while d < -math.pi: d += 2 * math.pi
    return d


def hdist(ax, ay, bx, by):
    """Returns the straight-line horizontal distance between two 2-D points."""
    return math.hypot(ax - bx, ay - by)


def road_canonical_dir(road):
    """
    Returns the unit direction vector of a road segment pointing from its
    start junction to its end junction.
    """
    dx = road.ex - road.sx
    dy = road.ey - road.sy
    l  = math.hypot(dx, dy)
    return (dx / l, dy / l) if l > 1e-9 else (1.0, 0.0)


def road_right_normal(road):
    """
    Returns the unit vector perpendicular to the road, pointing to the right
    of the canonical travel direction (i.e. toward the kerb side).
    """
    fdx, fdy = road_canonical_dir(road)
    return fdy, -fdx


def apply_lateral(cx, cy, road, canonical_sl):
    """
    Shifts a centreline point (cx, cy) sideways by canonical_sl metres along
    the road's right-hand normal, returning the offset (x, y) position.
    """
    rx, ry = road_right_normal(road)
    return cx + rx * canonical_sl, cy + ry * canonical_sl


def _first_wp_ahead(wps, cx, cy, heading, min_fwd):
    """
    Returns the index of the first waypoint that is at least min_fwd metres
    directly ahead of (cx, cy) in the direction of heading.  Used when
    starting or resuming a lane change so the car targets a point well ahead
    rather than one it has already passed.  Falls back to the last waypoint
    if none qualify.
    """
    for wi, (wx, wy) in enumerate(wps):
        fwd = (wx - cx) * math.cos(heading) + (wy - cy) * math.sin(heading)
        if fwd >= min_fwd:
            return wi
    return max(0, len(wps) - 1)


def _skip_behind_wps(wps, cx, cy, heading):
    """
    Returns the index of the first waypoint that is strictly ahead of the
    car's current position, skipping any that are behind it.  Used when
    freshly generating waypoints for a new road segment so the car does not
    attempt to reverse back to a waypoint it has already passed.
    """
    for wi, (wx, wy) in enumerate(wps):
        fwd = (wx - cx) * math.cos(heading) + (wy - cy) * math.sin(heading)
        if fwd > 0:
            return wi
    return max(0, len(wps) - 1)


# ===========================================================================
# Road graph data structures
# ===========================================================================

class Junction:
    """
    A road junction (intersection) with a unique string ID and a 2-D world
    position.  Junctions are the nodes of the navigation graph.
    """
    def __init__(self, jid, x, y):
        self.id  = jid
        self.x   = x
        self.y   = y


class StraightRoad:
    """
    A straight road segment connecting two junctions.  Generates evenly
    spaced waypoints along the segment, laterally offset into the target lane.

    Attributes
    ----------
    curved         : always False; used to distinguish from CurvedRoad
    arc_cx/cy/radius : zero placeholders so road objects share an interface
    id             : unique string identifier from the Webots scene
    sjid / ejid    : start and end junction IDs
    sx/sy/ex/ey    : world coordinates of the start and end points
    """
    curved = False
    arc_cx = arc_cy = radius = 0.0

    def __init__(self, rid, sjid, ejid, sx, sy, ex, ey):
        self.id          = rid
        self.sjid        = sjid
        self.ejid        = ejid
        self.sx, self.sy = sx, sy
        self.ex, self.ey = ex, ey

    def other_end(self, jid):
        """Returns the junction ID at the opposite end of this segment from jid."""
        return self.ejid if jid == self.sjid else self.sjid

    def centreline_point(self, t):
        """
        Returns the world position at fractional distance t along the
        centreline, where t=0 is the start junction and t=1 is the end.
        """
        return (
            self.sx + (self.ex - self.sx) * t,
            self.sy + (self.ey - self.sy) * t,
        )

    def waypoints(self, from_jid, canonical_sl):
        """
        Generates a list of (x, y) world positions spaced WP_SPACING metres
        apart along this segment, travelling away from from_jid and offset
        laterally by canonical_sl metres from the centreline.
        """
        t0, t1  = (0.1, 0.9) if from_jid == self.sjid else (0.9, 0.1)
        seg_len = math.hypot(self.ex - self.sx, self.ey - self.sy)
        n       = max(2, int(seg_len / WP_SPACING))
        wps     = []
        for i in range(1, n + 1):
            t      = t0 + (t1 - t0) * i / n
            cx, cy = self.centreline_point(t)
            wps.append(apply_lateral(cx, cy, self, canonical_sl))
        return wps


class CurvedRoad:
    """
    A curved (circular arc) road segment connecting two junctions.  The arc
    is defined by its centre point and radius.  Generates evenly spaced
    waypoints along the arc, laterally offset into the target lane.

    Attributes
    ----------
    curved          : always True; used to distinguish from StraightRoad
    id              : unique string identifier from the Webots scene
    sjid / ejid     : start and end junction IDs
    sx/sy/ex/ey     : world coordinates of the start and end points
    arc_cx / arc_cy : world coordinates of the arc centre
    radius          : arc radius (m)
    """
    curved = True

    def __init__(self, rid, sjid, ejid, sx, sy, ex, ey, arc_cx, arc_cy, radius):
        self.id          = rid
        self.sjid        = sjid
        self.ejid        = ejid
        self.sx, self.sy = sx, sy
        self.ex, self.ey = ex, ey
        self.arc_cx      = arc_cx
        self.arc_cy      = arc_cy
        self.radius      = radius

    def other_end(self, jid):
        """Returns the junction ID at the opposite end of this segment from jid."""
        return self.ejid if jid == self.sjid else self.sjid

    def waypoints(self, from_jid, canonical_sl):
        """
        Generates a list of (x, y) world positions spaced WP_SPACING metres
        apart along the arc, travelling away from from_jid and offset
        laterally by canonical_sl metres from the arc centreline.
        """
        if from_jid == self.sjid:
            a0 = math.atan2(self.sy - self.arc_cy, self.sx - self.arc_cx)
            a1 = math.atan2(self.ey - self.arc_cy, self.ex - self.arc_cx)
        else:
            a0 = math.atan2(self.ey - self.arc_cy, self.ex - self.arc_cx)
            a1 = math.atan2(self.sy - self.arc_cy, self.sx - self.arc_cx)

        da      = angle_diff(a1, a0)
        arc_len = abs(da) * self.radius
        n       = max(3, int(arc_len / WP_SPACING))
        wps     = []
        for i in range(1, n + 1):
            t     = 0.1 + (i / n) * 0.8
            angle = a0 + da * t
            cx    = self.arc_cx + self.radius * math.cos(angle)
            cy    = self.arc_cy + self.radius * math.sin(angle)
            wps.append(apply_lateral(cx, cy, self, canonical_sl))
        return wps


# ===========================================================================
# Road graph -- build and query
# ===========================================================================

# Module-level graph tables populated by build_graph()
junctions    = {}   # jid  -> Junction
roads        = {}   # rid  -> StraightRoad | CurvedRoad
graph        = {}   # jid  -> [neighbour jid, ...]
road_between = {}   # (jid_a, jid_b) -> road


def build_graph(car_supervisor):
    """
    Walks the Webots scene tree rooted at car_supervisor and populates the
    junctions, roads, graph, and road_between module-level dictionaries.

    Crossroad and RoadIntersection nodes become Junction objects.
    StraightRoadSegment and CurvedRoadSegment nodes become StraightRoad or
    CurvedRoad objects.  graph maps each junction ID to its list of
    reachable neighbour IDs.
    """
    root = car_supervisor.getRoot()
    kids = root.getField("children")

    for i in range(kids.getCount()):
        node = kids.getMFNode(i)
        if node.getTypeName() in ("Crossroad", "RoadIntersection"):
            jid            = node.getField("id").getSFString()
            p              = node.getField("translation").getSFVec3f()
            junctions[jid] = Junction(jid, p[0], p[1])
            graph[jid]     = []

    for i in range(kids.getCount()):
        node = kids.getMFNode(i)
        typ  = node.getTypeName()
        if typ not in ("StraightRoadSegment", "CurvedRoadSegment"):
            continue

        rid  = node.getField("id").getSFString()
        sjid = node.getField("startJunction").getSFString()
        ejid = node.getField("endJunction").getSFString()
        sj, ej = junctions[sjid], junctions[ejid]

        if typ == "CurvedRoadSegment":
            p      = node.getField("translation").getSFVec3f()
            radius = node.getField("curvatureRadius").getSFFloat()
            road   = CurvedRoad(rid, sjid, ejid,
                                sj.x, sj.y, ej.x, ej.y,
                                p[0], p[1], radius)
        else:
            road = StraightRoad(rid, sjid, ejid, sj.x, sj.y, ej.x, ej.y)

        roads[rid]                 = road
        road_between[(sjid, ejid)] = road
        road_between[(ejid, sjid)] = road
        if ejid not in graph[sjid]: graph[sjid].append(ejid)
        if sjid not in graph[ejid]: graph[ejid].append(sjid)


def closest_road(px, py):
    """
    Finds the road segment whose centreline is nearest to the point (px, py).

    For straight roads, uses perpendicular projection onto the segment.
    For curved roads, uses the difference between the radial distance from
    the arc centre and the arc radius.

    Returns the nearest StraightRoad or CurvedRoad object.
    """
    best, best_d = None, 1e9
    for road in roads.values():
        if road.curved:
            d = abs(hdist(px, py, road.arc_cx, road.arc_cy) - road.radius)
        else:
            dx, dy = road.ex - road.sx, road.ey - road.sy
            l2     = dx * dx + dy * dy
            if l2 < 1e-9:
                continue
            t = max(0.0, min(1.0,
                ((px - road.sx) * dx + (py - road.sy) * dy) / l2))
            d = hdist(px, py, road.sx + dx * t, road.sy + dy * t)
        if d < best_d:
            best_d, best = d, road
    return best


def astar(start, goal, forbidden=None):
    """
    Finds the shortest path between two junction IDs using A* with Euclidean
    distance as the heuristic.

    The optional forbidden parameter prevents the search from leaving start
    via that specific neighbour, which avoids an immediate U-turn at the car's
    current position.

    Returns a list of junction IDs from start to goal (inclusive).
    """
    if start == goal:
        return [start]

    frontier  = [(0.0, start)]
    came_from = {start: None}
    g_cost    = {start: 0.0}

    while frontier:
        _, cur = heapq.heappop(frontier)
        if cur == goal:
            break
        for nb in graph.get(cur, []):
            if cur == start and nb == forbidden:
                continue
            nc = g_cost[cur] + hdist(junctions[cur].x, junctions[cur].y,
                                     junctions[nb].x,  junctions[nb].y)
            if nb not in g_cost or nc < g_cost[nb]:
                g_cost[nb]    = nc
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


def turn_label(from_jid, via_jid, to_jid):
    """
    Classifies the turn at via_jid when travelling from from_jid toward
    to_jid.  Uses the signed angle between the incoming and outgoing vectors.

    Returns one of: "STRAIGHT", "LEFT", "RIGHT", or "U-TURN".
    """
    in_dx  = junctions[via_jid].x - junctions[from_jid].x
    in_dy  = junctions[via_jid].y - junctions[from_jid].y
    out_dx = junctions[to_jid].x  - junctions[via_jid].x
    out_dy = junctions[to_jid].y  - junctions[via_jid].y
    angle  = math.atan2(in_dx * out_dy - in_dy * out_dx,
                        in_dx * out_dx  + in_dy * out_dy)
    if abs(angle) < 0.4:  return "STRAIGHT"
    if abs(angle) > 2.5:  return "U-TURN"
    return "LEFT" if angle > 0 else "RIGHT"


def snap_to_road_lane(click_x, click_y, ref_next_jid, ref_from_jid):
    """
    Projects a mouse-click position onto the nearest road centreline, then
    shifts the result laterally into the correct driveable lane based on which
    side of the centreline the click landed.

    Also determines which end of the goal road to approach by running A* from
    the car's current next junction to each end of the goal road, and picking
    the cheaper option.

    Returns (snapped_x, snapped_y, road, goal_junction_id).
    """
    g_road = closest_road(click_x, click_y)

    if g_road.curved:
        angle  = math.atan2(click_y - g_road.arc_cy, click_x - g_road.arc_cx)
        snap_x = g_road.arc_cx + g_road.radius * math.cos(angle)
        snap_y = g_road.arc_cy + g_road.radius * math.sin(angle)
    else:
        dx, dy = g_road.ex - g_road.sx, g_road.ey - g_road.sy
        l2     = dx * dx + dy * dy
        t      = max(0.1, min(0.9,
            ((click_x - g_road.sx) * dx + (click_y - g_road.sy) * dy) / l2))
        snap_x = g_road.sx + dx * t
        snap_y = g_road.sy + dy * t

    rx, ry   = road_right_normal(g_road)
    side_dot = (click_x - snap_x) * rx + (click_y - snap_y) * ry
    goal_sl  = OUR_LANE_OFFSETS[0] if side_dot >= 0 else -OUR_LANE_OFFSETS[0]
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


# ===========================================================================
# Goal marker (Webots scene object)
# ===========================================================================

_marker_node = None


def place_marker(car_supervisor, x, y, z):
    """
    Inserts a glowing orange sphere into the Webots scene at (x, y, z + 1.5)
    to mark the current navigation goal.  Calls remove_marker() first so only
    one marker exists at a time.
    """
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
    """Removes the current goal marker sphere from the Webots scene, if present."""
    global _marker_node
    if _marker_node is not None:
        _marker_node.remove()
        _marker_node = None


# ===========================================================================
# Gesture recognition -- background daemon thread
# ===========================================================================

# Edge-detection state (private to gesture subsystem)
_last_gesture_name = None
_same_gesture_ct   = 0


def _gesture_callback(result, output_image, timestamp_ms):
    """
    Receives each gesture recognition result from MediaPipe and translates it
    into driving commands.

    A Closed_Fist sets the stop event on the first frame it appears and clears
    it when the hand opens again.  Thumb_Up (-1, toward kerb) and Thumb_Down
    (+1, toward inner) queue a lane-change direction, but only on the rising
    edge of a new gesture so that holding the hand still does not repeat the
    command.  After holding the same gesture for GESTURE_GAP_FRAMES * 8
    frames the command re-fires once, allowing the operator to repeat the
    gesture without dropping their hand.
    """
    global _last_gesture_name, _same_gesture_ct

    gesture_name = None
    if result.gestures:
        top = result.gestures[0][0]
        if top.score > GESTURE_MIN_SCORE:
            gesture_name = top.category_name

    if gesture_name == "Closed_Fist":
        if not _fist_event.is_set():
            print("[GESTURE] Closed_Fist -> STOPPED")
            _fist_event.set()
    else:
        if _fist_event.is_set():
            print("[GESTURE] Hand open -> RESUMED")
            _fist_event.clear()

    if gesture_name in ("Thumb_Up", "Thumb_Down"):
        if gesture_name != _last_gesture_name:
            _same_gesture_ct = 1
            direction = -1 if gesture_name == "Thumb_Up" else +1
            label     = "kerb (lane 0)" if direction == -1 else "inner (lane 1)"
            with _gesture_lock:
                _gesture_queue.append(direction)
            print(f"[GESTURE] {gesture_name} -> {label} queued")
        else:
            _same_gesture_ct += 1
            if _same_gesture_ct % (GESTURE_GAP_FRAMES * 8) == 0:
                direction = -1 if gesture_name == "Thumb_Up" else +1
                with _gesture_lock:
                    _gesture_queue.append(direction)
                print(f"[GESTURE] {gesture_name} held -> re-queued")
    else:
        if _last_gesture_name in ("Thumb_Up", "Thumb_Down"):
            _same_gesture_ct = 0

    _last_gesture_name = gesture_name


def _gesture_thread():
    """
    Opens the first available webcam (indices 0-3) and feeds frames into the
    MediaPipe gesture recogniser in live-stream mode.  Recognition results are
    delivered asynchronously via _gesture_callback.  If no webcam is found the
    thread exits silently and gesture control is unavailable for the session.
    """
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
                print(f"[GESTURE] Webcam opened on index {idx}")
                break
        test.release()

    if cap is None:
        print("[GESTURE] ERROR: No webcam found. Gesture control disabled.")
        return

    print("[GESTURE] Ready -- closed-fist stop/resume, thumb-up kerb lane, thumb-down inner lane")
    with mp_vision.GestureRecognizer.create_from_options(options) as recognizer:
        timestamp_ms = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            frame_rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms += 33
            recognizer.recognize_async(mp_image, timestamp_ms)
            time.sleep(0.01)
    cap.release()


# ===========================================================================
# Speech recognition -- background daemon thread
# ===========================================================================

def _speech_thread():
    """
    Listens continuously for spoken commands using the system default
    microphone.

    Recognised commands:
        "faster"  -- increases SPEED by SPEED_STEP (up to SPEED_MAX)
        "slower"  -- decreases SPEED by SPEED_STEP (down to SPEED_MIN)
        "left"    -- queues a left-turn override for the next junction
        "right"   -- queues a right-turn override for the next junction

    Requires the SpeechRecognition package and an active internet connection
    for the Google Web Speech API.  If either is unavailable the thread exits
    silently and speech control is disabled for the session.
    """
    global SPEED, _speech_turn_cmd

    try:
        import speech_recognition as sr
    except ImportError:
        print("[SPEECH] 'SpeechRecognition' not installed. Speech control disabled.")
        return

    r = sr.Recognizer()
    r.energy_threshold         = SPEECH_ENERGY_THRESHOLD
    r.dynamic_energy_threshold = False
    r.pause_threshold          = SPEECH_PAUSE_THRESHOLD

    try:
        mic = sr.Microphone()
    except Exception as e:
        print(f"[SPEECH] No microphone ({e}). Disabled.")
        return

    print("[SPEECH] Calibrating microphone...")
    with mic as source:
        r.adjust_for_ambient_noise(source, duration=1)
    print("[SPEECH] Ready -- say 'faster', 'slower', 'left', or 'right'.")

    while True:
        try:
            with mic as source:
                audio = r.listen(source, timeout=5, phrase_time_limit=3)
            text = r.recognize_google(audio).lower()
            print(f"[SPEECH] Heard: '{text}'")

            with _speed_lock:
                if "faster" in text:
                    SPEED = min(SPEED_MAX, round(SPEED + SPEED_STEP, 1))
                    print(f"[SPEECH] speed up -> {SPEED}")
                elif "slower" in text:
                    SPEED = max(SPEED_MIN, round(SPEED - SPEED_STEP, 1))
                    print(f"[SPEECH] speed down -> {SPEED}")

            with _turn_lock:
                if "left" in text and "right" not in text:
                    _speech_turn_cmd = "left"
                    print("[SPEECH] turn override: LEFT queued")
                elif "right" in text and "left" not in text:
                    _speech_turn_cmd = "right"
                    print("[SPEECH] turn override: RIGHT queued")

        except (sr.WaitTimeoutError, sr.UnknownValueError):
            pass
        except sr.RequestError as e:
            print(f"[SPEECH] API error: {e}")
            time.sleep(2)
        except Exception as e:
            print(f"[SPEECH] Error: {e}")
            time.sleep(1)


# ===========================================================================
# Camera -- lane detection
# ===========================================================================

def _bgr_from_webots(raw_bytes, w, h):
    """
    Converts raw BGRA bytes from a Webots camera image into a standard
    three-channel BGR numpy array suitable for OpenCV processing.
    """
    arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((h, w, 4))
    return arr[:, :, :3]


def _detect_lane_from_camera(bgr):
    """
    Analyses a BGR camera frame to determine which of the two driveable lanes
    the car is currently in, and whether the yellow centre line is blocking
    the path ahead.

    The function crops to the lower ROI_TOP_FRAC of the frame (discarding sky
    and horizon) and builds per-column histograms of yellow and white pixels.
    The yellow centre-line position relative to the image centre indicates
    which side of the road we are on; white kerb lines confirm this when the
    yellow line is absent.

    Returns (lane_index, yellow_blocked):
        lane_index     -- 0 = kerb lane, 1 = inner lane, None if uncertain
        yellow_blocked -- True when the yellow line is on the right (our path)
    """
    h, w     = bgr.shape[:2]
    roi      = bgr[int(h * ROI_TOP_FRAC):, :]
    rh, rw   = roi.shape[:2]
    scan     = roi[rh // 2:, :]
    sh       = scan.shape[0]
    scan_hsv = cv2.cvtColor(scan, cv2.COLOR_BGR2HSV)

    yellow_hist = np.sum(
        cv2.inRange(scan_hsv,
                    np.array([15,  80,  80]),
                    np.array([40, 255, 255])),
        axis=0).astype(float)
    white_hist = np.sum(
        cv2.inRange(scan_hsv,
                    np.array([0,    0, 160]),
                    np.array([180, 40, 255])),
        axis=0).astype(float)

    # Smooth histograms with a 15-pixel box filter to reduce noise
    k           = np.ones(15) / 15
    yellow_hist = np.convolve(yellow_hist, k, mode="same")
    white_hist  = np.convolve(white_hist,  k, mode="same")

    cx              = rw / 2.0
    yellow_col      = None
    left_white_col  = None
    right_white_col = None

    if yellow_hist.max() > sh * 5:
        yellow_col = float(np.argmax(yellow_hist))
    if white_hist[:int(cx)].max() > sh * 3:
        left_white_col = float(np.argmax(white_hist[:int(cx)]))
    if white_hist[int(cx):].max() > sh * 3:
        right_white_col = float(np.argmax(white_hist[int(cx):])) + cx

    our_lane       = None
    yellow_blocked = False

    if yellow_col is not None:
        if yellow_col / rw < 0.25:
            our_lane = 0
        else:
            our_lane       = 1
            yellow_blocked = True

    if our_lane is None:
        if   right_white_col is not None and left_white_col is None:
            our_lane = 1
        elif left_white_col  is not None and right_white_col is None:
            our_lane = 0
        elif left_white_col  is not None and right_white_col is not None:
            our_lane = 0 if abs(right_white_col - cx) > abs(left_white_col - cx) else 1

    return our_lane, yellow_blocked


# ===========================================================================
# Camera -- obstacle detection
# ===========================================================================

# Monotonically increasing timestamp fed to the async object detector
_obj_ts = 0


def _obj_det_callback(result, output_image, timestamp_ms):
    """
    Receives each asynchronous object detection result from MediaPipe
    EfficientDet and updates the shared _det_hit flags.

    For every detected object above OBJ_MIN_SCORE whose bounding-box bottom
    edge is below OBJ_SEE_FRAC, the function assigns the object to lane 0 or
    lane 1 based on whether its horizontal centre is in the left or right half
    of the frame (flipped when travelling in reverse).  Objects whose bottom
    edge reaches OBJ_CLOSE_FRAC are marked as imminent and set the hit flag
    for that lane.
    """
    global _det_hit
    if result is None:
        return

    h, w  = output_image.height, output_image.width
    mid_x = w / 2
    hit   = [False, False]

    for det in result.detections:
        score = det.categories[0].score if det.categories else 0.0
        if score < OBJ_MIN_SCORE:
            continue

        bb         = det.bounding_box
        box_bottom = (bb.origin_y + bb.height) / h
        box_cx     = bb.origin_x + bb.width / 2

        if box_bottom < OBJ_SEE_FRAC:
            continue

        # Lane assignment depends on travel direction
        if _travelling_forward:
            z = 0 if box_cx > mid_x else 1
        else:
            z = 0 if box_cx < mid_x else 1

        lbl  = ["kerb(0)", "inner(1)"][z]
        name = det.categories[0].category_name if det.categories else "?"

        if box_bottom >= OBJ_CLOSE_FRAC:
            hit[z] = True
            print(f"[OBSTACLE] Lane {lbl} '{name}' score={score:.2f} "
                  f"bottom={box_bottom:.2f} -> IMMINENT")
        else:
            print(f"[OBSTACLE] Lane {lbl} '{name}' score={score:.2f} "
                  f"bottom={box_bottom:.2f} -- approaching")

    with _det_lock:
        _det_hit[0] = hit[0]
        _det_hit[1] = hit[1]


def _obstacle_detect(bgr):
    """
    Submits a BGR camera frame to the async object detector and returns the
    most recently completed hit flags as a two-element list [lane0, lane1].
    Because detection is asynchronous, the returned values may reflect the
    previous frame rather than the one just submitted.
    """
    global _obj_ts
    _obj_ts += 33
    rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    _obj_detector.detect_async(mp_img, _obj_ts)
    with _det_lock:
        return [_det_hit[0], _det_hit[1]]


def _update_zone_blocked_state(hit):
    """
    Updates the persistent lane-blocked flags from the latest detection hit
    results.  A lane is marked blocked immediately on the first hit frame, but
    only cleared after MISS_NEEDED consecutive frames with no detection.  This
    hysteresis prevents a single missed frame from prematurely un-blocking a
    lane that still contains an obstacle.
    """
    for z in range(2):
        lbl = ["kerb(0)", "inner(1)"][z]
        if hit[z]:
            _near_miss_ct[z] = 0
            if not _lane_near_blocked[z]:
                _lane_near_blocked[z] = True
                print(f"[OBSTACLE] Lane {lbl} BLOCKED")
        else:
            _near_miss_ct[z] += 1
            if _lane_near_blocked[z] and _near_miss_ct[z] >= MISS_NEEDED:
                _lane_near_blocked[z] = False
                print(f"[OBSTACLE] Lane {lbl} clear")


def _update_lane_from_camera():
    """
    Reads the latest camera frame and runs both lane detection and obstacle
    detection.  Updates the module-level _camera_lane, _yellow_blocked, and
    _lane_near_blocked state used by the main loop.  Does nothing if no camera
    is available.
    """
    global _camera_lane, _yellow_blocked
    if not USE_CAMERA:
        return
    raw = cam.getImage()
    if raw is None:
        return
    bgr            = _bgr_from_webots(raw, CAM_W, CAM_H)
    lane, yblocked = _detect_lane_from_camera(bgr)
    if lane is not None:
        _camera_lane    = lane
        _yellow_blocked = yblocked
    hit = _obstacle_detect(bgr)
    _update_zone_blocked_state(hit)


# ===========================================================================
# Lane-change state and helpers
# ===========================================================================

def _current_canonical_sl():
    """
    Returns the car's current lateral offset from the road centreline in the
    canonical right-hand coordinate system.

    During a lane change, interpolates smoothly between _lane_offset_from and
    _lane_offset_to using a smoothstep curve parameterised by _lane_t (0->1).
    Outside of a transition, returns the committed offset for _cur_lane.
    The result is negated when the car is travelling in the reverse direction.
    """
    if _lane_t >= 1.0:
        offset = OUR_LANE_OFFSETS[_cur_lane]
    else:
        s      = _lane_t * _lane_t * (3.0 - 2.0 * _lane_t)
        offset = _lane_offset_from + s * (_lane_offset_to - _lane_offset_from)
    return offset if _travelling_forward else -offset


def _measure_actual_canonical_sl(road, px, py, travelling_fwd):
    """
    Measures how far (px, py) sits to the right of the road centreline in the
    road's canonical coordinate system.  Positive = kerb side, negative =
    inner side.  Handles both straight and curved roads.  Used at startup to
    determine which lane the car spawned in.
    """
    if not road.curved:
        dx, dy = road.ex - road.sx, road.ey - road.sy
        l2     = dx * dx + dy * dy
        t      = ((px - road.sx) * dx + (py - road.sy) * dy) / max(l2, 1e-9)
        proj_x = road.sx + dx * t
        proj_y = road.sy + dy * t
        l      = math.sqrt(l2)
        rx, ry = dy / l, -dx / l
        return (px - proj_x) * rx + (py - proj_y) * ry
    else:
        dist = math.hypot(px - road.arc_cx, py - road.arc_cy)
        a0   = math.atan2(road.sy - road.arc_cy, road.sx - road.arc_cx)
        a1   = math.atan2(road.ey - road.arc_cy, road.ex - road.arc_cx)
        da   = angle_diff(a1, a0)
        sign = 1 if da > 0 else -1
        return sign * (dist - road.radius)


def _try_lane_change(direction, source="gesture"):
    """
    Attempts to begin a lane change in the given direction
    (-1 toward kerb, +1 toward inner lane).

    Validation rules applied in order:
      1. The target lane must be within [0, NUM_OUR_LANES).
      2. If a transition to that exact lane is already in progress, the
         request is ignored.
      3. Autonomous avoidance is blocked from entering a lane that the
         obstacle detector has marked as blocked.
      4. A human gesture into a blocked lane executes with a warning -- the
         human is assumed to have information the detector does not.

    On approval, sets up the smoothstep transition using the car's actual
    current offset (which may be mid-transition) as the start point, and
    regenerates waypoints for the new lateral offset.

    Returns True if the lane change was accepted and started.
    """
    global _cur_lane, _target_lane, _lane_t
    global _lane_offset_from, _lane_offset_to
    global waypoints, wp_idx

    base_lane = _target_lane
    new_lane  = base_lane + direction

    if new_lane < 0 or new_lane >= NUM_OUR_LANES:
        side = "kerb" if direction < 0 else "inner"
        print(f"[LANE] [{source}] Already at {side} boundary (lane {base_lane}).")
        return False

    if _lane_t < 1.0 and new_lane == _target_lane:
        print(f"[LANE] [{source}] Already transitioning to lane {new_lane} -- ignored.")
        return False

    if source == "avoidance" and _lane_near_blocked[new_lane]:
        print(f"[LANE] [avoidance] Target lane {new_lane} blocked -- dodge aborted.")
        return False

    if source == "gesture" and _lane_near_blocked[new_lane]:
        print(f"[LANE] [gesture] Lane {new_lane} has obstacle -- executing human override.")

    # Start from the car's actual current lateral position (may be mid-transition)
    if _lane_t >= 1.0:
        _lane_offset_from = OUR_LANE_OFFSETS[_cur_lane]
    else:
        s = _lane_t * _lane_t * (3.0 - 2.0 * _lane_t)
        _lane_offset_from = _lane_offset_from + (_lane_offset_to - _lane_offset_from) * s

    _lane_offset_to = OUR_LANE_OFFSETS[new_lane]
    _target_lane    = new_lane
    _lane_t         = 0.0

    side = "kerb" if new_lane == 0 else "inner"
    print(f"[LANE] [{source}] lane {base_lane} -> {new_lane} ({side})")
    return True


# ===========================================================================
# Operator handover
# ===========================================================================

def _stop_for_human(reason):
    """
    Halts autonomous driving by setting the fist stop event.  Prints the
    reason and instructs the operator to provide a new goal click or open
    their hand to resume.
    """
    print(f"[HANDOFF] STOP: {reason}")
    print("[HANDOFF] Waiting for a new goal click or open-hand gesture to resume.")
    _fist_event.set()


# ===========================================================================
# Obstacle avoidance logic
# ===========================================================================

_near_miss_ct      = [0, 0]          # Consecutive no-detection frames per lane
_lane_near_blocked = [False, False]  # Persistent blocked state per lane
_dodged            = False           # Lane index the car moved away from, or False
_both_blocked_ct   = 0               # Consecutive steps both lanes appeared blocked


def _handle_obstacle_avoidance():
    """
    Checks whether the current lane is blocked and, if so, attempts to dodge
    into the adjacent lane.

    The function does nothing when:
      - A lane change is already in progress (_lane_t < 1.0)
      - The car is on the final approach to the goal (final_phase is True)
      - A human command was issued within HUMAN_OVERRIDE_COOLDOWN steps

    If both lanes appear blocked for BOTH_BLOCKED_NEEDED consecutive steps,
    control is handed to the operator via _stop_for_human.  The _dodged flag
    prevents the car from trying to dodge back into a lane it already fled.
    """
    global _dodged, _both_blocked_ct

    if _lane_t < 1.0:
        return
    if final_phase:
        return
    if _avoidance_suppressed > 0:
        return

    adj_lane = 1 - _cur_lane

    # Clear dodged memory once the vacated lane is confirmed free
    if _dodged is not False and not _lane_near_blocked[_dodged]:
        print(f"[OBSTACLE] Lane {_dodged} now clear -- _dodged reset")
        _dodged = False

    if not _lane_near_blocked[_cur_lane]:
        _both_blocked_ct = 0
        return

    if _lane_near_blocked[adj_lane]:
        _both_blocked_ct += 1
        if _both_blocked_ct >= BOTH_BLOCKED_NEEDED:
            _stop_for_human("Both lanes blocked -- full stop, human required")
        else:
            print(f"[OBSTACLE] Both blocked ({_both_blocked_ct}/{BOTH_BLOCKED_NEEDED})"
                  " -- waiting to confirm")
        return

    _both_blocked_ct = 0

    if _dodged is not False and adj_lane == _dodged and _lane_near_blocked[_dodged]:
        _stop_for_human(f"Lane {adj_lane} still blocked -- full stop")
        return

    print(f"[OBSTACLE] Lane {_cur_lane} blocked -> dodging to lane {adj_lane}")
    ok = _try_lane_change(adj_lane - _cur_lane, source="avoidance")
    if not ok:
        _stop_for_human(f"obstacle in lane {_cur_lane}, lane change failed")
    else:
        _dodged = _cur_lane


# ===========================================================================
# Speech turn-override helper
# ===========================================================================

def _apply_speech_turn_override(arrived_jid, from_jid, goal_jid_ref):
    """
    Consumes the pending speech turn command (_speech_turn_cmd) and searches
    the neighbours of arrived_jid for one whose turn direction matches the
    spoken command.  If multiple neighbours match, the one with the sharpest
    turn angle is selected.

    Returns (next_junction_id, goal_jid_ref) on success, or (None, None) if
    no matching turn exists at the current junction.
    """
    global _speech_turn_cmd

    with _turn_lock:
        cmd              = _speech_turn_cmd
        _speech_turn_cmd = None

    if cmd is None:
        return None, None

    best_nb    = None
    best_angle = None

    for nb in graph.get(arrived_jid, []):
        if nb == from_jid:
            continue
        lbl = turn_label(from_jid, arrived_jid, nb)
        if lbl.upper() == cmd.upper():
            in_dx  = junctions[arrived_jid].x - junctions[from_jid].x
            in_dy  = junctions[arrived_jid].y - junctions[from_jid].y
            out_dx = junctions[nb].x - junctions[arrived_jid].x
            out_dy = junctions[nb].y - junctions[arrived_jid].y
            a = math.atan2(in_dx * out_dy - in_dy * out_dx,
                           in_dx * out_dx  + in_dy * out_dy)
            if best_nb is None or abs(a) > abs(best_angle):
                best_nb    = nb
                best_angle = a

    if best_nb is None:
        print(f"[SPEECH] No {cmd} turn available at {arrived_jid} -- ignoring.")
        return None, None

    print(f"[SPEECH] Overriding nav: turning {cmd.upper()} -> {best_nb}")
    return best_nb, goal_jid_ref


# ===========================================================================
# Initialisation
# ===========================================================================

# Download ML models if not already present
_ensure_model(GESTURE_MODEL_PATH, GESTURE_MODEL_URL)
_ensure_model(OBJ_MODEL_PATH,     OBJ_MODEL_URL)

# Start background threads before connecting to Webots so they are ready
# as soon as the simulation begins stepping
threading.Thread(target=_gesture_thread, daemon=True).start()
threading.Thread(target=_speech_thread,  daemon=True).start()

# Connect to the Webots simulation
car      = Supervisor()
timestep = int(car.getBasicTimeStep())
car_node = car.getSelf()
GROUND_Z = car_node.getPosition()[2]

# Locate and enable the front camera
cam = None
for _name in CAM_CANDIDATES:
    _dev = car.getDevice(_name)
    if _dev is not None:
        cam = _dev
        print(f"[CAMERA] Found camera as '{_name}'")
        break

if cam is None:
    print("[CAMERA] Known names not found -- scanning all devices...")
    for _i in range(car.getNumberOfDevices()):
        _dev = car.getDeviceByIndex(_i)
        if _dev.getNodeType() == car.CAMERA:
            cam = _dev
            print(f"[CAMERA] Found by scan: '{_dev.getName()}'")
            break

if cam is None:
    print("[CAMERA] WARNING: No camera found. Lane detection + obstacle avoidance disabled.")
    USE_CAMERA = False
    CAM_W = CAM_H = 0
else:
    cam.enable(timestep)
    USE_CAMERA = True
    CAM_W      = cam.getWidth()
    CAM_H      = cam.getHeight()
    print(f"[CAMERA] Enabled at {CAM_W}x{CAM_H}")

_camera_lane    = None
_yellow_blocked = False

# Build the async MediaPipe object detector
_obj_options = ObjectDetectorOptions(
    base_options=mp_python.BaseOptions(model_asset_path=OBJ_MODEL_PATH),
    running_mode=RunningMode.LIVE_STREAM,
    max_results=10,
    score_threshold=OBJ_MIN_SCORE,
    result_callback=_obj_det_callback,
)
_obj_detector = ObjectDetector.create_from_options(_obj_options)
print("[OBSTACLE] MediaPipe object detector ready.")

# Enable mouse click input for goal selection
mouse = car.getMouse()
mouse.enable(timestep)
mouse.enable3dPosition()
print("[INPUT] Click on the road to set the goal.")

# Build the road navigation graph from the Webots scene
build_graph(car)

# Determine the car's starting position and which lane it is in
pos          = car_node.getPosition()
car_x, car_y = pos[0], pos[1]

cur_road = closest_road(car_x, car_y)

_rot        = car_node.getField("rotation").getSFRotation()
car_heading = _rot[3] if _rot[2] > 0 else -_rot[3]

fdx, fdy         = road_canonical_dir(cur_road)
road_fwd_heading = math.atan2(fdy, fdx)
_travelling_forward = abs(angle_diff(car_heading, road_fwd_heading)) < math.pi / 2

cur_from_jid = cur_road.sjid if _travelling_forward else cur_road.ejid
cur_next_jid = cur_road.other_end(cur_from_jid)

_startup_canonical_sl = _measure_actual_canonical_sl(
    cur_road, car_x, car_y, _travelling_forward)

travel_sl = _startup_canonical_sl if _travelling_forward else -_startup_canonical_sl
_cur_lane = min(range(NUM_OUR_LANES),
                key=lambda i: abs(travel_sl - OUR_LANE_OFFSETS[i]))

# Lane-change transition state
_target_lane      = _cur_lane
_lane_t           = 1.0   # 1.0 means no transition in progress
_lane_offset_from = OUR_LANE_OFFSETS[_cur_lane]
_lane_offset_to   = OUR_LANE_OFFSETS[_cur_lane]

# Steps remaining before autonomous avoidance is allowed after a human command
_avoidance_suppressed = 0

# Take one step to prime the camera before logging initial state
if USE_CAMERA:
    car.step(timestep)
    _update_lane_from_camera()

print(f"[INIT] pos=({car_x:.1f},{car_y:.1f})  heading={math.degrees(car_heading):.1f} deg")
print(f"[INIT] road={cur_road.id}  {cur_from_jid}->{cur_next_jid}")
print(f"[INIT] startup sl={_startup_canonical_sl:.2f}  lane={_cur_lane}  "
      f"cam_lane={_camera_lane}  yellow_blocked={_yellow_blocked}")
print(f"[INIT] SPEED={SPEED} m/s")


# ===========================================================================
# Wait for the first goal click before starting to drive
# ===========================================================================

waypoints   = cur_road.waypoints(cur_from_jid, _startup_canonical_sl)
final_phase = False
goal_x = goal_y = None
g_road  = None
goal_jid = None
prev_left = False

while car.step(timestep) != -1:
    state     = mouse.getState()
    left_down = (state.left == 1)
    if left_down and not prev_left:
        x, y, z = state.x, state.y, state.z
        if not (math.isnan(x) or math.isnan(y) or math.isnan(z)):
            goal_x, goal_y, g_road, goal_jid = snap_to_road_lane(
                x, y, cur_next_jid, cur_from_jid)
            place_marker(car, goal_x, goal_y, GROUND_Z)
            print(f"[INPUT] Goal set on road {g_road.id} ({goal_x:.1f},{goal_y:.1f})")
            break
        else:
            print("[INPUT] Click missed geometry -- try clicking on the road.")
    prev_left = left_down

route     = astar(cur_next_jid, goal_jid, forbidden=cur_from_jid)
route_idx = 0
print(f"[INIT] route: {cur_from_jid} -> {' -> '.join(route)}")

wp_idx = _skip_behind_wps(waypoints, car_x, car_y, car_heading)
tx, ty = waypoints[wp_idx] if waypoints else (car_x, car_y)

# How often (in simulation steps) to poll the camera for lane/obstacle data
CAM_POLL_STEPS = 1


# ===========================================================================
# Main simulation loop
# ===========================================================================

step_count = 0

while car.step(timestep) != -1:
    dt          = timestep / 1000.0
    step_count += 1

    with _speed_lock:
        current_speed = SPEED

    # Poll camera for lane and obstacle data at the configured rate
    if step_count % CAM_POLL_STEPS == 0:
        _update_lane_from_camera()

    # Tick down the human-override avoidance suppression countdown
    if _avoidance_suppressed > 0:
        _avoidance_suppressed -= 1

    # --- Priority 1: fist stop -----------------------------------------------
    # While stopped, only a new goal click will clear the stop and resume.
    if _fist_event.is_set():
        state     = mouse.getState()
        left_down = (state.left == 1)
        if left_down and not prev_left:
            x, y, z = state.x, state.y, state.z
            if not (math.isnan(x) or math.isnan(y) or math.isnan(z)):
                goal_x, goal_y, g_road, goal_jid = snap_to_road_lane(
                    x, y, cur_next_jid, cur_from_jid)
                place_marker(car, goal_x, goal_y, GROUND_Z)
                route       = astar(cur_next_jid, goal_jid, forbidden=cur_from_jid)
                route_idx   = 0
                final_phase = False
                _fist_event.clear()
                print(f"[REROUTE] Resuming -- road {g_road.id}  {' -> '.join(route)}")
        prev_left = left_down
        continue

    # --- Priority 2: human gesture lane change --------------------------------
    # Drain the entire gesture queue each step so no command waits an extra tick.
    with _gesture_lock:
        pending_directions = list(_gesture_queue)
        _gesture_queue.clear()

    for direction in pending_directions:
        ok = _try_lane_change(direction, source="gesture")
        if ok:
            _avoidance_suppressed = HUMAN_OVERRIDE_COOLDOWN
            print(f"[LANE] Human override -- avoidance suppressed for "
                  f"{HUMAN_OVERRIDE_COOLDOWN} steps")
            break  # At most one lane change per step

    # --- Priority 3: autonomous obstacle avoidance ---------------------------
    _handle_obstacle_avoidance()

    # --- Lane-change transition tick -----------------------------------------
    if _lane_t < 1.0:
        _lane_t   = min(1.0, _lane_t + dt / LANE_CHANGE_DURATION)
        waypoints = cur_road.waypoints(cur_from_jid, _current_canonical_sl())
        wp_idx    = _first_wp_ahead(waypoints, car_x, car_y, car_heading,
                                    LANE_CHANGE_LOOKAHEAD)
        if _lane_t >= 1.0:
            _cur_lane         = _target_lane
            _lane_offset_from = OUR_LANE_OFFSETS[_cur_lane]
            _lane_offset_to   = OUR_LANE_OFFSETS[_cur_lane]
            waypoints = cur_road.waypoints(cur_from_jid, _current_canonical_sl())
            wp_idx    = _first_wp_ahead(waypoints, car_x, car_y, car_heading,
                                        LANE_CHANGE_LOOKAHEAD)
            print(f"[LANE] Complete -- lane={_cur_lane}  dodged={_dodged}")

    # --- Mouse click: set a new goal -----------------------------------------
    state     = mouse.getState()
    left_down = (state.left == 1)
    if left_down and not prev_left:
        x, y, z = state.x, state.y, state.z
        if not (math.isnan(x) or math.isnan(y) or math.isnan(z)):
            goal_x, goal_y, g_road, goal_jid = snap_to_road_lane(
                x, y, cur_next_jid, cur_from_jid)
            place_marker(car, goal_x, goal_y, GROUND_Z)
            route       = astar(cur_next_jid, goal_jid, forbidden=cur_from_jid)
            route_idx   = 0
            final_phase = False
            print(f"[REROUTE] road {g_road.id}  {' -> '.join(route)}")
    prev_left = left_down

    # --- Goal reached: wait for the next click -------------------------------
    if hdist(car_x, car_y, goal_x, goal_y) < JUNC_RADIUS:
        print(f"[DONE] Reached goal at step {step_count}.")
        prev_left = False
        while car.step(timestep) != -1:
            state     = mouse.getState()
            left_down = (state.left == 1)
            if left_down and not prev_left:
                x, y, z = state.x, state.y, state.z
                if not (math.isnan(x) or math.isnan(y) or math.isnan(z)):
                    goal_x, goal_y, g_road, goal_jid = snap_to_road_lane(
                        x, y, cur_next_jid, cur_from_jid)
                    place_marker(car, goal_x, goal_y, GROUND_Z)
                    route       = astar(cur_next_jid, goal_jid, forbidden=cur_from_jid)
                    route_idx   = 0
                    final_phase = False
                    print(f"[INPUT] New goal: road {g_road.id}")
                    break
            prev_left = left_down

    # --- Advance past reached waypoints --------------------------------------
    while (wp_idx < len(waypoints) and
           hdist(car_x, car_y, *waypoints[wp_idx]) < WP_RADIUS):
        wp_idx += 1

    # --- Waypoints exhausted: handle junction or final approach --------------
    if wp_idx >= len(waypoints):
        arrived = cur_next_jid

        if final_phase or arrived == goal_jid or route_idx >= len(route) - 1:
            # Begin the final approach run on the goal road segment
            if not final_phase:
                final_phase = True

                dist_to_sj  = hdist(car_x, car_y,
                                    junctions[g_road.sjid].x, junctions[g_road.sjid].y)
                dist_to_ej  = hdist(car_x, car_y,
                                    junctions[g_road.ejid].x, junctions[g_road.ejid].y)
                arrived_end         = g_road.sjid if dist_to_sj < dist_to_ej else g_road.ejid
                _travelling_forward = (arrived_end == g_road.sjid)
                final_sl            = (OUR_LANE_OFFSETS[0] if _travelling_forward
                                       else -OUR_LANE_OFFSETS[0])

                if _cur_lane != 0:
                    print(f"[INFO] Final approach: lane {_cur_lane}->0 on {g_road.id}")
                    _lane_offset_from = OUR_LANE_OFFSETS[_cur_lane]
                    _lane_offset_to   = OUR_LANE_OFFSETS[0]
                    _target_lane      = 0
                    _lane_t           = 0.0
                else:
                    _lane_t = 1.0

                waypoints    = g_road.waypoints(arrived_end, final_sl)
                wp_idx       = _skip_behind_wps(waypoints, car_x, car_y, car_heading)
                cur_from_jid = arrived_end
                cur_next_jid = g_road.other_end(arrived_end)
                cur_road     = g_road
                print(f"[INFO] Final approach on {g_road.id}  "
                      f"fwd={_travelling_forward}  sl={final_sl:.2f}  "
                      f"wp_idx={wp_idx}/{len(waypoints)}")
        else:
            # Mid-route junction: optionally override via speech, otherwise follow A*
            dist_to_junc = hdist(car_x, car_y,
                                 junctions[arrived].x, junctions[arrived].y)
            time_to_junc = dist_to_junc / max(current_speed, 0.1)

            forced_nb = None
            if time_to_junc <= TURN_OVERRIDE_WINDOW:
                forced_nb, _ = _apply_speech_turn_override(
                    arrived, cur_from_jid, goal_jid)

            if forced_nb is not None:
                next_jid  = forced_nb
                next_road = road_between.get((arrived, next_jid))
                if next_road is None:
                    print(f"[SPEECH] No road {arrived}->{next_jid}, override ignored.")
                    forced_nb = None
                else:
                    route               = astar(next_jid, goal_jid, forbidden=arrived)
                    route_idx           = 0
                    cur_road            = next_road
                    cur_from_jid        = arrived
                    cur_next_jid        = next_jid
                    _travelling_forward = (cur_from_jid == cur_road.sjid)
                    waypoints           = cur_road.waypoints(
                        cur_from_jid, _current_canonical_sl())
                    wp_idx = 0
                    print(f"[SPEECH] New route: {arrived} -> {' -> '.join(route)}")

            if forced_nb is None:
                route_idx += 1
                next_jid  = route[route_idx]
                next_road = road_between.get((arrived, next_jid))
                if next_road is None:
                    print(f"[ERR] No road {arrived}->{next_jid}")
                    _stop_for_human(f"no road {arrived}->{next_jid}")
                    break
                print(f"[TURN] {arrived}: "
                      f"{turn_label(cur_from_jid, arrived, next_jid)} -> {next_road.id}")
                cur_road            = next_road
                cur_from_jid        = arrived
                cur_next_jid        = next_jid
                _travelling_forward = (cur_from_jid == cur_road.sjid)
                waypoints           = cur_road.waypoints(
                    cur_from_jid, _current_canonical_sl())
                wp_idx = 0

    if wp_idx < len(waypoints):
        tx, ty = waypoints[wp_idx]

    # --- Steer toward the current target waypoint ----------------------------
    dx, dy = tx - car_x, ty - car_y
    if math.hypot(dx, dy) > 0.1:
        err         = angle_diff(math.atan2(dy, dx), car_heading)
        car_heading += max(-MAX_STEER * dt, min(MAX_STEER * dt, err))

    # --- Advance car position ------------------------------------------------
    car_x += current_speed * dt * math.cos(car_heading)
    car_y += current_speed * dt * math.sin(car_heading)

    car_node.getField("translation").setSFVec3f([car_x, car_y, GROUND_Z])
    car_node.getField("rotation").setSFRotation([0.0, 0.0, 1.0,
                                                 car_heading + BODY_OFFSET])

    # --- Periodic status log (every 300 steps) --------------------------------
    if step_count % 300 == 0:
        status = "STOPPED" if _fist_event.is_set() else f"lane={_cur_lane}"
        tstr   = f"->{_target_lane}({_lane_t*100:.0f}%)" if _lane_t < 1.0 else ""
        def _lane_tag(z):
            return "BLOCKED" if _lane_near_blocked[z] else "ok"
        obs = (f" kerb={_lane_tag(0)} inner={_lane_tag(1)}"
               f" dodged={_dodged} suppress={_avoidance_suppressed}")
        print(f"[{step_count:5d}] road={cur_road.id} wp={wp_idx}/{len(waypoints)} "
              f"pos=({car_x:.1f},{car_y:.1f}) "
              f"dist_goal={hdist(car_x, car_y, goal_x, goal_y):.1f}m "
              f"speed={current_speed:.1f}{obs} [{status}{tstr}]")
