# (Auto)pilot: Multimodal Human-Robot Interaction for Autonomous Vehicles

## Overview

This project implements a shared-autonomy autonomous vehicle system that allows a human to guide an autonomous car using gesture and voice commands while the vehicle continues performing navigation, perception, and control tasks autonomously.

The system integrates computer vision, speech recognition, gesture recognition, and graph-based navigation into a real-time control pipeline. Rather than switching between fully manual and fully autonomous modes, the system allows humans to provide high-level guidance while the autonomous system maintains low-level driving behavior.

The project was implemented and evaluated in the Webots robotics simulator using a Tesla car model with a front-facing camera.

---

## Features

**Autonomous Navigation**
- Graph-based road network representation built directly from the Webots scene
- A* path planning for shortest-route generation
- Waypoint-based lane following for both straight and curved roads
- Lane-aware navigation (correct side of road, proper lane offsets)

**Perception**
- Lane detection using OpenCV HSV color segmentation
- Obstacle detection using MediaPipe EfficientDet
- Rule-based obstacle avoidance with automatic lane changes

**Gesture Commands** (requires webcam)

| Gesture | Action |
|---|---|
| Closed fist | Stop vehicle |
| Open hand | Resume driving |
| Thumbs up | Change to kerb lane |
| Thumbs down | Change to inner lane |

**Voice Commands** (requires microphone + internet)

| Command | Action |
|---|---|
| "faster" | Increase speed by 1 m/s |
| "slower" | Decrease speed by 1 m/s |
| "left" | Turn left at next junction |
| "right" | Turn right at next junction |

**Shared Autonomy**

Human commands are integrated into the autonomy pipeline rather than overriding it. Safety constraints are always enforced:
- Lane boundary limits (cannot gesture outside valid lanes)
- Autonomous avoidance will not move into a blocked lane
- If both lanes are blocked and the system cannot resolve the situation, it halts and waits for human input

Human gesture lane changes take priority over autonomous obstacle avoidance. After a human gesture, autonomous avoidance is suppressed briefly so the system does not immediately undo the human's input.

---

## System Architecture

**Autonomous Vehicle Subsystem** — handles perception, navigation, and vehicle control:
- Road graph built from Webots scene nodes at startup
- A* planner computes junction-level routes
- Waypoints generated along each road segment and offset laterally into the target lane
- Lane detection and obstacle detection run on each camera frame

**Human Interaction Subsystem** — handles gesture and voice input:
- MediaPipe Gesture Recognizer runs on webcam frames in a background thread
- SpeechRecognition listens on the microphone in a separate background thread
- Commands are passed to the main loop via thread-safe queues and locks

The main control loop runs at each simulation timestep and processes inputs in strict priority order:
1. Fist stop (blocks all motion)
2. Human gesture lane change
3. Autonomous obstacle avoidance

---

## Repository Structure

```
.
├── worlds/
│   └── city.wbt                      # Webots world file
├── controllers/
│   └── controller/
│       ├── controller.py             # Main controller
│       └── models/                   # Place downloaded model files here
│           ├── gesture_recognizer.task
│           └── efficientdet_lite0.tflite
├── requirements.txt
└── README.md
```

---

## Installation

**1. Install Webots**

Download and install Webots R2025a from https://cyberbotics.com. Other versions have not been tested.

**2. Clone the repository**

```bash
git clone https://github.com/anvithasm/cs188-autopilot
cd cs188-autopilot
```

**3. Install Python dependencies**

```bash
pip install -r requirements.txt
```

On Linux, install PortAudio before PyAudio:
```bash
sudo apt install portaudio19-dev
pip install -r requirements.txt
```

On macOS:
```bash
brew install portaudio
pip install -r requirements.txt
```

**4. Download the ML model files**

Create the models directory and download both files into it:

```bash
mkdir -p controllers/controller/models
```

Then download:
- **Gesture recognizer** → save as `controllers/controller/models/gesture_recognizer.task`
  https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task

- **Object detector** → save as `controllers/controller/models/efficientdet_lite0.tflite`
  https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite

The controller will raise an error on startup if either file is missing.

---

## Running the Simulation

1. Launch Webots
2. Open `worlds/city.wbt` via **File → Open World**
3. Press the **Play** button
4. The console will print `[INPUT] Click on the road to set the goal.`
5. Click anywhere on the environment to set a destination — the car begins driving immediately to a goal point on the road closest to the clicked point
6. Click a new point at any time to update the goal
7. Use gestures or voice commands to intervene while the car drives autonomously

> **Note:** Speech recognition requires an active internet connection at runtime for the Google Web Speech API. Gesture control requires a webcam. Both are required for human interaction but optional otherwise — mouse-click navigation and autonomous avoidance work without any additional hardware.

---

## Hardware Requirements

| Hardware | Required for |
|---|---|
| Webcam | Gesture control |
| Microphone | Voice commands |
| Internet connection | Voice commands (Google STT API) |
