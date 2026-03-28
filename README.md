# BlinkLock

## What it does
Counts rapid blinks using EAR. Three rapid blinks lock the screen. Unlock via slow wink or PIN fallback.

## Tech stack
- Python
- OpenCV
- MediaPipe FaceMesh

## Features
- EAR calculation
- Blink debounce logic
- 3-state state machine
- Lock screen trigger
- Wink or PIN unlock

## Run
pip install opencv-python mediapipe numpy
python blinklock.py