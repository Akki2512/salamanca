#!/bin/bash
# Show the robot's eyes in a Quest over the USB cable (no Wi-Fi involved).
#   Needs: adb (Android platform tools), the Quest in developer mode, plugged in, and "Allow USB debugging" accepted
#   in the headset. The session server (port 8000) and Unreal with the project open (port 8011) must be running.
# adb reverse makes the headset's own localhost:8000 and :8011 reach this Mac through the cable.
set -e
adb devices -l
adb reverse tcp:8000 tcp:8000      # the viewer page (session server)
adb reverse tcp:8011 tcp:8011      # the robot's-eye frames (Unreal)
adb reverse tcp:8765 tcp:8765      # the robot node: thumbstick walking and the clutch
adb reverse --list
# open the viewer in the headset's browser
adb shell am start -a android.intent.action.VIEW -d "http://localhost:8000/quest" >/dev/null
echo "Opened http://localhost:8000/quest in the headset. Press 'Enter VR' there (or 'Full screen' for a flat window)."
