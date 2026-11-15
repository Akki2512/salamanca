# remote-hands

**What it is (corrected 2026-11-14).** A person with a mobility issue calls for help ("pass me the blue one").
Nemotron reads the request against what the robot's camera sees. A remote worker in a Rokoko suit moves the
robot's arms to fetch it, seeing through the robot's eyes on Meta glasses via a phone video call. The robot is a
simulated Unitree G1. The Aalto repo is used only as the robot / simulator interface; this is not the Aalto
sorting challenge. `../REMOTE_HANDS_SPEC.md` and `../REMOTE_HANDS_ARCHITECTURE.md` predate this correction (they
put the glasses on the person asking).

Pages served by the session server:
- `/caller`  the person asking: hold to speak or type; hears status back.
- `/`        the remote worker's station: the request with the object boxed, accept / done, live robot camera, map.
- `/pov`     the robot's camera full screen: open it on the phone that is in the video call to the glasses.

**The scene (changed 2026-11-14).** A wooden table at home with five everyday things and one serving tray; no
sorting bins. `home/scene.yaml` holds the geometry, `remote_hands/home.py` the items. The simulator only has
45 mm blocks, so each item is a colour + shape stand-in:

| Item | Looks like | Also understood as |
|---|---|---|
| pill bottle | white cylinder | pills, medicine, medication, tablets |
| soda can | red cylinder | soda, can, cola |
| cup | blue cylinder | mug, water, glass |
| tea box | yellow cube | tea |
| juice box | green cube | juice, carton |

Nemotron gets the robot's camera picture and the caller's sentence and returns the item, its box in the picture,
an instruction for the worker and a sentence for the caller. If it does not answer within `NEMOTRON_BUDGET_S`
seconds (`.env`), plain word matching against the table above is used instead (`"source": "keywords"`).

On the pod simulator the items have real outlines and parts (cap, label, can top, mug handle, carton top):
`home/home_items.py`, put into our simulator copy by `home/apply_to_sim_copy.py`. The laptop's mock robot still
draws them as plain blocks.

The camera stream is flow-controlled: a page gets its next frame only after it answers the last one with
`{"type": "pov_ack"}`. The link from the pod carried about 160 KB/s on 2026-11-14; without this, a few open tabs
filled the SSH tunnel and everything stalled. `tools/pod_tunnel.sh` keeps the tunnel up.

Status:
- Mock robot on a laptop: everything runs; 24 tests pass.
- Simulator on the pod: runs from our own copy (`/workspace/aalto/repo_home`), see "On the GPU pod".
- The suit drove the mock robot live through Rokoko Studio (`--source studio`), no safety clips.
- A 27-minute suit recording (`recordings/suit_104834.rkkudp`, 48,625 frames) replays with `--source replay`.
- Not done: the actual phone-to-glasses call, a full fetch driven by a person in the suit.

## Setup (already done on this laptop)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy scipy pyyaml pin websockets httpx fastapi "uvicorn[standard]" pillow
```

The Aalto repo is used as a library from `../aaltonomous-humanoid` (override with `$AALTO_REPO`). It needs the
G1 robot model at `third_party/xr_teleoperate/` inside that repo, fetched at the revision pinned in
`aaltonomous/cloud/setup_instance.sh`. On the GPU pod both already exist.

Secrets go in `.env` (git-ignored): `NEMOTRON_API_KEY` and `NEMOTRON_BASE_URL` (the Junction Nemotron
container hosted by Verda; OpenAI-compatible, model `nemotron-3.5-super-vl`, looked up from `/models`).

## Run

```bash
# one process: scripted human -> mock robot
.venv/bin/python -m remote_hands.robot_node --backend mock --source synthetic --seconds 20

# two processes, as in the real system
.venv/bin/python -m remote_hands.robot_node --backend mock --listen 8765
.venv/bin/python -m remote_hands.bridge --source synthetic --url ws://127.0.0.1:8765

# the whole request flow on one laptop (four terminals), then open http://localhost:8000
.venv/bin/python -m uvicorn remote_hands.session_server:app --port 8000
.venv/bin/python -m remote_hands.robot_node --backend mock --listen 8765 --session-url "ws://127.0.0.1:8000/ws/robot?session=default"
.venv/bin/python -m remote_hands.bridge --source synthetic
.venv/bin/python tools/scripted_fetch.py "can I have my pills please"     # a scripted caller and worker

# the suit instead of the scripted human (driver in its own terminal; suit on the same Wi-Fi as this laptop)
rokoko-sdk
.venv/bin/python -m remote_hands.bridge --source rokoko --height 1.80
#   it speaks two calibration poses (arms down, then arms straight out in front), 3 s each.
#   Enter in that terminal calibrates again. The robot follows only while the clutch is ON:
#   space bar (or a presenter clicker's Page Down) on the helper page.

# the recording instead of the suit (seconds into the recording; minutes 4-7 are busy two-handed motion)
.venv/bin/python -m remote_hands.bridge --source replay --start 240 --end 420
# live suit through Rokoko Studio: Livestream -> Custom, 127.0.0.1, port 14043, JSON v3 (LZ4 is fine)
.venv/bin/python -m remote_hands.bridge --source studio
# record Studio's stream (stop the bridge first)
.venv/bin/python tools/record_studio.py
# other devices on the same network (the phone): start the server and node with --host 0.0.0.0,
# then open http://<laptop address>:8000/pov or /caller on the phone

# Nemotron alone, on any image
.venv/bin/python tools/try_extractor.py ../dbg/s1_rgb.png "pass me the blue one" "give me the red one"

# tests (about 30 s)
.venv/bin/python -m unittest discover -s tests -v
```

### On the GPU pod

The main simulator and `/workspace/aalto/repo` belong to the other challenge (a teammate trains on them): do not
edit, restart or command them. Remote Hands has its own copy, `/workspace/aalto/repo_home` (the `aaltonomous`
folder copied, `third_party` linked), whose scene file and table texture are replaced by ours, run as a
separate instance:

```bash
# refresh the copy's scene after editing home/scene.yaml (rsync remote-hands to the pod first, without .env)
python3 remote-hands/home/apply_to_sim_copy.py /workspace/aalto/repo_home
# simulator (tmux rh-sim)
source /workspace/aalto/env.sh; cd /workspace/aalto/repo_home
AALTO_INSTANCE=b TASK=Aalto-Sort-G129-Dex3-Joint bash aaltonomous/sim/run_sim.sh > /workspace/aalto/sim_rh.log 2>&1
# robot node (tmux rh-node), in the isaac conda env, from /workspace/aalto/remote-hands
export REMOTE_HANDS_SIM=instance AALTO_DDS_DOMAIN=7 AALTO_CAM_PORT_BASE=56555 AALTO_SCENE_PORT=56580 AALTO_RGBD_PORT=56560
export AALTO_REPO=/workspace/aalto/repo_home PYTHONPATH=/workspace/aalto/repo_home:/workspace/aalto/remote-hands
python -m remote_hands.robot_node --backend sim --listen 8765 --session-url "ws://127.0.0.1:18021/ws/robot?session=default"
# laptop: session server on 0.0.0.0:8000, then the tunnel (node to the laptop, session server to the pod)
bash tools/pod_tunnel.sh
```

The node places the home items on the table when it connects. Stop the instance with `tmux kill-session -t
rh-sim` and, if the process survives, by its own pid (check `AALTO_INSTANCE=b` in `/proc/<pid>/environ` first).
Never `aalto stop`: it would kill the main simulator. The simulator runs at about 0.2-0.35 of real time.

### Modal

`modal deploy modal_deploy.py` puts the session server at
`the URL `modal deploy` prints (set it as `REMOTE_HANDS_DEPLOY_URL` in `.env`)`. It only accepts the session name in
`REMOTE_HANDS_DEPLOY_SESSION` (`.env`). Helper page: that address with `/?session=<name>`.
`.venv/bin/python tools/check_deployment.py` runs a full request through it.

## Files

| File | Runs on | Does |
|---|---|---|
| `remote_hands/messages.py` | both | The wrist-pose message and its frame contract |
| `remote_hands/sources.py` | helper laptop | Mocap sources: `synthetic` (scripted human) and `rokoko` |
| `remote_hands/rokoko.py` | helper laptop | Rokoko driver client (RGMP), two-pose calibration, sensor orientations -> wrist positions |
| `remote_hands/home.py` | both | The home scene's items, their names and start positions |
| `remote_hands/grasp_assist.py` | pod | The pick / release state machine for the task's object |
| `remote_hands/bridge.py` | helper laptop | Sends a source's poses to the robot node, prints robot status |
| `remote_hands/retarget.py` | pod | Clutched hand mapping, IK worker thread, rate limiting |
| `remote_hands/robot_node.py` | pod | The 30 Hz loop, table guard, telemetry, WebSocket endpoint |
| `remote_hands/extractor.py` | session server | Photo + sentence -> validated task JSON via Nemotron |
| `remote_hands/session_server.py` | laptop or Modal | Pairing, request lifecycle, logging, serves the helper page |
| `helper/index.html` | helper's browser | Wearer's frame with the target boxed, Nemotron's output, accept / done, robot status |
| `tools/fake_request.py` | anywhere | Stand-in for the phone app: sends one photo and sentence |

## Behaviour worth knowing

- **Clutch.** When the helper engages, their hand and the robot's hand are tied together where each one is.
  The robot hand then moves 0.75 m per metre of human hand travel (`--scale`). Clutch open or no pose for
  0.2 s: the robot holds. Re-engaging never moves the robot.
- **Speed.** The robot hand's goal follows the helper's hand at 0.30 m/s at most, and joints at 0.9 rad/s at
  most, so the Aalto safety filter (1.0 rad/s) never has to clip. Telemetry carries `target` (the helper's hand)
  and `goal` (where the robot is being sent) for the helper's ghost display.
- **Palms down.** The IK keeps both palms facing the counter. Wrist orientation from the suit is not used.
- **Ready pose.** The node first moves to the Aalto home hand positions with palms turned down.
- **Reach box.** Hand targets are clamped to x 0.22-0.48, |y| up to 0.36, z 0.035-0.32 m (pelvis frame).
- **Table guard.** No palm, finger or fingertip point may be driven lower once it is within 12 mm of the
  counter. The Aalto filter does not cover fingers.
  If the path to a goal just above the counter would dip below that, the goal is raised a few millimetres
  at a time until it clears.
- **Clutch switch.** Besides the source's own clutch flag, the node has a switch set by `{"type": "clutch"}`
  messages (helper page). The Rokoko bridge turns it off when it connects.
- **Grasp assist.** Needs object positions: the sim's scene service (`$AALTO_SCENE_PORT`) or, on the mock
  backend, the Aalto baseline's kinematic world. Capture radius 11 cm around the object, below 17 cm above grasp
  height; pulling the hand 19 cm away lets go (widened 14 Nov after live suit runs: the worker could not get
  inside 8 cm without seeing through the robot's eyes). Only the tray triggers a release. `RobotNode(assist=False)` turns it off.
- **Relaxed episode rules.** No time limit and no abort after 30 clipped ticks. The limits themselves are
  unchanged.
