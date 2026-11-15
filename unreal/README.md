# Remote Hands in Unreal Engine

`RemoteHandsLink/` is an Unreal Engine 5.8 plugin (C++, runtime module). It listens on a UDP port for the
robot node's state and poses a Unitree G1 and the items in the level. Unreal only draws: the robot node decides
the joint angles and where every item is. The packet format is in `../remote_hands/unreal_stream.py`.

## The project

`RemoteHandsBedroom/` is a ready UE 5.8 C++ project with the plugin inside it
(`RemoteHandsBedroom/Plugins/RemoteHandsLink`; `RemoteHandsLink` next to this file is a link to it). Open
`RemoteHandsBedroom/RemoteHandsBedroom.uproject`. It is already built for the editor on this Mac; after
changing C++ rebuild with:

```bash
"/Users/Shared/Epic Games/UE_5.8/Engine/Build/BatchFiles/Mac/Build.sh" RemoteHandsBedroomEditor Mac Development -project="$PWD/RemoteHandsBedroom/RemoteHandsBedroom.uproject"
```

It opens on `Content/Maps/RobotStage`, made by `Scripts/make_robot_stage.py`: the real Unitree G1 (49 mesh
parts imported from its URDF, one actor per link, labelled with the link names), a Remote Hands Receiver at its
pelvis with all 28 arm and hand joints bound, and three cameras (robot eyes on the head camera mount, person,
room). There is no room yet: the bedroom is to be built from Fab / Megascans assets around the robot.
Re-running the script rebuilds that level from scratch and **discards anything added to it by hand**, so build
the bedroom in its own level and bring the robot in, or stop re-running it.

**The bedroom** is `Content/Maps/HomeBedroom` (the project's startup level), built by `Scripts/build_bedroom.py`:
a room with a window, laminate floor and plaster walls; a made bed, a bedside table, armchair, shelf unit, TV,
bookshelf, lamps, rug, and clutter (books, laptop, cushions, boxes, bin bag, laundry basket). The G1 stands at
the bedside table, whose top is 1 cm above its pelvis as the robot node assumes; the bed is on its right. On
the table are the things it can be asked for: a water jug, an alarm clock, a tea cup (already mapped in the
receiver's Items). Three cameras: robot eyes, person in bed (at the pillow), room.
Furniture and surface materials are CC0 models from Poly Haven, fetched by `../tools/fetch_polyhaven.py` into
`SourceAssets/PolyHaven` and imported by `Scripts/import_polyhaven.py`. Rebuilding recreates the level from
scratch (delete `Content/Maps/HomeBedroom.umap` first) and discards hand edits.
Since 14 Nov evening the room is 6.85 x 6.3 m with a clear floor, and the person is an elderly man in a
wheelchair out in the room with a call bell on the armrest (made by `../tools/make_wheelchair_man.py` with
Blender's MPFB extension and MakeHuman's CC0 asset pack; the bell is in `SourceAssets/Props`). The cameras are robot eyes, person, room, and one on the man.

- `../tools/export_g1_for_unreal.py` turns the URDF + STL files into `SourceAssets/G1` (OBJ per link, `g1.json`).
- `Scripts/render_views.py` renders every camera in a level to PNG without opening the editor window.
- Checked 14 Nov: posed from a real packet, all 53 links are within 0.001 cm of the URDF's own kinematics.
  Pictures: `renders/`.

Epic's "Unreal MCP" plugin is enabled in the project, set to port 8010 with auto start
(`Saved/Config/MacEditor/EditorPerProjectUserSettings.ini`), bound to localhost. Register it once with
`claude mcp add --transport http --scope user unreal http://127.0.0.1:8010/mcp`.

To use the plugin in another project instead: copy `RemoteHandsBedroom/Plugins/RemoteHandsLink` into that
project's `Plugins/` folder (a real copy, not a symlink: the build fails through a symlink).

## Set up a level

1. Place a **Remote Hands Receiver** actor **at the robot's pelvis (waist joint)**, with its arrow pointing the
   way the robot faces. Everything in the packets is relative to this actor.
2. With nothing else set up, start the robot node (below). Coloured debug shapes appear: two spheres for the
   hands, a box per item with its name, the put-down point, and the table top. Use them to line the bedroom
   furniture up with what the robot can reach **before** rigging anything.
3. Set **Robot Actor** to the G1 and press **Auto Bind Joints** in the Details panel. It finds, for each of the
   28 joints, a scene component or a bone named after the joint (`left_elbow_joint`) or its link
   (`left_elbow_link`), and stores the pose it finds as zero. The Output Log says how many it found.
   - Parts as separate components or attached actors (typical for a URDF import): works as is.
   - A skinned mesh: use a **Poseable Mesh** component (not Skeletal Mesh) so bones can be set directly.
   - A joint that turns the wrong way: set that joint's **Scale** to -1 in the **Joints** map. Wrong axis: edit
     **Axis** (Unreal axes, in the part's own frame). The defaults assume the parts keep the URDF's frames with
     y mirrored.
4. Fill **Items**: `pill bottle`, `soda can`, `cup`, `tea box`, `juice box` -> the actor to move for each.
   Their meshes can be any size and look; the robot grasps a 45 mm wide thing at that position.
5. It runs in the editor viewport without pressing Play (**Run In Editor**), and in Play.

For Blueprints: `State` (hands, items, task), `bConnected`, and the events `OnState` and
`OnTaskStateChanged(Item, NewState)` with states free, approach, descend, closing, holding, releasing, done.

## Feed it

```bash
cd remote-hands
.venv/bin/python -m remote_hands.robot_node --backend mock --listen 8765 --unreal-udp 127.0.0.1:7777
.venv/bin/python tools/scripted_fetch.py x --direct "pill bottle"     # makes the robot fetch something
```

Unreal on another machine: use that machine's address instead of `127.0.0.1`.

## The three views

Add three Camera actors: one attached to the G1's head (robot view), one at the person's eyes in bed, one in a
corner of the room. The robot view is what goes into the phone call for the glasses.

## Frames

Robot frame: origin at the pelvis, x forward, y to the robot's left, z up, metres. Unreal position relative to
the receiver actor: `(100 x, -100 y, 100 z)` cm. The table top is 1 cm above the pelvis and starts 14 cm in
front of it; the hands reach 0.22-0.48 m forward and up to 0.36 m to either side.

## Status (14 Nov 2026)

Written and compiled against UE 5.8 on macOS. Not yet run inside a project: no Unreal project or G1 import
exists yet.
