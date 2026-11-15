"""Renders a recorded run (tools/record_packets.py) to numbered PNG frames, without opening the editor window.

    RH_PACKETS=recordings/run.jsonl RH_OUT=/some/folder RH_VIEW="x,y,z,pitch,yaw" [RH_FPS=15] [RH_MAP=/Game/Maps/HomeBedroom] \
      UnrealEditor-Cmd RemoteHandsBedroom.uproject -ExecCmds="py Scripts/render_clip.py" -unattended -RenderOffscreen
RH_VIEW=eyes uses the robot's head camera. Then: ffmpeg -framerate 15 -i frame_%04d.png -pix_fmt yuv420p clip.mp4
Each frame: apply the packet nearest that moment, let the renderer settle a few ticks, take a screenshot.
"""
import json, os, traceback
import unreal

out = os.environ["RH_OUT"]
fps = float(os.environ.get("RH_FPS", "15"))
STEP = int(os.environ.get("RH_TICKS_PER_FRAME", "6"))
WARMUP = 150
st = {"n": 0, "h": None, "res": {}}


def setup():
    les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    les.load_level(os.environ.get("RH_MAP", "/Game/Maps/HomeBedroom"))
    actors = eas.get_all_level_actors()
    st["rx"] = [a for a in actors if a.get_class().get_name() == "RemoteHandsReceiver"][0]
    st["rx"].set_editor_property("port", 7799)            # keep this copy away from the live stream's port
    st["rx"].set_editor_property("extra_ports", [])
    rows = [json.loads(l) for l in open(os.environ["RH_PACKETS"])]
    total = rows[-1]["at"]
    frames, k = [], 0
    for i in range(int(total * fps) + 1):
        while k + 1 < len(rows) and rows[k + 1]["at"] <= i / fps:
            k += 1
        frames.append(json.dumps(rows[k]["packet"]))
    st["frames"] = frames
    view = os.environ.get("RH_VIEW", "40,330,150,-18,-28")
    if view == "eyes":
        cam = [a for a in actors if isinstance(a, unreal.CameraActor) and "Robot eyes" in a.get_actor_label()][0]
    else:
        v = [float(t) for t in view.split(",")]
        cam = eas.spawn_actor_from_class(unreal.CameraActor, unreal.Vector(v[0], v[1], v[2]), unreal.Rotator(roll=0, pitch=v[3], yaw=v[4]))
        cam.camera_component.set_editor_property("field_of_view", float(os.environ.get("RH_FOV", "70")))
    st["cam"] = cam
    st["rx"].apply_packet(frames[0])
    st["res"]["frames"] = len(frames)


def tick(dt):
    try:
        st["n"] += 1
        n = st["n"]
        if n == 5:
            setup()
        elif n >= WARMUP:
            i, phase = divmod(n - WARMUP, STEP)
            if i >= len(st["frames"]) + 2:
                raise SystemExit
            if i < len(st["frames"]):
                if phase == 0:
                    st["rx"].apply_packet(st["frames"][i])
                elif phase == 2:
                    unreal.AutomationLibrary.take_high_res_screenshot(1280, 720, os.path.join(out, "frame_%04d.png" % i), camera=st["cam"])
    except SystemExit:
        finish()
    except Exception:
        st["res"]["error"] = traceback.format_exc()
        finish()


def finish():
    json.dump(st["res"], open(os.path.join(out, "clip.json"), "w"))
    unreal.unregister_slate_post_tick_callback(st["h"])
    unreal.SystemLibrary.quit_editor()


st["h"] = unreal.register_slate_post_tick_callback(tick)
