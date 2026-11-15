"""Renders the level's camera views to PNG files without opening the editor window.

    RH_MAP=/Game/Maps/RobotStage RH_OUT=/some/folder UnrealEditor-Cmd RemoteHandsBedroom.uproject \
        -ExecCmds="py Scripts/render_views.py" -unattended -RenderOffscreen
Optional: RH_EXTRA_VIEW="x,y,z,pitch,yaw" adds one more view; RH_PACKET=<robot_state json file> poses the robot.
It waits about 150 frames (meshes and lighting need time to get ready), captures, then quits the editor.
"""
import json, os, traceback
import unreal

out = os.environ.get("RH_OUT", "/tmp")
state = {"tick": 0, "caps": [], "res": {}, "handle": None}


def setup():
    les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    les.load_level(os.environ.get("RH_MAP", "/Game/Maps/RobotStage"))
    world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
    actors = eas.get_all_level_actors()
    for a in actors:                                  # this copy of the editor only takes pictures: no eye stream
        if a.get_class().get_name() == "RemoteHandsReceiver":
            a.set_editor_property("stream_eyes", False)
    if os.environ.get("RH_PACKET"):
        for a in actors:
            if a.get_class().get_name() == "RemoteHandsReceiver":
                state["res"]["posed"] = bool(a.apply_packet(open(os.environ["RH_PACKET"]).read()))
    sm = [a for a in actors if isinstance(a, unreal.StaticMeshActor) and a.static_mesh_component.static_mesh]
    state["res"]["mesh_actors"] = len(sm)
    views = [(c.get_actor_label(), c.get_actor_location(), c.get_actor_rotation(), c.camera_component.field_of_view)
             for c in actors if isinstance(c, unreal.CameraActor)]
    if os.environ.get("RH_EXTRA_VIEW"):
        v = [float(t) for t in os.environ["RH_EXTRA_VIEW"].split(",")]
        views.append(("Close-up", unreal.Vector(v[0], v[1], v[2]), unreal.Rotator(roll=0, pitch=v[3], yaw=v[4]), 60.0))
    eas2 = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    for label, loc, rot, fov in views:
        cam = eas2.spawn_actor_from_class(unreal.CameraActor, loc, rot)       # a throwaway camera per view
        cam.camera_component.set_editor_property("field_of_view", fov)
        state["caps"].append((label, cam))
    state["world"] = world


def tick(dt):
    try:
        state["tick"] += 1
        if state["tick"] == 5:
            setup()
        elif state["tick"] >= 160 and (state["tick"] - 160) % 70 == 0:
            k = (state["tick"] - 160) // 70
            if k < len(state["caps"]):                  # one screenshot at a time: each needs a few frames
                label, cam = state["caps"][k]
                name = label.replace(" - ", "_").replace(" ", "_").replace("-", "_") + ".png"
                unreal.AutomationLibrary.take_high_res_screenshot(int(os.environ.get("RH_W", "1280")), int(os.environ.get("RH_H", "720")), os.path.join(out, name), camera=cam)
                state["res"][label] = name
            elif k == len(state["caps"]) + 1:
                raise SystemExit
    except SystemExit:
        finish()
    except Exception:
        state["res"]["error"] = traceback.format_exc()
        finish()


def finish():
    json.dump(state["res"], open(os.path.join(out, "render.json"), "w"), indent=1)
    unreal.unregister_slate_post_tick_callback(state["handle"])
    unreal.SystemLibrary.quit_editor()


state["handle"] = unreal.register_slate_post_tick_callback(tick)
