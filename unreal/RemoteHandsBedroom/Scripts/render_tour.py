"""Renders a moving-camera clip of the room, with or without the robot acting out recorded packets.

    RH_SCRIPT=<this file> RH_OUT=<abs folder> RH_CAMS='[[t, x, y, z, lookx, looky, lookz, fov], ...]' RH_LENGTH=9 \
      [RH_PACKETS=<abs path, one packet a line, 30 a second> RH_PACKET_OFFSET=<seconds> RH_HOLD=1 (stay on that packet)] \
      [RH_EVERY=1 RH_W=1920 RH_H=1080]  UnrealEditor-Cmd RemoteHandsBedroom.uproject -ExecCmds="py <this file>" -unattended -RenderOffscreen
The camera goes smoothly through the key positions (level centimetres; look = the point it looks at). 24 frames a second.
"""
import json, os, traceback
import unreal

out = os.environ["RH_OUT"]
FPS, PPS = 24.0, 30.0
LENGTH = float(os.environ.get("RH_LENGTH", "9"))
OFFSET, HOLD = float(os.environ.get("RH_PACKET_OFFSET", "0")), os.environ.get("RH_HOLD") == "1"
EVERY = int(os.environ.get("RH_EVERY", "1"))
W, H = int(os.environ.get("RH_W", "1920")), int(os.environ.get("RH_H", "1080"))
CAMS = json.loads(os.environ["RH_CAMS"])
STEP, WARMUP = 7, 170
st = {"n": 0, "h": None, "i": 0, "res": {}}


def setup():
    les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    les.load_level("/Game/Maps/HomeBedroom")
    rx = [a for a in eas.get_all_level_actors() if a.get_class().get_name() == "RemoteHandsReceiver"][0]
    rx.set_editor_property("port", 7799)
    rx.set_editor_property("extra_ports", [])
    rx.set_editor_property("stream_eyes", False)
    st["rx"] = rx
    st["packets"] = open(os.environ["RH_PACKETS"]).read().splitlines() if os.environ.get("RH_PACKETS") else None
    if st["packets"]:
        rx.apply_packet(st["packets"][int(OFFSET * PPS)])
    cam = eas.spawn_actor_from_class(unreal.CameraActor, unreal.Vector(*CAMS[0][1:4]), unreal.Rotator())
    cam.camera_component.set_editor_property("constrain_aspect_ratio", False)
    st["cam"] = cam


def at(t):
    """The camera at t: between the two keys round it, eased."""
    k = 0
    while k + 2 < len(CAMS) and t >= CAMS[k + 1][0]:
        k += 1
    a, b = CAMS[k], CAMS[min(k + 1, len(CAMS) - 1)]
    u = 0.0 if b[0] == a[0] else min(1.0, max(0.0, (t - a[0]) / (b[0] - a[0])))
    u = u * u * (3 - 2 * u)
    return [a[j] + (b[j] - a[j]) * u for j in range(1, 8)]


def pose(i):
    t = i / FPS
    x, y, z, lx, ly, lz, fov = at(t)
    p = unreal.Vector(x, y, z)
    st["cam"].set_actor_location_and_rotation(p, unreal.MathLibrary.find_look_at_rotation(p, unreal.Vector(lx, ly, lz)), False, False)
    st["cam"].camera_component.set_editor_property("field_of_view", fov)
    if st["packets"]:
        st["rx"].apply_packet(st["packets"][min(len(st["packets"]) - 1, int(round(((0.0 if HOLD else t) + OFFSET) * PPS)))])


def tick(dt):
    try:
        st["n"] += 1
        n = st["n"]
        if n == 5:
            setup()
        elif n >= WARMUP:
            phase = (n - WARMUP) % STEP
            i = st["i"]
            name = os.path.join(out, "frame_%04d.png" % i)
            if phase == 0 and (os.path.exists(name) or st.get("tries", 0) >= 4):
                i = st["i"] = i + EVERY
                st["tries"] = 0
                name = os.path.join(out, "frame_%04d.png" % i)
            if i >= int(LENGTH * FPS):
                raise SystemExit
            if phase == 0:
                if not st.get("tries"):
                    pose(i)
                st["tries"] = st.get("tries", 0) + 1
            elif phase == 3:
                unreal.AutomationLibrary.take_high_res_screenshot(W, H, name, camera=st["cam"])
    except SystemExit:
        finish()
    except Exception:
        st["res"]["error"] = traceback.format_exc()
        finish()


def finish():
    json.dump(st["res"], open(os.path.join(out, "tour.json"), "w"))
    unreal.unregister_slate_post_tick_callback(st["h"])
    unreal.SystemLibrary.quit_editor()


st["h"] = unreal.register_slate_post_tick_callback(tick)
