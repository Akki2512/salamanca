"""The demo video's opening shot: close on the man's finger on the bell, a slow pull back, the robot walks in.

    RH_PACKETS=<abs path>/shot_opening_packets.jsonl RH_OUT=<abs folder> [RH_EVERY=1] [RH_W=1920 RH_H=1080] \
      UnrealEditor-Cmd RemoteHandsBedroom.uproject -ExecCmds="py <abs path>/Scripts/render_shot.py" -unattended -RenderOffscreen
Packets are 30 per second (tools/shot_opening.py); the shot is 24 frames per second. shot.json gets every frame's
camera (place, look-at point, field of view) so tools/cut_opening.py can draw over the picture in the right place.

    0.0 - 1.6 s   close on the bell; he presses it over and over, to the end of the shot (the sound is added in the cut)
    1.6 - 4.2 s   slow pull back, stopping on his hand, chest and knees
    4.2 - ...     the camera stands still; the robot sidesteps in from the right, its back filling the picture
RH_PACKET_OFFSET = (step_from_s in the packets' .meta.json) - 4.7, so the first step comes half a second into the hold.
"""
import json, math, os, traceback
import unreal

out = os.environ["RH_OUT"]
FPS, PPS = 24.0, 30.0
LENGTH = float(os.environ.get("RH_LENGTH", "11.5"))
OFFSET = float(os.environ.get("RH_PACKET_OFFSET", "3.6"))     # seconds into the packets at the shot's first frame
EVERY = int(os.environ.get("RH_EVERY", "1"))
W, H = int(os.environ.get("RH_W", "1920")), int(os.environ.get("RH_H", "1080"))
STEP, WARMUP = 7, 170
RING_AT, CLOSE_UNTIL, WIDE_FROM = 0.9, 1.6, 4.2
# He hammers the bell without stopping, a little unevenly: one press about every quarter of a second.
RINGS = [round(0.45 + 0.26 * k + 0.035 * math.sin(2.3 * k), 3) for k in range(int((LENGTH - 0.45) / 0.26))]
HOLD_ROBOT = os.environ.get("RH_ROBOT_HOLD") == "1"          # a second pass without the robot: it stays out of the picture
FROM = int(os.environ.get("RH_FROM", "0"))                    # first frame to render
st = {"n": 0, "h": None, "i": FROM, "res": {"fps": FPS, "frames": [], "rings": RINGS}}

room = lambda x, y, z: unreal.Vector(150.0 + 100.0 * x, 230.0 - 100.0 * y, 100.0 * z)     # home frame (m) -> level (cm)
mix = lambda a, b, k: a + (b - a) * k
ease = lambda k: k * k * k * (k * (6 * k - 15) + 10)


def setup():
    les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    les.load_level("/Game/Maps/HomeBedroom")
    actors = eas.get_all_level_actors()
    rx = [a for a in actors if a.get_class().get_name() == "RemoteHandsReceiver"][0]
    rx.set_editor_property("port", 7799)                  # keep this copy away from the live stream
    rx.set_editor_property("extra_ports", [])
    rx.set_editor_property("stream_eyes", False)
    st["rx"] = rx
    st["packets"] = open(os.environ["RH_PACKETS"]).read().splitlines()
    rx.apply_packet(st["packets"][int(OFFSET * PPS)])
    by = {a.get_actor_label(): a for a in actors}
    st["button"] = by["Call bell button"]
    st["man"] = by["Man in the wheelchair"]
    # The flick of his finger: two more copies of the man, the same but for the right index finger lifted off the
    # button (tools/make_wheelchair_man.py --pose finger_up / finger_mid). One of the three is in the chair at a time.
    project = os.path.dirname(os.path.dirname(os.path.abspath(os.environ["RH_SCRIPT"])))
    imported = json.load(open(os.path.join(project, "SourceAssets", "PolyHaven", "imported.json")))["models"]
    st["man_at"] = st["man"].get_actor_location()
    rot, scale = st["man"].get_actor_rotation(), st["man"].get_actor_scale3d()
    st["fingers"] = {}
    for pose_name in ("finger_up", "finger_mid"):
        parts = []
        for m in imported["seated_man_" + pose_name]["meshes"]:
            a = eas.spawn_actor_from_class(unreal.StaticMeshActor, st["man_at"], rot)
            a.static_mesh_component.set_mobility(unreal.ComponentMobility.MOVABLE)
            a.static_mesh_component.set_static_mesh(unreal.load_asset(m["path"]))
            a.set_actor_scale3d(scale)
            parts.append(a)
        st["fingers"][pose_name] = parts
    bell = st["button"].get_actor_location()
    st["bell"] = bell
    chair = by["Wheelchair"].get_actor_location()
    yaw = math.radians(19.0)
    f, side = unreal.Vector(math.cos(yaw), math.sin(yaw), 0), unreal.Vector(-math.sin(yaw), math.cos(yaw), 0)
    s = 1.0 if (bell - chair).dot(side) > 0 else -1.0      # which side of him the bell is on
    st["close"] = (bell + f * 34.0 + side * (s * 16.0) + unreal.Vector(0, 0, 15.0), bell + unreal.Vector(0, 0, 2.0), 38.0)
    far = (room(-1.71, 0.81, 1.22), room(-4.55, 2.05, 0.98))
    # the pull back stops early: his hand on the bell, his chest and his knees; the robot then steps in front of the lens
    st["wide"] = (mix(st["close"][0], far[0], 0.25), mix(st["close"][1], far[1], 0.25), 42.0)
    st["res"]["wide"] = [[v.x, v.y, v.z] for v in st["wide"][:2]]
    cam = eas.spawn_actor_from_class(unreal.CameraActor, st["close"][0], unreal.Rotator())
    cam.camera_component.set_editor_property("constrain_aspect_ratio", False)
    st["cam"] = cam
    st["res"].update(bell=[bell.x, bell.y, bell.z], chair=[chair.x, chair.y, chair.z], size=[W, H], offset=OFFSET)
    st["count"] = int(LENGTH * FPS)


def pose(i):
    t = i / FPS
    k = ease(min(1.0, max(0.0, (t - CLOSE_UNTIL) / (WIDE_FROM - CLOSE_UNTIL))))
    (p0, l0, f0), (p1, l1, f1) = st["close"], st["wide"]
    p, look, fov = mix(p0, p1, k), mix(l0, l1, k), mix(f0, f1, k)
    st["cam"].set_actor_location_and_rotation(p, unreal.MathLibrary.find_look_at_rotation(p, look), False, False)
    st["cam"].camera_component.set_editor_property("field_of_view", fov)
    st["rx"].apply_packet(st["packets"][min(len(st["packets"]) - 1, int(round(((0.0 if HOLD_ROBOT else t) + OFFSET) * PPS)))])
    # The press: only his index finger moves, the wrist stays where it is. Up between presses, down on the button.
    press = 0.0
    for ring in RINGS:
        u = t - ring
        press = max(press, max(0.0, 1.0 + u / 0.06) if u < 0 else max(0.0, 1.0 - u / 0.11))
    shown = "idle" if press >= 0.6 else "finger_mid" if press >= 0.25 else "finger_up"
    away = unreal.Vector(0, 0, -3000.0)
    st["man"].set_actor_location(st["man_at"] + (unreal.Vector() if shown == "idle" else away), False, False)
    for pose_name, parts in st["fingers"].items():
        for a in parts:
            a.set_actor_location(st["man_at"] + (unreal.Vector() if shown == pose_name else away), False, False)
    st["button"].set_actor_location(st["bell"] - unreal.Vector(0, 0, 0.6 if shown == "idle" else 0.0), False, False)
    st["res"]["frames"].append({"i": i, "t": round(t, 4), "cam": [p.x, p.y, p.z], "look": [look.x, look.y, look.z], "fov": fov})


def tick(dt):
    try:
        st["n"] += 1
        n = st["n"]
        if n == 5:
            setup()
        elif n >= WARMUP:
            phase = (n - WARMUP) % STEP
            i = st.setdefault("i", 0)
            name = os.path.join(out, "frame_%04d.png" % i)
            if phase == 0 and (os.path.exists(name) or st.get("tries", 0) >= 4):      # a screenshot is sometimes dropped: ask again
                i = st["i"] = i + EVERY
                st["tries"] = 0
                name = os.path.join(out, "frame_%04d.png" % i)
            if i >= st["count"]:
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
    json.dump(st["res"], open(os.path.join(out, "shot.json"), "w"))
    unreal.unregister_slate_post_tick_callback(st["h"])
    unreal.SystemLibrary.quit_editor()


st["h"] = unreal.register_slate_post_tick_callback(tick)
