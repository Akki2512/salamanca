"""Imports the Unitree G1 (SourceAssets/G1, made by remote-hands/tools/export_g1_for_unreal.py) and builds
/Game/Maps/RobotStage: the real robot, standing, wired to a Remote Hands Receiver, with three cameras.

    UnrealEditor-Cmd RemoteHandsBedroom.uproject -ExecutePythonScript=Scripts/make_robot_stage.py -unattended -nullrhi

The robot is one actor per link, attached parent to child and labelled with the URDF link name, so the
receiver's Auto Bind Joints finds every joint. Environment variables (optional, for the self-check):
RH_PACKET = a robot_state packet (JSON file), RH_EXPECTED = {link: [x, y, z] cm from the pelvis} for that
packet, RH_REPORT = where to write the comparison.
"""
import json, os, traceback
import unreal

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "SourceAssets", "G1")
MAP = "/Game/Maps/RobotStage"
PELVIS_Z = 79.3                     # cm above the floor with the legs straight
report = {}


def material(name, rgb, roughness, metallic):
    path = "/Game/G1/Materials/" + name
    if unreal.EditorAssetLibrary.does_asset_exist(path):
        return unreal.load_asset(path)
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    m = tools.create_asset(name, "/Game/G1/Materials", unreal.Material, unreal.MaterialFactoryNew())
    mel = unreal.MaterialEditingLibrary
    c = mel.create_material_expression(m, unreal.MaterialExpressionConstant3Vector, -400, 0)
    c.set_editor_property("constant", unreal.LinearColor(rgb[0], rgb[1], rgb[2], 1.0))
    mel.connect_material_property(c, "", unreal.MaterialProperty.MP_BASE_COLOR)
    for value, prop, y in ((roughness, unreal.MaterialProperty.MP_ROUGHNESS, 200), (metallic, unreal.MaterialProperty.MP_METALLIC, 400)):
        k = mel.create_material_expression(m, unreal.MaterialExpressionConstant, -400, y)
        k.set_editor_property("r", value)
        mel.connect_material_property(k, "", prop)
    mel.recompile_material(m)
    unreal.EditorAssetLibrary.save_asset(path)
    return m


try:
    links = json.load(open(os.path.join(SRC, "g1.json")))["links"]
    mats = {"dark": material("M_G1_Dark", (0.035, 0.037, 0.04), 0.42, 0.15),
            "light": material("M_G1_Shell", (0.78, 0.79, 0.80), 0.35, 0.05)}

    # ---- meshes
    tasks = []
    for e in links:
        if e["mesh"] and not unreal.EditorAssetLibrary.does_asset_exist("/Game/G1/Meshes/" + e["name"]):
            t = unreal.AssetImportTask()
            t.filename = os.path.join(SRC, e["mesh"])
            t.destination_path = "/Game/G1/Meshes"
            t.automated, t.replace_existing, t.save = True, True, True
            tasks.append(t)
    if tasks:
        unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks(tasks)
    meshes = {}
    for e in links:
        if e["mesh"]:
            sm = unreal.load_asset("/Game/G1/Meshes/" + e["name"])
            if sm is None:
                raise RuntimeError("mesh did not import: " + e["name"])
            sm.set_material(0, mats[e["color"]])
            unreal.EditorAssetLibrary.save_loaded_asset(sm)
            meshes[e["name"]] = sm
    report["meshes"] = len(meshes)

    # ---- level
    les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    if unreal.EditorAssetLibrary.does_asset_exist(MAP):
        unreal.EditorAssetLibrary.delete_asset(MAP)
    assert les.new_level(MAP)

    def spawn(cls, label, loc=(0, 0, 0), rot=(0, 0, 0), folder=""):
        a = eas.spawn_actor_from_class(cls, unreal.Vector(*loc), unreal.Rotator(roll=rot[0], pitch=rot[1], yaw=rot[2]))
        a.set_actor_label(label)
        if folder:
            a.set_folder_path(folder)
        return a

    spawn(unreal.DirectionalLight, "Sun", (0, 0, 400), (0, -48, -40), "Lighting")
    spawn(unreal.SkyLight, "SkyLight", (0, 0, 300), folder="Lighting").light_component.set_editor_property("real_time_capture", True)
    spawn(unreal.SkyAtmosphere, "SkyAtmosphere", folder="Lighting")
    spawn(unreal.ExponentialHeightFog, "Fog", folder="Lighting")
    floor = spawn(unreal.StaticMeshActor, "Floor", (0, 0, 0))
    floor.static_mesh_component.set_static_mesh(unreal.load_asset("/Engine/BasicShapes/Plane"))
    floor.set_actor_scale3d(unreal.Vector(12, 12, 1))

    # ---- the robot: one actor per link
    actors = {}
    for e in links:                                     # the URDF lists parents before children
        a = spawn(unreal.StaticMeshActor, e["name"], folder="G1")
        c = a.static_mesh_component
        c.set_mobility(unreal.ComponentMobility.MOVABLE)
        if e["name"] in meshes:
            c.set_static_mesh(meshes[e["name"]])
        c.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
        actors[e["name"]] = a
        q = unreal.Quat(*e["quat_xyzw"])
        if e["parent"] is None:
            a.set_actor_location(unreal.Vector(0, 0, PELVIS_Z), False, False)
        else:
            a.attach_to_actor(actors[e["parent"]], "", unreal.AttachmentRule.KEEP_RELATIVE, unreal.AttachmentRule.KEEP_RELATIVE,
                              unreal.AttachmentRule.KEEP_RELATIVE, False)
            a.set_actor_relative_location(unreal.Vector(*e["pos_cm"]), False, False)
            a.set_actor_relative_rotation(q.rotator(), False, False)
    report["link_actors"] = len(actors)

    # ---- receiver at the pelvis
    rx = spawn(unreal.load_class(None, "/Script/RemoteHandsLink.RemoteHandsReceiver"), "RemoteHandsReceiver", (0, 0, PELVIS_Z))
    rx.set_editor_property("robot_actor", actors["pelvis"])
    rx.set_editor_property("draw_debug", False)
    rx.auto_bind_joints()
    joints = rx.get_editor_property("joints")
    report["joints_bound"] = len(joints)

    # ---- the three views
    eyes = spawn(unreal.CameraActor, "View 1 - Robot eyes", folder="Views")
    eyes.attach_to_actor(actors["d435_link"], "", unreal.AttachmentRule.SNAP_TO_TARGET, unreal.AttachmentRule.SNAP_TO_TARGET,
                         unreal.AttachmentRule.KEEP_WORLD, False)
    eyes.camera_component.set_editor_property("field_of_view", 87.0)
    spawn(unreal.CameraActor, "View 2 - Person", (70, 160, 95), (0, -8, -105), "Views")
    spawn(unreal.CameraActor, "View 3 - Room", (190, -210, 150), (0, -14, 132), "Views")
    spawn(unreal.PlayerStart, "PlayerStart", (190, -210, 100), (0, 0, 132))

    # ---- self-check: pose from a real packet, compare every link with the URDF's own kinematics
    if os.environ.get("RH_PACKET"):
        ok = rx.apply_packet(open(os.environ["RH_PACKET"]).read())
        expected = json.load(open(os.environ["RH_EXPECTED"]))
        errs = {}
        for name, want in expected.items():
            if name in actors:
                p = actors[name].get_actor_location()
                errs[name] = round(((p.x - want[0]) ** 2 + (p.y - want[1]) ** 2 + (p.z - PELVIS_Z - want[2]) ** 2) ** 0.5, 3)
        worst = sorted(errs.items(), key=lambda kv: -kv[1])[:6]
        report.update(packet_applied=bool(ok), links_compared=len(errs), worst_error_cm=worst,
                      max_error_cm=max(errs.values()))
        d = eyes.get_actor_forward_vector()
        report["robot_eyes_forward"] = [round(d.x, 3), round(d.y, 3), round(d.z, 3)]

    assert les.save_current_level()
    report["saved"] = MAP
except Exception:
    report["error"] = traceback.format_exc()
if os.environ.get("RH_REPORT"):
    json.dump(report, open(os.environ["RH_REPORT"], "w"), indent=1)
