"""Builds /Game/Maps/HomeBedroom: a lived-in bedroom with the Unitree G1 standing at the bedside table.

    UnrealEditor-Cmd RemoteHandsBedroom.uproject -ExecutePythonScript=Scripts/build_bedroom.py -unattended -nullrhi

Needs: Scripts/make_robot_stage.py run once (imports the G1), Scripts/import_polyhaven.py run once (imports
the furniture and writes SourceAssets/PolyHaven/imported.json). Re-running rebuilds the level from scratch.

Layout (cm, Unreal axes). The robot stands at ROBOT facing +X, towards the wall the bed's headboard is on.
In front of it is the bedside table (its top 1 cm above the robot's pelvis: what the robot node assumes).
The bed is on the robot's right (+Y), so the person's pillow is beside the table and within the robot's reach.
"""
import json, math, os, traceback
import unreal

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP = "/Game/Maps/HomeBedroom"
ROOM_X, ROOM_Y, ROOM_H = (-450.0, 235.0), (-200.0, 430.0), 260.0   # 6.85 x 6.3 m (remote_hands/walk.py WALLS)
CHAIR, CHAIR_FACES = (-340.0, 60.0), 19.0   # the wheelchair, beside the radio, and the way it faces: towards the robot (remote_hands/home.py WHEELCHAIR)
ROBOT = (150.0, 230.0)
PELVIS_Z = 79.3
TABLE_TOP = PELVIS_Z + 1.0
BED_YAW = float(os.environ.get("RH_BED_YAW", "90"))     # pillow end at the head wall, beside the table
report = {}

les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
mel = unreal.MaterialEditingLibrary
tools = unreal.AssetToolsHelpers.get_asset_tools()
EAL = unreal.EditorAssetLibrary


# ---------------------------------------------------------------- materials for the room's surfaces
def surface_material(tex):
    """A tiling material from a Poly Haven texture set: colour, normal, roughness; TilesU / TilesV set per use."""
    path = "/Game/Room/ML_" + tex
    if EAL.does_asset_exist(path):
        return unreal.load_asset(path)
    base = "/Game/PolyHaven/Textures/%s/%s_" % (tex, tex)
    col, nor, rou = (unreal.load_asset(base + s) for s in ("color", "normal", "roughness"))
    nor.set_editor_property("srgb", False)
    nor.set_editor_property("compression_settings", unreal.TextureCompressionSettings.TC_NORMALMAP)
    rou.set_editor_property("srgb", False)
    for t in (nor, rou):
        EAL.save_loaded_asset(t)
    m = tools.create_asset("ML_" + tex, "/Game/Room", unreal.Material, unreal.MaterialFactoryNew())
    m.set_editor_property("two_sided", True)
    tint = mel.create_material_expression(m, unreal.MaterialExpressionVectorParameter, -500, -420)
    tint.set_editor_property("parameter_name", "Tint"); tint.set_editor_property("default_value", unreal.LinearColor(1, 1, 1, 1))
    uv = mel.create_material_expression(m, unreal.MaterialExpressionTextureCoordinate, -1100, 0)
    tu = mel.create_material_expression(m, unreal.MaterialExpressionScalarParameter, -1100, 150)
    tu.set_editor_property("parameter_name", "TilesU"); tu.set_editor_property("default_value", 1.0)
    tv = mel.create_material_expression(m, unreal.MaterialExpressionScalarParameter, -1100, 250)
    tv.set_editor_property("parameter_name", "TilesV"); tv.set_editor_property("default_value", 1.0)
    ap = mel.create_material_expression(m, unreal.MaterialExpressionAppendVector, -900, 200)
    mel.connect_material_expressions(tu, "", ap, "A"); mel.connect_material_expressions(tv, "", ap, "B")
    mul = mel.create_material_expression(m, unreal.MaterialExpressionMultiply, -750, 100)
    mel.connect_material_expressions(uv, "", mul, "A"); mel.connect_material_expressions(ap, "", mul, "B")
    for tex_asset, sampler, prop, y in ((col, unreal.MaterialSamplerType.SAMPLERTYPE_COLOR, unreal.MaterialProperty.MP_BASE_COLOR, -200),
                                        (nor, unreal.MaterialSamplerType.SAMPLERTYPE_NORMAL, unreal.MaterialProperty.MP_NORMAL, 100),
                                        (rou, unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR, unreal.MaterialProperty.MP_ROUGHNESS, 400)):
        s = mel.create_material_expression(m, unreal.MaterialExpressionTextureSample, -500, y)
        s.set_editor_property("texture", tex_asset)
        s.set_editor_property("sampler_type", sampler)
        mel.connect_material_expressions(mul, "", s, "UVs")
        if prop == unreal.MaterialProperty.MP_BASE_COLOR:      # colour x Tint (white leaves the texture as it is)
            mix = mel.create_material_expression(m, unreal.MaterialExpressionMultiply, -250, -300)
            mel.connect_material_expressions(s, "RGB", mix, "A"); mel.connect_material_expressions(tint, "", mix, "B")
            # ... then faded towards Pale by Lighten (0 = the scan's own colour, 1 = plain Pale): for light paint
            # that keeps the plaster's unevenness
            pale = mel.create_material_expression(m, unreal.MaterialExpressionVectorParameter, -250, -520)
            pale.set_editor_property("parameter_name", "Pale"); pale.set_editor_property("default_value", unreal.LinearColor(0.88, 0.88, 0.86, 1))
            lighten = mel.create_material_expression(m, unreal.MaterialExpressionScalarParameter, -250, -620)
            lighten.set_editor_property("parameter_name", "Lighten"); lighten.set_editor_property("default_value", 0.0)
            fade = mel.create_material_expression(m, unreal.MaterialExpressionLinearInterpolate, -80, -360)
            mel.connect_material_expressions(mix, "", fade, "A"); mel.connect_material_expressions(pale, "", fade, "B")
            mel.connect_material_expressions(lighten, "", fade, "Alpha")
            mel.connect_material_property(fade, "", prop)
        elif prop == unreal.MaterialProperty.MP_ROUGHNESS:     # roughness + Rougher (0 leaves the scan as it is), kept in 0..1
            more = mel.create_material_expression(m, unreal.MaterialExpressionScalarParameter, -500, 560)
            more.set_editor_property("parameter_name", "Rougher"); more.set_editor_property("default_value", 0.0)
            add = mel.create_material_expression(m, unreal.MaterialExpressionAdd, -250, 450)
            mel.connect_material_expressions(s, "R", add, "A"); mel.connect_material_expressions(more, "", add, "B")
            sat = mel.create_material_expression(m, unreal.MaterialExpressionSaturate, -120, 450)
            mel.connect_material_expressions(add, "", sat, "")
            mel.connect_material_property(sat, "", prop)
        else:
            mel.connect_material_property(s, "RGB" if prop != unreal.MaterialProperty.MP_ROUGHNESS else "R", prop)
    mel.recompile_material(m)
    EAL.save_asset(path)
    return m


def tiled(tex, tiles_u, tiles_v, tint=None, rougher=0.0, lighten=0.0):
    name = "MIL_%s_%dx%d%s%s%s" % (tex, round(tiles_u * 10), round(tiles_v * 10), "_%02x%02x%02x" % tuple(int(255 * c) for c in tint) if tint else "",
                                   "_r%d" % round(rougher * 100) if rougher else "", "_l%d" % round(lighten * 100) if lighten else "")
    path = "/Game/Room/" + name
    if EAL.does_asset_exist(path):
        return unreal.load_asset(path)
    mi = tools.create_asset(name, "/Game/Room", unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew())
    mel.set_material_instance_parent(mi, surface_material(tex))
    mel.set_material_instance_scalar_parameter_value(mi, "TilesU", tiles_u)
    mel.set_material_instance_scalar_parameter_value(mi, "TilesV", tiles_v)
    mel.set_material_instance_scalar_parameter_value(mi, "Rougher", rougher)
    mel.set_material_instance_scalar_parameter_value(mi, "Lighten", lighten)
    if tint:
        mel.set_material_instance_vector_parameter_value(mi, "Tint", unreal.LinearColor(tint[0], tint[1], tint[2], 1.0))
    EAL.save_asset(path)
    return mi


def plain_material(name, rgb, roughness=0.9, metallic=0.0, two_sided=False):
    path = "/Game/Room/" + name
    if EAL.does_asset_exist(path):
        return unreal.load_asset(path)
    m = tools.create_asset(name, "/Game/Room", unreal.Material, unreal.MaterialFactoryNew())
    m.set_editor_property("two_sided", two_sided)
    c = mel.create_material_expression(m, unreal.MaterialExpressionConstant3Vector, -400, 0)
    c.set_editor_property("constant", unreal.LinearColor(rgb[0], rgb[1], rgb[2], 1.0))
    mel.connect_material_property(c, "", unreal.MaterialProperty.MP_BASE_COLOR)
    r = mel.create_material_expression(m, unreal.MaterialExpressionConstant, -400, 200)
    r.set_editor_property("r", roughness)
    mel.connect_material_property(r, "", unreal.MaterialProperty.MP_ROUGHNESS)
    if metallic:
        k = mel.create_material_expression(m, unreal.MaterialExpressionConstant, -400, 350)
        k.set_editor_property("r", metallic)
        mel.connect_material_property(k, "", unreal.MaterialProperty.MP_METALLIC)
    mel.recompile_material(m)
    EAL.save_asset(path)
    return m


# ---------------------------------------------------------------- helpers
def spawn(cls, label, loc=(0, 0, 0), rot=(0, 0, 0), folder=""):
    a = eas.spawn_actor_from_class(cls, unreal.Vector(*loc), unreal.Rotator(roll=rot[0], pitch=rot[1], yaw=rot[2]))
    a.set_actor_label(label)
    if folder:
        a.set_folder_path(folder)
    return a


CUBE = unreal.load_asset("/Engine/BasicShapes/Cube")


def slab(label, lo, hi, mat, folder="Room"):
    """A wall / floor / ceiling piece filling the box lo..hi."""
    c = [(a + b) / 2 for a, b in zip(lo, hi)]
    a = spawn(unreal.StaticMeshActor, label, c, folder=folder)
    a.static_mesh_component.set_static_mesh(CUBE)
    a.set_actor_scale3d(unreal.Vector(*[(b - a_) / 100.0 for a_, b in zip(lo, hi)]))
    a.static_mesh_component.set_material(0, mat)
    return a


IMPORTED = json.load(open(os.path.join(HERE, "SourceAssets", "PolyHaven", "imported.json")))["models"]


def place(model, loc, yaw=0.0, scale=1.0, label=None, only=None, recentre=False, pitch=0.0, roll=0.0, folder="Furniture",
          movable=False, base_below=0.0, grip=None):
    """Put a Poly Haven model in the level. All its meshes share one origin, so they are spawned at one
    transform and attached to the first. only: keep meshes whose name contains one of these. recentre: `loc` is
    where the centre of the kept meshes' footprint goes (for one piece taken out of a set).
    Returns the main actor."""
    meshes = [m for m in IMPORTED[model]["meshes"] if only is None or any(o in m["path"].rsplit("/", 1)[-1] for o in only)]
    if not meshes:
        raise RuntimeError("no mesh for %s %s" % (model, only))
    off = (0.0, 0.0, 0.0)
    if grip is not None:
        # grip: a point on the model, in its own axes (cm), that must end up exactly at `loc`: where the robot's
        # hand takes hold of it. The handle actor sits there, so moving the handle carries the item by that point.
        y = math.radians(yaw)
        gx, gy, gz = (g * scale for g in grip)
        off = (-(gx * math.cos(y) - gy * math.sin(y)), -(gx * math.sin(y) + gy * math.cos(y)), -gz)
    elif recentre:
        lo = [min(m["min"][i] for m in meshes) for i in range(3)]
        hi = [max(m["max"][i] for m in meshes) for i in range(3)]
        cx, cy = (lo[0] + hi[0]) / 2 * scale, (lo[1] + hi[1]) / 2 * scale
        y = math.radians(yaw)
        # base_below: how far under `loc` the piece's base sits (for a handle at the piece's centre, not its base)
        off = (-(cx * math.cos(y) - cy * math.sin(y)), -(cx * math.sin(y) + cy * math.cos(y)), -lo[2] * scale - base_below)
    pos = (loc[0] + off[0], loc[1] + off[1], loc[2] + off[2])
    main = handle = None
    for m in meshes:
        a = spawn(unreal.StaticMeshActor, (label or model) if main is None else m["path"].rsplit("/", 1)[-1], pos,
                  (roll, pitch, yaw), folder)
        a.static_mesh_component.set_static_mesh(unreal.load_asset(m["path"]))
        a.set_actor_scale3d(unreal.Vector(scale, scale, scale))
        if movable:
            a.static_mesh_component.set_mobility(unreal.ComponentMobility.MOVABLE)
            a.static_mesh_component.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
        if main is None:
            main = a
            if (recentre or grip is not None) and movable:   # a handle actor at `loc`: moving it moves the piece
                h = spawn(unreal.StaticMeshActor, label or model, loc, folder=folder)
                h.static_mesh_component.set_mobility(unreal.ComponentMobility.MOVABLE)
                a.set_actor_label((label or model) + " mesh")
                a.attach_to_actor(h, "", unreal.AttachmentRule.KEEP_WORLD, unreal.AttachmentRule.KEEP_WORLD,
                                  unreal.AttachmentRule.KEEP_WORLD, False)
                handle = h
        else:
            a.attach_to_actor(main, "", unreal.AttachmentRule.KEEP_WORLD, unreal.AttachmentRule.KEEP_WORLD,
                              unreal.AttachmentRule.KEEP_WORLD, False)
    return handle or main


def prop(name):
    """A mesh modelled for this room (SourceAssets/Props/<name>.obj: the call bell)."""
    path = "/Game/Props/" + name
    if not EAL.does_asset_exist(path):
        t = unreal.AssetImportTask()
        t.filename = os.path.join(HERE, "SourceAssets", "Props", name + ".obj")
        t.destination_path = "/Game/Props"
        t.automated, t.replace_existing, t.save = True, True, True
        tools.import_asset_tasks([t])
    return unreal.load_asset(path)


def on_chair(label, mesh, mat, chair, local=(0.0, 0.0, 0.0), scale=1.0):
    """A part fixed to the wheelchair, given in the chair's own axes (x across, y the way it faces, z up; cm)."""
    a = spawn(unreal.StaticMeshActor, label, folder="Person")
    a.static_mesh_component.set_mobility(unreal.ComponentMobility.MOVABLE)
    a.static_mesh_component.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
    a.static_mesh_component.set_static_mesh(mesh)
    for i in range(max(1, len(mesh.static_materials))):
        a.static_mesh_component.set_material(i, mat)
    a.attach_to_actor(chair, "", unreal.AttachmentRule.KEEP_RELATIVE, unreal.AttachmentRule.KEEP_RELATIVE,
                      unreal.AttachmentRule.KEEP_RELATIVE, False)
    a.set_actor_relative_location(unreal.Vector(*local), False, False)
    a.set_actor_relative_scale3d(unreal.Vector(scale, scale, scale))
    return a


def thing(label, mesh, mat, loc, yaw=0.0, folder="Items"):
    """A movable thing whose mesh origin is the point the robot node reports for it."""
    a = spawn(unreal.StaticMeshActor, label, loc, (0, 0, yaw), folder)
    c = a.static_mesh_component
    c.set_mobility(unreal.ComponentMobility.MOVABLE)
    c.set_static_mesh(mesh)
    c.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
    for i in range(max(1, len(mesh.static_materials))):
        c.set_material(i, mat)
    return a


def glass_material(name="M_Glass", opacity=0.28, rgb=(0.82, 0.90, 0.92)):
    path = "/Game/Room/" + name
    if EAL.does_asset_exist(path):
        return unreal.load_asset(path)
    m = tools.create_asset(name, "/Game/Room", unreal.Material, unreal.MaterialFactoryNew())
    m.set_editor_property("blend_mode", unreal.BlendMode.BLEND_TRANSLUCENT)
    m.set_editor_property("two_sided", True)
    for value, prop_, y in ((None, unreal.MaterialProperty.MP_BASE_COLOR, 0), (opacity, unreal.MaterialProperty.MP_OPACITY, 200),
                            (0.05, unreal.MaterialProperty.MP_ROUGHNESS, 350)):
        if value is None:
            e = mel.create_material_expression(m, unreal.MaterialExpressionConstant3Vector, -400, y)
            e.set_editor_property("constant", unreal.LinearColor(rgb[0], rgb[1], rgb[2], 1.0))
        else:
            e = mel.create_material_expression(m, unreal.MaterialExpressionConstant, -400, y)
            e.set_editor_property("r", value)
        mel.connect_material_property(e, "", prop_)
    mel.recompile_material(m)
    EAL.save_asset(path)
    return m


def sky_material(cube):
    """What is seen out of the window: the sky panorama, looked up by the direction of view (so it is infinitely far)."""
    path = "/Game/Room/M_SkyDome"
    if EAL.does_asset_exist(path):
        return unreal.load_asset(path)
    m = tools.create_asset("M_SkyDome", "/Game/Room", unreal.Material, unreal.MaterialFactoryNew())
    m.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
    m.set_editor_property("two_sided", True)
    cam = mel.create_material_expression(m, unreal.MaterialExpressionCameraVectorWS, -900, 0)
    neg = mel.create_material_expression(m, unreal.MaterialExpressionConstant, -900, 150)
    neg.set_editor_property("r", -1.0)
    view = mel.create_material_expression(m, unreal.MaterialExpressionMultiply, -700, 0)
    mel.connect_material_expressions(cam, "", view, "A"); mel.connect_material_expressions(neg, "", view, "B")
    tex = mel.create_material_expression(m, unreal.MaterialExpressionTextureSample, -500, 0)
    tex.set_editor_property("texture", cube)
    tex.set_editor_property("sampler_type", unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR)
    mel.connect_material_expressions(view, "", tex, "UVs")
    gain = mel.create_material_expression(m, unreal.MaterialExpressionScalarParameter, -500, 250)
    gain.set_editor_property("parameter_name", "Brightness"); gain.set_editor_property("default_value", 1.7)
    # The panorama is a measurement of real daylight: its sky is hundreds of times brighter than the room, and
    # would show as plain white. Squeezed first (c / (1 + c)), so sky, clouds and trees all stay visible.
    one = mel.create_material_expression(m, unreal.MaterialExpressionConstant, -380, 140)
    one.set_editor_property("r", 1.0)
    plus = mel.create_material_expression(m, unreal.MaterialExpressionAdd, -380, 60)
    mel.connect_material_expressions(tex, "RGB", plus, "A"); mel.connect_material_expressions(one, "", plus, "B")
    squeeze = mel.create_material_expression(m, unreal.MaterialExpressionDivide, -300, 0)
    mel.connect_material_expressions(tex, "RGB", squeeze, "A"); mel.connect_material_expressions(plus, "", squeeze, "B")
    out = mel.create_material_expression(m, unreal.MaterialExpressionMultiply, -200, 0)
    mel.connect_material_expressions(squeeze, "", out, "A"); mel.connect_material_expressions(gain, "", out, "B")
    mel.connect_material_property(out, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    mel.recompile_material(m)
    EAL.save_asset(path)
    return m


def lamp(label, loc, intensity, radius=600.0, rgb=(1.0, 0.82, 0.62)):
    a = spawn(unreal.PointLight, label, loc, folder="Lighting")
    c = a.point_light_component
    c.set_editor_property("intensity_units", unreal.LightUnits.LUMENS)
    c.set_editor_property("intensity", intensity)             # lumens: a 60 W bulb is about 800
    c.set_editor_property("attenuation_radius", radius)
    c.set_editor_property("light_color", unreal.Color(int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255), 255))
    c.set_editor_property("source_radius", 6.0)
    return a


try:
    if EAL.does_asset_exist(MAP):
        EAL.delete_asset(MAP)
    assert les.new_level(MAP)
    (x0, x1), (y0, y1) = ROOM_X, ROOM_Y
    T = 12.0                                             # wall thickness

    # ---- the room: laminate floor, painted plaster walls, a window over the bed
    # the walls: a very light grey paint over the plaster scan (its unevenness and roughness kept)
    wall = lambda w, h: tiled("room_plaster", max(0.5, w / 220.0), max(0.5, h / 220.0), lighten=0.85)     # the team room's plaster (tools/import_room_textures.py)
    slab("Floor", (x0 - T, y0 - T, -T), (x1 + T, y1 + T, 0), tiled("room_oak", (x1 - x0) / 240.0, (y1 - y0) / 240.0, rougher=0.15))    # oak boards from the team room, matt
    slab("Ceiling", (x0 - T, y0 - T, ROOM_H), (x1 + T, y1 + T, ROOM_H + T), tiled("room_ceiling", (x1 - x0) / 320.0, (y1 - y0) / 320.0, lighten=0.45))     # the team room's ceiling, kept light
    hy0, hy1, hz0, hz1 = -150.0, -10.0, 100.0, 220.0     # a second window, in the head wall, towards the door-side corner
    slab("Wall head - below", (x1, y0 - T, 0), (x1 + T, y1 + T, hz0), wall(y1 - y0, hz0))
    slab("Wall head - above", (x1, y0 - T, hz1), (x1 + T, y1 + T, ROOM_H), wall(y1 - y0, ROOM_H - hz1))
    slab("Wall head - left", (x1, y0 - T, hz0), (x1 + T, hy0, hz1), wall(hy0 - y0, hz1 - hz0))
    slab("Wall head - right", (x1, hy1, hz0), (x1 + T, y1 + T, hz1), wall(y1 - hy1, hz1 - hz0))
    slab("Wall foot", (x0 - T, y0 - T, 0), (x0, y1 + T, ROOM_H), wall(y1 - y0, ROOM_H))
    slab("Wall door side", (x0, y0 - T, 0), (x1, y0, ROOM_H), wall(x1 - x0, ROOM_H))
    wx0, wx1, wz0, wz1 = -40.0, 110.0, 105.0, 215.0     # window opening in the wall beside the bed
    slab("Wall window - below", (x0, y1, 0), (x1, y1 + T, wz0), wall(x1 - x0, wz0))
    slab("Wall window - above", (x0, y1, wz1), (x1, y1 + T, ROOM_H), wall(x1 - x0, ROOM_H - wz1))
    slab("Wall window - left", (x0, y1, wz0), (wx0, y1 + T, wz1), wall(wx0 - x0, wz1 - wz0))
    slab("Wall window - right", (wx1, y1, wz0), (x1, y1 + T, wz1), wall(x1 - wx1, wz1 - wz0))
    frame = plain_material("M_WindowFrame", (0.9, 0.9, 0.88), 0.5)
    for lo, hi in (((wx0 - 4, y1 - 2, wz0 - 4), (wx1 + 4, y1 + T + 2, wz0)), ((wx0 - 4, y1 - 2, wz1), (wx1 + 4, y1 + T + 2, wz1 + 4)),
                   ((wx0 - 4, y1 - 2, wz0), (wx0, y1 + T + 2, wz1)), ((wx1, y1 - 2, wz0), (wx1 + 4, y1 + T + 2, wz1)),
                   (((wx0 + wx1) / 2 - 2, y1 + 3, wz0), ((wx0 + wx1) / 2 + 2, y1 + 8, wz1))):
        slab("Window frame", lo, hi, frame)
    slab("Window sill", (wx0 - 8, y1 - 14, wz0 - 4), (wx1 + 8, y1, wz0), frame)
    for lo, hi in (((x1 - 2, hy0 - 4, hz0 - 4), (x1 + T + 2, hy1 + 4, hz0)), ((x1 - 2, hy0 - 4, hz1), (x1 + T + 2, hy1 + 4, hz1 + 4)),
                   ((x1 - 2, hy0 - 4, hz0), (x1 + T + 2, hy0, hz1)), ((x1 - 2, hy1, hz0), (x1 + T + 2, hy1 + 4, hz1)),
                   ((x1 + 3, (hy0 + hy1) / 2 - 2, hz0), (x1 + 8, (hy0 + hy1) / 2 + 2, hz1))):
        slab("Window 2 frame", lo, hi, frame)
    slab("Window 2 sill", (x1 - 14, hy0 - 8, hz0 - 4), (x1, hy1 + 8, hz0), frame)
    slab("Skirting head", (x1 - 1.5, y0, 0), (x1, y1, 9), frame)
    slab("Skirting foot", (x0, y0, 0), (x0 + 1.5, y1, 9), frame)
    slab("Skirting door side", (x0, y0, 0), (x1, y0 + 1.5, 9), frame)
    slab("Skirting window side", (x0, y1 - 1.5, 0), (x1, y1, 9), frame)
    # outside: a garden panorama on a dome far around the house (Poly Haven "garden_nook"), seen through glass
    HDRI = unreal.load_asset("/Game/PolyHaven/HDRI/garden_nook")
    dome = spawn(unreal.StaticMeshActor, "Garden outside (panorama)", (0, 0, 0), folder="Outside")
    dome.static_mesh_component.set_static_mesh(unreal.load_asset("/Engine/BasicShapes/Sphere"))
    dome.set_actor_scale3d(unreal.Vector(600, 600, 600))
    dome.static_mesh_component.set_material(0, sky_material(HDRI))
    dome.static_mesh_component.set_cast_shadow(False)
    dome.static_mesh_component.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
    slab("Window glass", (wx0, y1 + 5, wz0), (wx1, y1 + 5.6, wz1), glass_material("M_WindowGlass", 0.04, (0.35, 0.42, 0.44)))
    slab("Window 2 glass", (x1 + 5, hy0, hz0), (x1 + 5.6, hy1, hz1), glass_material("M_WindowGlass", 0.04, (0.35, 0.42, 0.44)))
    # curtains drawn back at both sides of the window, on a rod
    linen = tiled("room_linen", 1.6, 3.2)               # the team room's linen weave, coloured in tools/import_room_textures.py
    for n, cx in enumerate((wx0 - 22, wx1 + 22)):
        c = spawn(unreal.StaticMeshActor, "Curtain %d" % (n + 1), (cx, y1 - 9, 22), folder="Room")
        c.static_mesh_component.set_static_mesh(prop("curtain"))
        c.set_actor_scale3d(unreal.Vector(0.62, 1.0, 2.08))
        c.static_mesh_component.set_material(0, linen)
    slab("Curtain rod", (wx0 - 62, y1 - 10.2, 230), (wx1 + 62, y1 - 7.8, 232.4), plain_material("M_Brass", (0.80, 0.58, 0.22), 0.22, metallic=1.0))
    # a door in the door-side wall
    door = plain_material("M_Door", (0.83, 0.81, 0.76), 0.45)
    slab("Door", (128, y0 - 1, 0), (216, y0 + 3.5, 204), door)
    for lo, hi in (((120, y0, 0), (128, y0 + 5, 212)), ((216, y0, 0), (224, y0 + 5, 212)), ((120, y0, 204), (224, y0 + 5, 212))):
        slab("Door frame", lo, hi, frame)
    slab("Door handle", (203, y0 + 3.5, 100), (213, y0 + 8.5, 102.5), plain_material("M_Brass", (0.80, 0.58, 0.22), 0.22, metallic=1.0))
    slab("Rug", (-200, 10, 0), (40, 170, 1.2), tiled("room_rug", 1.0, 1.0))     # one patterned rug, not tiled

    # ---- light: sun through the window, sky, a ceiling lamp and a bedside lamp
    # Daylight does the lighting: a low sun that comes in through both windows (it travels towards -x and -y),
    # the sky, and a soft panel of sky light just inside each window. The lamps are on low, as in daytime.
    sun = spawn(unreal.DirectionalLight, "Sun", (0, 600, 400), (0, -27, -128), "Lighting")
    sun.light_component.set_editor_property("intensity", 1.8)
    sun.light_component.set_editor_property("light_color", unreal.Color(226, 242, 255, 255))      # (b, g, r): warm white sunlight
    sun.light_component.set_editor_property("light_source_angle", 1.5)                            # soft shadow edges
    sky = spawn(unreal.SkyLight, "SkyLight", (0, 0, 300), folder="Lighting")
    sky.light_component.set_editor_property("source_type", unreal.SkyLightSourceType.SLS_SPECIFIED_CUBEMAP)
    sky.light_component.set_editor_property("cubemap", HDRI)
    sky.light_component.set_editor_property("intensity", 1.3)
    # daylight coming in at the window, as a soft panel of light just inside the glass
    for label, loc, yaw, w_, h_ in (("Window daylight", ((wx0 + wx1) / 2, y1 - 6, (wz0 + wz1) / 2), -90, wx1 - wx0, wz1 - wz0),
                                    ("Window 2 daylight", (x1 - 6, (hy0 + hy1) / 2, (hz0 + hz1) / 2), 180, hy1 - hy0, hz1 - hz0)):
        win = spawn(unreal.RectLight, label, loc, (0, 0, yaw), "Lighting")
        wc = win.get_component_by_class(unreal.RectLightComponent)
        for k, v in (("intensity_units", unreal.LightUnits.LUMENS), ("intensity", 600.0), ("source_width", w_), ("source_height", h_),
                     ("attenuation_radius", 1100.0), ("barn_door_angle", 85.0),
                     ("cast_shadows", False),            # fill light: its shadows put big grainy dark patches on the walls
                     ("light_color", unreal.Color(255, 246, 236, 255))):   # (b, g, r): cool sky
            wc.set_editor_property(k, v)
    ppv = spawn(unreal.PostProcessVolume, "Look", folder="Lighting")
    ppv.set_editor_property("unbound", True)
    s = ppv.get_editor_property("settings")
    look = {"auto_exposure_bias": -0.25, "auto_exposure_speed_up": 2.0, "auto_exposure_speed_down": 1.5,
            "bloom_intensity": 0.12, "vignette_intensity": 0.28, "ambient_occlusion_intensity": 0.6, "ambient_occlusion_radius": 60.0,
            "lumen_scene_lighting_quality": 2.0, "lumen_scene_detail": 2.0, "lumen_final_gather_quality": 2.0, "lumen_reflection_quality": 2.0,
            "film_slope": 0.86, "film_toe": 0.52, "white_temp": 7000.0, "color_saturation": unreal.Vector4(1.04, 1.04, 1.04, 1.0),
            "color_contrast": unreal.Vector4(1.05, 1.05, 1.05, 1.0)}
    report["look_skipped"] = []
    for k, v in look.items():
        try:
            s.set_editor_property("override_" + k, True)
            s.set_editor_property(k, v)
        except Exception:
            report["look_skipped"].append(k)
    ppv.set_editor_property("settings", s)

    rx_, ry_ = ROBOT
    # ---- bed and bedside table
    bed = place("GothicBed_01", (131.0, 345.0, 0.0), yaw=BED_YAW, label="Bed")
    k = TABLE_TOP / 61.6
    place("painted_wooden_nightstand", (rx_ + 14 + 25.3 * k + 1, ry_, 0.0), yaw=0.0, scale=k, label="Bedside table")
    tx = rx_ + 14 + 25.3 * k + 1                         # table centre x; its top spans about +-33 cm
    # things on the bedside table (the ones the robot can be asked for are movable and listed in `fetch`)
    # The three things the robot can be asked for. Each handle actor is where the robot node says the item is:
    # the point the hand takes hold of (remote_hands/home.py ITEMS "at" / "grip_z"; robot frame m -> + 100 x, - 100 y).
    # The jug is turned so its handle faces the robot and is held by the top of the handle; the small things are
    # gripped 2.25 cm above the table.
    fetch = {
        "water jug": place("jug_01", (rx_ + 30, ry_ + 22, TABLE_TOP + 15.5), yaw=0, label="water jug", folder="Items", movable=True,
                           grip=(-12.0, 0.0, 15.5)),
        "alarm clock": place("alarm_clock_01", (rx_ + 43, ry_ + 4, TABLE_TOP + 2.25), yaw=60, label="alarm clock", folder="Items",
                             movable=True, grip=(0.0, 1.0, 2.25)),
    }
    # a glass of water (modelled: SourceAssets/Props; its origin is where it is gripped, 5.5 cm above its base)
    glass = thing("glass of water", prop("glass"), glass_material(), (rx_ + 29, ry_ - 2, TABLE_TOP + 5.5))
    water = thing("water in the glass", prop("glass_water"), plain_material("M_Water", (0.55, 0.72, 0.80), 0.08), (rx_ + 29, ry_ - 2, TABLE_TOP + 5.5))
    water.attach_to_actor(glass, "", unreal.AttachmentRule.KEEP_WORLD, unreal.AttachmentRule.KEEP_WORLD, unreal.AttachmentRule.KEEP_WORLD, False)
    fetch["glass of water"] = glass
    # a sweater left on the floor (remote_hands/home.py "sweater" "room": -0.55, 1.35 from the robot's start: beside the rug)
    knit = tiled("room_knit", 5.0, 3.5)             # the team room's dark red knit
    fetch["sweater"] = thing("sweater", prop("sweater"), knit, (rx_ - 55, ry_ - 135, 7.3), yaw=35)
    # the same sweater hanging from one point (tools/make_sweater.py): shown in the robot's hand, where the node
    # swings it; it waits under the floor until then
    fetch["sweater hanging"] = thing("sweater hanging", prop("sweater_hanging"), knit, (0.0, 0.0, -100000.0))
    place("standing_picture_frame_01", (tx + 24, ry_ + 24, TABLE_TOP), yaw=200, label="Photo on the bedside table")
    place("potted_plant_04", (tx + 22, ry_ + 2, TABLE_TOP), yaw=40, label="Small plant")
    place("tea_set_01", (tx + 8, ry_ - 24, TABLE_TOP), yaw=10, label="Saucer", only=["plate_large_circular_03"], recentre=True)
    place("desk_lamp_arm_01", (tx + 27, ry_ - 26, TABLE_TOP - 8), yaw=75, label="Bedside lamp")   # away from the bed side: the jug is carried there
    lamp("Bedside lamp light", (tx + 8, ry_ - 10, TABLE_TOP + 62), 90.0, 300.0)
    place("hanging_picture_frame_01", (x1 - 2.0, ry_ - 5, 165.0), yaw=90, label="Picture above the table")
    # the frame comes with a black canvas: a painting from the team room goes in front of it
    slab("Painting above the table", (x1 - 4.4, ry_ - 5 - 25.5, 165.0 - 38.0), (x1 - 3.9, ry_ - 5 + 25.5, 165.0 + 38.0), tiled("room_picture", 1.0, 1.0), "Furniture")

    # ---- the rest of the room, lived in
    place("side_table_01", (x0 + 32, 150.0, 0.0), yaw=90, label="TV table")
    place("Television_01", (x0 + 32, 150.0, 55.0), yaw=90, label="Television")
    place("drawer_cabinet", (x0 + 28, 300.0, 0.0), yaw=90, label="Shelf unit")
    place("book_encyclopedia_set_01", (x0 + 30, 268.0, 113.0), yaw=90, label="Books on the shelf unit")
    place("wooden_bookshelf_worn", (40.0, y0 + 31, 0.0), yaw=180, label="Bookshelf")
    place("modern_arm_chair_01", (x0 + 62, y0 + 62, 0.0), yaw=132, label="Armchair")   # in the corner, facing the middle of the room
    place("throw_pillows_01", (x0 + 65, y0 + 65, 44.0), yaw=142, label="Cushion on the armchair", only=["pillow02"], recentre=True)
    place("potted_plant_01", (x1 - 32, y0 + 150, 0.0), yaw=0, scale=0.85, label="Plant")
    # furnishing along the walls (the middle of the room stays clear for walking). These models face their own +Y.
    place("sofa_03", (-190.0, y0 + 50, 0.0), yaw=0, label="Sofa")
    place("coffee_table_round_01", (-170.0, y0 + 140, 0.0), yaw=20, scale=0.8, label="Coffee table")
    place("ceramic_vase_03", (-160.0, y0 + 138, 49 * 0.8), yaw=0, scale=0.7, label="Vase on the coffee table")
    place("fancy_picture_frame_01", (-190.0, y0 + 1.0, 168.0), yaw=0, scale=1.7, label="Painting above the sofa")
    place("vintage_wooden_drawer_01", (-250.0, y1 - 60, 0.0), yaw=180, label="Chest of drawers")
    place("mantel_clock_01", (-285.0, y1 - 52, 109.0), yaw=180, label="Mantel clock")
    place("brass_vase_01", (-195.0, y1 - 50, 109.0), yaw=0, scale=0.55, label="Brass vase")
    place("ornate_mirror_01", (-250.0, y1 - 1.0, 186.0), yaw=180, scale=1.5, label="Mirror")
    place("vintage_grandfather_clock_01", (x0 + 27, y1 - 42, 0.0), yaw=-90, label="Grandfather clock")
    place("fancy_picture_frame_01", (x0 + 1.0, 150.0, 172.0), yaw=-90, scale=1.3, label="Painting above the radio")
    place("wall_clock", (x1 - 1.0, 70.0, 195.0), yaw=90, scale=1.2, label="Wall clock")
    place("Ottoman_01", (-12.0, 350.0, 0.0), yaw=90, label="Ottoman at the foot of the bed")
    place("wicker_basket_02", (-130.0, y1 - 40, 0.0), yaw=30, scale=1.6, label="Basket")
    cx_, cy_ = (x0 + x1) / 2, (y0 + y1) / 2             # nothing is left on the floor: the robot walks here
    place("modern_ceiling_lamp_01", (cx_, cy_, ROOM_H - 117.3), label="Ceiling lamp")
    lamp("Ceiling lamp light", (cx_, cy_, ROOM_H - 110.0), 150.0, 700.0)

    # ---- the person: an elderly man in a wheelchair, a brass call bell on the armrest under his right hand.
    # The chair model faces its own +Y. The man (tools/make_wheelchair_man.py: MPFB, CC0) is modelled in the chair's axes.
    # The receiver moves the chair and shows one pose of the man at a time (ext.scene in the packets): "idle" and
    # "hand_up" (coughing, drinking) ride the chair, "bed" lies in the bed. The poses not shown wait under the floor.
    chair = place("wheelchair_01", (CHAIR[0], CHAIR[1], 0.0), yaw=CHAIR_FACES - 90.0, label="Wheelchair", folder="Person", movable=True)
    AWAY = (0.0, 0.0, -100000.0)
    poses = {"idle": place("seated_man", (CHAIR[0], CHAIR[1], 0.0), yaw=CHAIR_FACES - 90.0, label="Man in the wheelchair", folder="Person", movable=True),
             "hand_up": place("seated_man_hand_up", AWAY, label="Man, hand at his mouth", folder="Person", movable=True),
             "bed": place("lying_man", AWAY, label="Man lying in bed", folder="Person", movable=True)}
    BED_POSE = ((134.0, 345.0, 45.0), 90.0)             # his hips on the mattress, head on the pillow at the head wall
    brass = plain_material("M_Brass", (0.80, 0.58, 0.22), 0.22, metallic=1.0)
    # the bell stands on the armrest on his right, its button under the tip of his index finger (pose.json is
    # written with the man by tools/make_wheelchair_man.py)
    tip = json.load(open(os.path.join(HERE, "SourceAssets", "PolyHaven", "models", "seated_man", "pose.json")))["bell_tip_cm"]
    bell_at = (tip[0], tip[1], 90.6)
    on_chair("Call bell", prop("call_bell_dome"), brass, chair, bell_at, 0.7)
    on_chair("Call bell base", prop("call_bell_base"), plain_material("M_BellBase", (0.03, 0.03, 0.03), 0.4), chair, bell_at, 0.7)
    on_chair("Call bell button", prop("call_bell_button"), brass, chair, bell_at, 0.7)

    # ---- the robot: one actor per link, as in make_robot_stage.py
    links = json.load(open(os.path.join(HERE, "SourceAssets", "G1", "g1.json")))["links"]
    actors = {}
    for e in links:
        a = spawn(unreal.StaticMeshActor, e["name"], folder="G1")
        c = a.static_mesh_component
        c.set_mobility(unreal.ComponentMobility.MOVABLE)
        if e["mesh"]:
            c.set_static_mesh(unreal.load_asset("/Game/G1/Meshes/" + e["name"]))
        c.set_collision_enabled(unreal.CollisionEnabled.NO_COLLISION)
        actors[e["name"]] = a
        if e["parent"] is None:
            a.set_actor_location(unreal.Vector(rx_, ry_, PELVIS_Z), False, False)
        else:
            a.attach_to_actor(actors[e["parent"]], "", unreal.AttachmentRule.KEEP_RELATIVE, unreal.AttachmentRule.KEEP_RELATIVE,
                              unreal.AttachmentRule.KEEP_RELATIVE, False)
            a.set_actor_relative_location(unreal.Vector(*e["pos_cm"]), False, False)
            a.set_actor_relative_rotation(unreal.Quat(*e["quat_xyzw"]).rotator(), False, False)
    rx = spawn(unreal.load_class(None, "/Script/RemoteHandsLink.RemoteHandsReceiver"), "RemoteHandsReceiver", (rx_, ry_, PELVIS_Z))
    rx.set_editor_property("robot_actor", actors["pelvis"])
    rx.set_editor_property("draw_debug", False)
    # the headset's picture: a little smaller and more compressed than the plugin's default (1280x1024 at 70), so
    # the headset's browser can download and decode 30 a second
    for k, v in (("eye_vr_width", 1024), ("eye_vr_height", 820), ("eye_jpeg_quality", 60)):
        rx.set_editor_property(k, v)
    rx.set_editor_property("items", fetch)
    rx.set_editor_property("chair_actor", chair)
    rx.set_editor_property("man_poses", poses)
    rx.set_editor_property("bed_pose_transform", unreal.Transform(unreal.Vector(*BED_POSE[0]), unreal.Rotator(roll=0.0, pitch=0.0, yaw=BED_POSE[1]), unreal.Vector(1, 1, 1)))
    # the robot node's room frame: origin on the floor under the robot's starting place, x the way it faces there
    rx.set_editor_property("sim_world_origin", unreal.Transform(location=[rx_, ry_, 0.0], rotation=[0, 0, 0], scale=[1, 1, 1]))
    rx.set_editor_property("item_offset", unreal.Vector(0, 0, 0))        # each item's handle actor is already at its grip point
    rx.auto_bind_joints()
    report["joints_bound"] = len(rx.get_editor_property("joints"))
    if os.environ.get("RH_PACKET"):
        report["posed"] = bool(rx.apply_packet(open(os.environ["RH_PACKET"]).read()))

    # ---- the three views
    eyes = spawn(unreal.CameraActor, "View 1 - Robot eyes", folder="Views")
    eyes.attach_to_actor(actors["d435_link"], "", unreal.AttachmentRule.SNAP_TO_TARGET, unreal.AttachmentRule.SNAP_TO_TARGET,
                         unreal.AttachmentRule.KEEP_WORLD, False)
    eyes.camera_component.set_editor_property("field_of_view", 87.0)
    rx.set_editor_property("eye_camera", eyes)          # served at http://127.0.0.1:8011/eyes.jpg for the headset
    fy = math.radians(CHAIR_FACES)                      # the person's view: from his eyes, the way the chair faces
    person_cam = spawn(unreal.CameraActor, "View 2 - Person", (CHAIR[0] + 24 * math.cos(fy), CHAIR[1] + 24 * math.sin(fy), 141.0), (0, -8, CHAIR_FACES), "Views")
    person_cam.camera_component.set_editor_property("field_of_view", 85.0)
    rx.set_editor_property("person_camera", person_cam)     # served at :8011/person.jpg; the receiver keeps it at his eyes
    spawn(unreal.CameraActor, "View 3 - Room", (x1 - 30, y0 + 30, 195.0), (0, -17, 137), "Views").camera_component.set_editor_property("field_of_view", 85.0)
    spawn(unreal.CameraActor, "View 4 - The man", (CHAIR[0] + 150 * math.cos(fy) + 40, CHAIR[1] + 150 * math.sin(fy) - 60, 138.0),
          (0, -9, CHAIR_FACES + 180 - 22), "Views").camera_component.set_editor_property("field_of_view", 60.0)
    for cam in eas.get_all_level_actors():           # the editor draws a camera as a 60 cm model: shrink it (the view is unaffected)
        if isinstance(cam, unreal.CameraActor):
            cam.set_actor_scale3d(unreal.Vector(0.12, 0.12, 0.12))
    spawn(unreal.PlayerStart, "PlayerStart", (-100, 100, 100), (0, 0, 35))

    rough = {}
    for path in EAL.list_assets("/Game/PolyHaven/Models", recursive=True):
        mi = unreal.load_asset(path)
        if not isinstance(mi, unreal.MaterialInstanceConstant) or "man" in path.split("/")[4]:
            continue
        # every scanned model a little more matt than the scan; its colours as scanned
        if "RoughnessFactor" in [str(n) for n in mel.get_scalar_parameter_names(mi)]:
            mel.set_material_instance_scalar_parameter_value(mi, "RoughnessFactor", 1.35)
            rough["RoughnessFactor"] = rough.get("RoughnessFactor", 0) + 1
        if "BaseColorFactor" in [str(n) for n in mel.get_vector_parameter_names(mi)]:
            mel.set_material_instance_vector_parameter_value(mi, "BaseColorFactor", unreal.LinearColor(1.0, 1.0, 1.0, 1.0))
            rough["BaseColorFactor"] = rough.get("BaseColorFactor", 0) + 1
        EAL.save_loaded_asset(mi)
    report["model_material_parameters"] = rough
    report["actors"] = len(eas.get_all_level_actors())
    report["fetch_items"] = {k: [round(v, 1) for v in a.get_actor_location().to_tuple()] for k, a in fetch.items()}
    assert les.save_current_level()
    report["saved"] = MAP
except Exception:
    report["error"] = traceback.format_exc()
if os.environ.get("RH_REPORT"):
    json.dump(report, open(os.environ["RH_REPORT"], "w"), indent=1)
