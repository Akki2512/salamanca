"""Makes the elderly man sitting in the Poly Haven wheelchair (wheelchair_01), for the demo room. Run with Blender:

    /Applications/Blender.app/Contents/MacOS/Blender -b --python tools/make_wheelchair_man.py [-- --preview out.png]

Needs the MPFB extension (Blender: extensions.blender.org, "mpfb") and MakeHuman's CC0 system asset pack unpacked
into MPFB's user data folder (skins, clothes, eyes, eyebrows). The body, skin and clothes are CC0.

Writes unreal/RemoteHandsBedroom/SourceAssets/PolyHaven/models/seated_man/seated_man.gltf: static meshes, posed,
in the wheelchair's own frame (the chair faces Blender -Y, floor at z 0, metres), so Unreal's import_polyhaven.py
imports it like the furniture and build_bedroom.py places it with the chair's own transform.
"""
import glob, json, math, os, sys
import bpy, bmesh
import numpy as np
from mathutils import Matrix, Vector
from bl_ext.blender_org.mpfb.services.humanservice import HumanService
from bl_ext.blender_org.mpfb.services.locationservice import LocationService
from bl_ext.blender_org.mpfb.services.targetservice import TargetService

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "unreal", "RemoteHandsBedroom", "SourceAssets", "PolyHaven", "models")
DATA = LocationService.get_user_data()
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
# --pose idle: in the chair, finger on the bell (model seated_man). hand_up: the same with his left hand at his
# mouth, for coughing and for drinking (seated_man_hand_up). bed: lying on his back (lying_man: origin under
# his hips on the mattress, head towards Blender +Y).
POSE = argv[argv.index("--pose") + 1] if "--pose" in argv else "idle"
# finger_up / finger_mid: as idle with only the right index finger lifted off the bell's button (24 and 11 degrees at
# the knuckle, the wrist where it was): shown in turn with idle, they are the flick of the finger that rings it.
NAME = {"idle": "seated_man", "hand_up": "seated_man_hand_up", "bed": "lying_man", "finger_up": "seated_man_finger_up",
        "finger_mid": "seated_man_finger_mid"}[POSE]
LIFT = {"finger_up": 24.0, "finger_mid": 11.0}.get(POSE)
OUT = os.path.join(MODELS, NAME)
preview = argv[argv.index("--preview") + 1] if "--preview" in argv else None

# The chair (cm, its front is -Y here): seat top at 58, back rest at y +13, armrest tops at 90 at x +-31, y 0..-42,
# footplates at 22, y -40..-53.
HIP = Vector((0.0, -0.025, 0.715))          # slid a little forward on the seat: he slouches           # where the hip joints go (m)

bpy.ops.wm.read_factory_settings(use_empty=True)
man = HumanService.create_human(macro_detail_dict={
    "gender": 1.0, "age": 1.0, "muscle": 0.3, "weight": 0.62, "proportions": 0.5, "height": 0.45, "cupsize": 0.5,
    "firmness": 0.5, "race": {"asian": 0.0, "caucasian": 1.0, "african": 0.0}})
# An old, scowling face, from MPFB's face-shape targets (weights 0..1): drawn-down brows, heavy lids, bags under
# the eyes, sagging cheeks and jowls, deep lines beside a turned-down mouth, a loose neck, an old man's nose and ears.
FACE = {"head-age-incr": 1.0, "head-fat-incr": 0.35, "eyebrows-angle-down": 1.0, "eyebrows-trans-down": 0.9, "eyebrows-trans-forward": 0.7,
        "mouth-angles-down": 1.0, "mouth-laugh-lines-in": 1.0, "mouth-lowerlip-middle-up": 0.7, "mouth-upperlip-volume-decr": 0.7,
        "mouth-lowerlip-ext-up": 0.5, "mouth-scale-horiz-incr": 0.3, "mouth-dimples-in": 0.5,
        "chin-jaw-drop-incr": 0.5, "chin-width-incr": 0.5, "neck-double-incr": 0.9, "neck-scale-horiz-incr": 0.3,
        "nose-volume-incr": 0.6, "nose-point-down": 0.6, "nose-scale-horiz-incr": 0.4, "nose-flaring-incr": 0.4,
        "forehead-nubian-incr": 0.3, "forehead-temple-decr": 0.4}
for side in "lr":
    FACE.update({side + "-eye-bag-incr": 1.0, side + "-eye-bag-out": 0.6, side + "-eye-eyefold-down": 0.9, side + "-eye-eyefold-angle-down": 0.5,
                 side + "-eye-height2-decr": 0.6, side + "-eye-corner2-down": 0.4, side + "-cheek-trans-down": 1.0,
                 side + "-cheek-volume-incr": 0.5, side + "-cheek-bones-decr": 0.4, side + "-ear-scale-incr": 0.6, side + "-ear-lobe-incr": 0.7})
for name, w in FACE.items():
    full = TargetService.target_full_path(name)
    if not full:
        print("NO TARGET", name)
        continue
    TargetService.load_target(man, full, weight=w, name=name)
rig = HumanService.add_builtin_rig(man, "default")


def asset(path, kind):
    f = glob.glob(os.path.join(DATA, path, "*.mhclo"))[0]
    HumanService.add_mhclo_asset(f, man, asset_type=kind, subdiv_levels=0, material_type="MAKESKIN")


asset("eyes/low-poly", "Eyes")
asset("eyebrows/eyebrow001", "Eyebrows")
asset("hair/short01", "Hair")
asset("clothes/male_casualsuit03", "Clothes")
asset("clothes/shoes01", "Clothes")
TEX = {"short01": "hair/short01/short01_diffuse.png", "Basemesh": "skins/old_caucasian_male/old_lightskinned_male_diffuse.png", "low-poly": "eyes/materials/brown_eye.png",
       "eyebrow": "eyebrows/eyebrow001/eyebrow001.png", "casualsuit": "clothes/male_casualsuit03/male_casualsuit03_diffuse.png",
       "shoes": "clothes/shoes01/shoes01_diffuse.png"}
NORMAL = {"casualsuit": "clothes/male_casualsuit03/male_casualsuit03_normal.png", "shoes": "clothes/shoes01/shoes01_normal.png"}


# ---- the pose: each bone is turned, about its own head, to point along a direction (x his left, -y forward, z up)
def aim(name, to):
    pb = rig.pose.bones[name]
    bpy.context.view_layer.update()
    M = rig.matrix_world @ pb.matrix
    head = M.translation.copy()
    now = (M.to_3x3() @ Vector((0, 1, 0))).normalized()
    q = now.rotation_difference(Vector(to).normalized())
    T = Matrix.Translation(head) @ q.to_matrix().to_4x4() @ Matrix.Translation(-head)
    pb.matrix = rig.matrix_world.inverted() @ T @ M
    bpy.context.view_layer.update()


def turn(name, axis, degrees):
    pb = rig.pose.bones[name]
    bpy.context.view_layer.update()
    M = rig.matrix_world @ pb.matrix
    head = M.translation.copy()
    T = Matrix.Translation(head) @ Matrix.Rotation(math.radians(degrees), 4, axis) @ Matrix.Translation(-head)
    pb.matrix = rig.matrix_world.inverted() @ T @ M
    bpy.context.view_layer.update()


def world(name, tail=False):
    pb = rig.pose.bones[name]
    return rig.matrix_world @ (pb.tail if tail else pb.head)


def palm_down(side, s):
    """Rolls the hand about its own length until the knuckles are level, little finger outwards."""
    bpy.context.view_layer.update()
    pb = rig.pose.bones["wrist." + side]
    M = rig.matrix_world @ pb.matrix
    head = M.translation.copy()
    d = (M.to_3x3() @ Vector((0, 1, 0))).normalized()
    across = world("finger5-1." + side) - world("finger2-1." + side)
    across = (across - across.dot(d) * d).normalized()
    want = Vector((s, 0, 0))
    want = (want - want.dot(d) * d).normalized()
    ang = math.atan2(d.dot(across.cross(want)), across.dot(want))
    T = Matrix.Translation(head) @ Matrix.Rotation(ang, 4, d) @ Matrix.Translation(-head)
    pb.matrix = rig.matrix_world.inverted() @ T @ M
    bpy.context.view_layer.update()


def local(name, axis, degrees):
    pb = rig.pose.bones[name]
    bpy.context.view_layer.update()
    pb.matrix = pb.matrix @ Matrix.Rotation(math.radians(degrees), 4, axis)
    bpy.context.view_layer.update()


def flex_axis(side):
    """Which way a finger bone turns to close the hand: tried on the middle finger of a palm-down hand."""
    best = None
    for axis in "XZ":
        for sign in (1, -1):
            keep = rig.pose.bones["finger3-1." + side].matrix.copy()
            local("finger3-1." + side, axis, 25 * sign)
            z = world("finger3-3." + side, True).z
            rig.pose.bones["finger3-1." + side].matrix = keep
            bpy.context.view_layer.update()
            if best is None or z < best[0]:
                best = (z, axis, sign)
    return best[1], best[2]


def curl(side, angles):
    axis, sign = FLEX[side]
    for f, per_joint in angles.items():
        for seg, a in enumerate(per_joint):
            local("finger%d-%d.%s" % (f, seg + 1, side), axis, a * sign)


# Slouched: the pelvis rolled back, the back curled forward over it, the head hanging forward and lifted to glare.
SLOUCH = (("spine05", -14), ("spine03", 7), ("spine02", 7), ("spine01", 5), ("neck01", -3), ("head", -8)) if POSE != "bed" else (("neck01", 10),)
TILT = sum(a for _, a in SLOUCH) + 9.0                    # how far the head is tipped forward of upright (degrees)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode="POSE")
for name, a in SLOUCH:
    turn(name, "X", a)
FLEX, GLASS = {}, None
for s, side in ((1, "L"), (-1, "R")):
    if POSE == "bed":                                     # on his back: legs straight, arms by his sides
        aim("upperarm01." + side, (0.16 * s, -0.05, -1.0))
        aim("lowerarm01." + side, (0.06 * s, -0.12, -1.0))
        continue
    aim("upperleg01." + side, (0.10 * s, -1.0, -0.02))    # thighs along the seat, knees a little apart
    aim("lowerleg01." + side, (0.0, -0.22, -1.0))         # shins down to the footplates
    aim("foot." + side, (0.10 * s, -1.0, -0.30))
    if side == "L":                                       # left forearm lying on the armrest, fingers over its front edge
        aim("upperarm01.L", (13, 2, -20))
        aim("lowerarm01.L", (-0.08, -1.0, -0.13))
        aim("wrist.L", (-0.04, -1.0, -0.04))
        angles = {f: (22, 30, 22) for f in (2, 3, 4, 5)}
    else:                                                 # right elbow drawn back: that hand is over the bell
        aim("upperarm01.R", (-14, 11, -19))
        aim("lowerarm01.R", (0.06, -1.0, -0.03))
        aim("wrist.R", (0.04, -1.0, -0.04))
        angles = {2: (6, 10, 6), 3: (42, 55, 35), 4: (48, 58, 38), 5: (52, 60, 40)}   # the index finger out, the rest curled
    palm_down(side, s)
    FLEX[side] = flex_axis(side)
    if side == "L" and POSE == "hand_up":                 # the left hand brought up in front of his mouth, half closed
        lips = (world("oris01") + world("oris05")) / 2
        aim("upperarm01.L", (-0.05, -0.80, -0.60))
        to = lips + Vector((0.035, -0.085, -0.035))
        aim("lowerarm01.L", to - world("lowerarm01.L"))
        aim("wrist.L", (lips + Vector((0.02, -0.05, 0.06))) - world("wrist.L"))
        palm_down("L", -1)                                # palm towards his face
        angles = {f: (35, 45, 30) for f in (2, 3, 4, 5)}
    curl(side, angles)
    if side == "L" and POSE == "hand_up":
        k = world("finger3-1.L")
        GLASS = k + (lips - k).normalized() * 0.045       # where a glass is held: between the knuckles and his lips
bpy.ops.object.mode_set(mode="OBJECT")
bpy.context.view_layer.update()
TIP = None
HIPS0 = (world("upperleg01.L") + world("upperleg01.R")) / 2
if POSE != "bed":
    hips = [rig.matrix_world @ rig.pose.bones["upperleg01." + s].head for s in "LR"]
    move = HIP - (hips[0] + hips[1]) / 2
    rig.location += move
    if GLASS is not None:
        GLASS = GLASS + move
    bpy.context.view_layer.update()
    # the tip of his right index finger rests on the bell's button: the bell stands on the armrest (top at
    # 90.6 cm, the button's top 5.1 cm above that), and build_bedroom.py puts it under the fingertip
    for _ in range(8):
        err = world("finger2-3.R", True).z - (0.957 + 0.008)
        turn("wrist.R", "X", math.degrees(math.asin(max(-0.5, min(0.5, err / 0.17)))))
    TIP = world("finger2-3.R", True).copy()
    if LIFT:
        axis, sign = FLEX["R"]
        local("finger2-1.R", axis, -LIFT * sign)
        local("finger2-2.R", axis, -0.3 * LIFT * sign)
        print("FINGER LIFTED", LIFT, "tip now", [round(v * 100, 1) for v in world("finger2-3.R", True)])
    print("BELL TIP", [round(v * 100, 1) for v in TIP], "GLASS", GLASS and [round(v * 100, 1) for v in GLASS], "FLEX", FLEX)

# ---- static, posed copies with plain image materials (what the glTF exporter and Unreal read reliably)
dg = bpy.context.evaluated_depsgraph_get()
made = []
os.makedirs(OUT, exist_ok=True)
for ob in [o for o in bpy.data.objects if o.type == "MESH"]:
    key = next((k for k in TEX if k.lower() in ob.name.lower()), None) or ("Basemesh" if ob == man else None)
    if key is None:
        print("SKIPPED", ob.name)
        continue
    me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    me.transform(ob.matrix_world)
    me.materials.clear()
    m = bpy.data.materials.new("man_" + key.lower())
    m.use_nodes = True
    nt = m.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    img = nt.nodes.new("ShaderNodeTexImage")
    img.image = bpy.data.images.load(os.path.join(DATA, TEX[key]))
    if key == "casualsuit":                               # the shirt made pale with grey-blue stripes; the jeans left alone
        px = np.array(img.image.pixels[:], dtype=np.float32).reshape(-1, 4)
        r, g, b = px[:, 0], px[:, 1], px[:, 2]
        shirt = ~((b > 1.12 * r) & (b > 0.04))
        stripe = np.clip(np.maximum(np.maximum(r, g), b) * 1.6, 0.0, 1.0)[shirt]
        for i, c in enumerate((0.80, 0.82, 0.86)):
            px[shirt, i] = c * (1.0 - 0.5 * stripe)
        img.image.pixels[:] = px.ravel().tolist()
        img.image.filepath_raw = os.path.join(OUT, "shirt_pale.png")
        img.image.file_format = "PNG"
        img.image.save()
    if key == "short01":                                  # white hair: the strands' shading kept, the colour replaced
        px = np.array(img.image.pixels[:], dtype=np.float32).reshape(-1, 4)
        lum = px[:, :3].max(axis=1)
        lo, hi = np.percentile(lum[px[:, 3] > 0.5], [5, 95])              # the strands' shading stretched to white and grey
        px[:, :3] = (0.50 + 0.48 * np.clip((lum - lo) / max(hi - lo, 1e-6), 0.0, 1.0) ** 0.8)[:, None]
        img.image.pixels[:] = px.ravel().tolist()
        img.image.filepath_raw = os.path.join(OUT, "hair_white.png")
        img.image.file_format = "PNG"
        img.image.save()
    if key == "eyebrow":                                  # white eyebrows: same shape, the colour replaced
        px = np.array(img.image.pixels[:], dtype=np.float32).reshape(-1, 4)
        px[:, :3] = 0.86
        img.image.pixels[:] = px.ravel().tolist()
        img.image.filepath_raw = os.path.join(OUT, "eyebrow_white.png")
        img.image.file_format = "PNG"
        img.image.save()
    nt.links.new(img.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.35 if key == "low-poly" else 0.62 if key == "Basemesh" else 0.9
    if key in ("eyebrow", "short01"):
        nt.links.new(img.outputs["Alpha"], bsdf.inputs["Alpha"])
        m.surface_render_method = "BLENDED"
    if key in NORMAL:
        nimg = nt.nodes.new("ShaderNodeTexImage")
        nimg.image = bpy.data.images.load(os.path.join(DATA, NORMAL[key]))
        nimg.image.colorspace_settings.name = "Non-Color"
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nt.links.new(nimg.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    me.materials.append(m)
    for p in me.polygons:
        p.use_smooth = True
    new = bpy.data.objects.new("man_" + key.lower(), me)
    bpy.context.collection.objects.link(new)
    made.append(new)
    lo = [min(v.co[i] for v in me.vertices) for i in range(3)]
    hi = [max(v.co[i] for v in me.vertices) for i in range(3)]
    print("PART", new.name, len(me.polygons), [round(v * 100, 1) for v in lo], [round(v * 100, 1) for v in hi])

def plain(name, rgb, rough):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Roughness"].default_value = rough
    return m


body = next(o for o in made if o.name == "man_basemesh")
eyes = next(o for o in made if o.name == "man_low-poly")
E = sum((v.co for v in eyes.data.vertices), Vector()) / len(eyes.data.vertices)      # between the eyes
top = max(v.co.z for v in body.data.vertices)

# white hair only at the sides and back: the hair piece with its top and front cut away (in the head's own axes)
untip = Matrix.Rotation(math.radians(-TILT), 3, "X")
hair = next(o for o in made if o.name == "man_short01")
bm = bmesh.new(); bm.from_mesh(hair.data)
rng = np.random.default_rng(5)
def bald(f):                                              # a ragged line, not a clean cut
    c = untip @ (f.calc_center_median() - E)
    return not (c.y > 0.05 + 0.012 * rng.random() and c.z < 0.050 + 0.15 * max(0.0, c.y - 0.12) + 0.018 * rng.random())
bmesh.ops.delete(bm, geom=[f for f in bm.faces if bald(f)], context="FACES")
bm.to_mesh(hair.data); bm.free()
print("PART hair faces left", len(hair.data.polygons))

# the oxygen tube: under the nose, across the cheeks, over the ears and down behind the jaw, lying on the skin
# (offsets from between the eyes with the head upright, started a little outside the face so that the nearest
# skin is the outside of the cheek, not the inside of the mouth; tipped with the head)
way = [(0.0, -0.046, -0.066), (0.034, -0.036, -0.060), (0.070, 0.008, -0.042), (0.094, 0.070, -0.012), (0.096, 0.105, 0.004),
       (0.086, 0.132, -0.035), (0.064, 0.118, -0.100), (0.034, 0.050, -0.185)]
tip = Matrix.Rotation(math.radians(TILT), 3, "X")
pts = [E + tip @ Vector((-x, y, z)) for x, y, z in reversed(way[1:])] + [E + tip @ Vector(w) for w in way]
fine = []
for a, b in zip(pts, pts[1:]):
    fine += [a.lerp(b, k / 8) for k in range(8)]
fine.append(pts[-1])
lm = bpy.data.meshes.new("line"); lm.from_pydata(fine, [(i, i + 1) for i in range(len(fine) - 1)], [])
line = bpy.data.objects.new("line", lm)
bpy.context.collection.objects.link(line)
sw = line.modifiers.new("on skin", "SHRINKWRAP")
sw.target, sw.wrap_method, sw.wrap_mode, sw.offset = body, "NEAREST_SURFACEPOINT", "ABOVE_SURFACE", 0.0035
on = bpy.data.meshes.new_from_object(line.evaluated_get(bpy.context.evaluated_depsgraph_get()))
cu = bpy.data.curves.new("man_tube", "CURVE")
cu.dimensions, cu.bevel_depth, cu.bevel_resolution = "3D", 0.0022, 3
sp = cu.splines.new("NURBS")
onv = list(on.vertices)[::4]                              # every fourth point: a smooth run, not every wrinkle
sp.points.add(len(onv) - 1)
for p_, v in zip(sp.points, onv):
    p_.co = (*v.co, 1.0)
sp.use_endpoint_u, sp.order_u = True, 4
tube = bpy.data.objects.new("tube_curve", cu)
bpy.context.collection.objects.link(tube)
bpy.context.view_layer.update()
tm = bpy.data.meshes.new_from_object(tube.evaluated_get(bpy.context.evaluated_depsgraph_get()))
tm.materials.append(plain("man_tube", (0.80, 0.86, 0.86), 0.3))
tubeo = bpy.data.objects.new("man_tube", tm)
bpy.context.collection.objects.link(tubeo)
made.append(tubeo)
print("PART man_tube", len(tm.polygons))

for ob in list(bpy.data.objects):
    if ob not in made:
        bpy.data.objects.remove(ob, do_unlink=True)

if POSE == "bed":                                         # laid on his back: head to +Y, face up, hips over the origin
    hips = HIPS0
    lay = Matrix.Rotation(math.radians(-90.0), 4, "X") @ Matrix.Translation((-hips.x, -hips.y, -hips.z))
    for ob in made:
        ob.data.transform(lay)
    low = min(v.co.z for ob in made if ob.name in ("man_basemesh", "man_casualsuit") for v in ob.data.vertices)
    for ob in made:
        ob.data.transform(Matrix.Translation((0.0, 0.0, -low)))
os.makedirs(OUT, exist_ok=True)
bpy.ops.object.select_all(action="SELECT")
bpy.ops.export_scene.gltf(filepath=os.path.join(OUT, NAME + ".gltf"), export_format="GLTF_SEPARATE", use_selection=True,
                          export_yup=True, export_apply=True, export_texture_dir="textures", export_image_format="AUTO")
ue = lambda v: [round(v.x * 100, 2), round(-v.y * 100, 2), round(v.z * 100, 2)]     # the chair's axes as Unreal has them, cm
json.dump({"bell_tip_cm": TIP and ue(TIP), "glass_cm": GLASS and ue(GLASS)}, open(os.path.join(OUT, "pose.json"), "w"))
print("WROTE", OUT)

if preview and POSE != "bed":                                               # pictures with the chair, to check the fit
    bpy.ops.import_scene.gltf(filepath=os.path.join(MODELS, "wheelchair_01", "wheelchair_01.gltf"))
    sc = bpy.context.scene
    sc.render.engine = "BLENDER_WORKBENCH"
    sc.display.shading.light = "STUDIO"
    sc.display.shading.color_type = "TEXTURE"
    sc.render.resolution_x, sc.render.resolution_y = 900, 900
    for loc, name in (((1.5, -2.0, 1.35), "a"), ((-2.3, -0.9, 1.1), "b"), ((0.0, -1.1, 1.45), "face")):
        cam = bpy.data.objects.new("cam" + name, bpy.data.cameras.new("cam" + name))
        cam.data.lens = 38 if name != "face" else 85
        cam.location = loc
        look = Vector((0.0, -0.1, 0.9 if name != "face" else 1.32)) - Vector(loc)
        cam.rotation_euler = look.to_track_quat("-Z", "Y").to_euler()
        bpy.context.collection.objects.link(cam)
        sc.camera = cam
        sc.render.filepath = preview.replace(".png", "_%s.png" % name)
        bpy.ops.render.render(write_still=True)
