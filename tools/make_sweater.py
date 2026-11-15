"""A sweater dropped on the floor, for the demo room: a sweater-shaped sheet let fall on a few lumps with Blender's
cloth simulation. Run: Blender -b --python tools/make_sweater.py
Writes unreal/RemoteHandsBedroom/SourceAssets/Props/sweater.obj (cm; origin 7 cm above the floor at its middle:
the point the robot node gives as the sweater's position)."""
import math, os
import bpy, bmesh
from mathutils import Vector

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "unreal", "RemoteHandsBedroom", "SourceAssets", "Props")
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
bpy.ops.mesh.primitive_grid_add(x_subdivisions=70, y_subdivisions=56, size=1.0)
sw = bpy.context.object
sw.scale = (1.40, 0.70, 1.0)
bpy.ops.object.transform_apply(scale=True)
bm = bmesh.new(); bm.from_mesh(sw.data)
def inside(c):                      # body 52 cm wide, sleeves out to the sides at the top, a neck notch
    x, y = abs(c.x), c.y
    return (x < 0.26 and -0.35 < y < 0.33 and not (x < 0.08 and y > 0.27)) or (0.26 <= x < 0.70 and 0.33 - 0.17 - 0.25 * (x - 0.26) < y < 0.33 - 0.10 * (x - 0.26))
bmesh.ops.delete(bm, geom=[f for f in bm.faces if not inside(f.calc_center_median())], context="FACES")
bm.to_mesh(sw.data); bm.free()
sw.location = (0.0, 0.0, 0.42)
sw.rotation_euler = (math.radians(62), math.radians(18), math.radians(25))
cl = sw.modifiers.new("cloth", "CLOTH")
cl.settings.quality = 6
cl.settings.mass = 0.4
cl.settings.bending_stiffness = 0.6
cl.settings.tension_stiffness = 12
cl.collision_settings.use_self_collision = True
cl.collision_settings.self_distance_min = 0.006
cl.collision_settings.distance_min = 0.008
bpy.ops.mesh.primitive_plane_add(size=6.0)
bpy.context.object.modifiers.new("col", "COLLISION")
for loc, r in (((0.05, 0.0, 0.03), 0.07), ((-0.18, 0.12, 0.02), 0.05), ((0.2, -0.1, 0.02), 0.045)):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=loc)
    bpy.context.object.modifiers.new("col", "COLLISION")
sc.frame_start, sc.frame_end = 1, 90
cl.point_cache.frame_start, cl.point_cache.frame_end = 1, 90
for f in range(1, 91):
    sc.frame_set(f)
dg = bpy.context.evaluated_depsgraph_get()
me = bpy.data.meshes.new_from_object(sw.evaluated_get(dg))
me.transform(sw.matrix_world)
ob = bpy.data.objects.new("sweater", me)
bpy.context.collection.objects.link(ob)
ob.modifiers.new("thick", "SOLIDIFY").thickness = 0.007
ob.modifiers.new("sub", "SUBSURF").levels = 1
me2 = bpy.data.meshes.new_from_object(ob.evaluated_get(bpy.context.evaluated_depsgraph_get()))
lo = Vector([min(v.co[i] for v in me2.vertices) for i in range(3)]); hi = Vector([max(v.co[i] for v in me2.vertices) for i in range(3)])
mid = (lo + hi) / 2
for v in me2.vertices:              # centimetres, y mirrored for Unreal's OBJ importer, origin 7 cm above the floor
    v.co = Vector(((v.co.x - mid.x) * 100, -(v.co.y - mid.y) * 100, (v.co.z - lo.z) * 100 - 7.0))
me2.flip_normals()
for p in me2.polygons:
    p.use_smooth = True
out = bpy.data.objects.new("sweater_out", me2)
bpy.context.collection.objects.link(out)
bpy.ops.object.select_all(action="DESELECT"); out.select_set(True); bpy.context.view_layer.objects.active = out
bpy.ops.wm.obj_export(filepath=os.path.join(OUT, "sweater.obj"), export_selected_objects=True, forward_axis="Y", up_axis="Z",
                      export_materials=False, export_uv=True, export_normals=True, export_triangulated_mesh=True)
print("SWEATER cm", [round(v, 1) for v in (hi - lo) * 100], len(me2.polygons))


# ---- the same sweater held up by one point: what it looks like hanging from the robot's hand ("sweater_hanging.obj",
# origin at the point it is held by). The node swings this one from the hand (remote_hands/scene.py).
def flat_sweater():
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=70, y_subdivisions=56, size=1.0)
    ob = bpy.context.object
    ob.scale = (1.40, 0.70, 1.0)
    bpy.ops.object.transform_apply(scale=True)
    bm = bmesh.new(); bm.from_mesh(ob.data)
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if not inside(f.calc_center_median())], context="FACES")
    bm.to_mesh(ob.data); bm.free()
    return ob


bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sw = flat_sweater()
sw.location = (0.0, 0.0, 1.5)
pin = sw.vertex_groups.new(name="pin")
held = [v.index for v in sw.data.vertices if math.hypot(v.co.x, v.co.y - 0.02) < 0.05]
pin.add(held, 1.0, "REPLACE")
cl = sw.modifiers.new("cloth", "CLOTH")
cl.settings.quality = 8
cl.settings.mass = 0.4
cl.settings.bending_stiffness = 0.5
cl.settings.tension_stiffness = 12
cl.settings.air_damping = 3.0
cl.settings.vertex_group_mass = "pin"
cl.collision_settings.use_self_collision = True
cl.collision_settings.self_distance_min = 0.006
sc.frame_start, sc.frame_end = 1, 170
cl.point_cache.frame_start, cl.point_cache.frame_end = 1, 170
for f in range(1, 171):
    sc.frame_set(f)
me = bpy.data.meshes.new_from_object(sw.evaluated_get(bpy.context.evaluated_depsgraph_get()))
me.transform(sw.matrix_world)
at = sum((me.vertices[i].co for i in held), Vector()) / len(held)
ob = bpy.data.objects.new("hanging", me)
bpy.context.collection.objects.link(ob)
ob.modifiers.new("thick", "SOLIDIFY").thickness = 0.007
ob.modifiers.new("sub", "SUBSURF").levels = 1
me2 = bpy.data.meshes.new_from_object(ob.evaluated_get(bpy.context.evaluated_depsgraph_get()))
for v in me2.vertices:
    v.co = Vector(((v.co.x - at.x) * 100, -(v.co.y - at.y) * 100, (v.co.z - at.z) * 100))
me2.flip_normals()
for p in me2.polygons:
    p.use_smooth = True
lo = [min(v.co[i] for v in me2.vertices) for i in range(3)]; hi = [max(v.co[i] for v in me2.vertices) for i in range(3)]
out = bpy.data.objects.new("hanging_out", me2)
bpy.context.collection.objects.link(out)
bpy.ops.object.select_all(action="DESELECT"); out.select_set(True); bpy.context.view_layer.objects.active = out
bpy.ops.wm.obj_export(filepath=os.path.join(OUT, "sweater_hanging.obj"), export_selected_objects=True, forward_axis="Y", up_axis="Z",
                      export_materials=False, export_uv=True, export_normals=True, export_triangulated_mesh=True)
print("HANGING cm from", [round(v, 1) for v in lo], "to", [round(v, 1) for v in hi], len(me2.polygons))
