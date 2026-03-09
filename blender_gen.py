"""
blender_gen.py — LEGO Synthetic Data Generator
Runs inside Blender's headless Python environment.

Usage:
    blender --background --python blender_gen.py -- \
        --parts 3001,3002 \
        --count 100 \
        --output ./datasets/raw \
        --ldraw "C:/Program Files/Studio 2.0/ldraw" \
        --resolution 640x640 \
        --hdri_dir ./hdri
"""

import sys
import os
import argparse
import math
import random
import shutil

import bpy
import bmesh
from mathutils import Vector, Matrix
from bpy_extras.object_utils import world_to_camera_view


# ---------------------------------------------------------------------------
# Argument parsing  (everything after the "--" separator)
# ---------------------------------------------------------------------------

def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="LEGO Blender renderer")
    parser.add_argument("--parts",      required=True,
                        help="Comma-separated LDraw part IDs, e.g. 3001,3002")
    parser.add_argument("--count",      type=int, default=100,
                        help="Images to render per part")
    parser.add_argument("--output",     required=True,
                        help="Root output directory (datasets/raw)")
    parser.add_argument("--ldraw",      required=True,
                        help="LDraw library root, e.g. C:/Program Files/Studio 2.0/ldraw")
    parser.add_argument("--resolution", default="640x640",
                        help="WxH, e.g. 640x640")
    parser.add_argument("--hdri_dir",   default="",
                        help="Directory containing .hdr/.exr files (optional)")
    parser.add_argument("--samples",    type=int, default=96,
                        help="Cycles render samples (lower = faster)")
    parser.add_argument("--device_type", default="AUTO",
                        choices=["AUTO", "OPTIX", "CUDA", "HIP", "METAL", "CPU"],
                        help="GPU backend: AUTO tries OptiX→CUDA→HIP→Metal then CPU")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# LEGO colour palette  (approximate sRGB values for common LEGO colours)
# ---------------------------------------------------------------------------

LEGO_COLORS = [
    (0.800, 0.100, 0.100),  # Red
    (0.100, 0.200, 0.800),  # Blue
    (0.950, 0.850, 0.100),  # Yellow
    (0.100, 0.600, 0.150),  # Green
    (0.900, 0.900, 0.900),  # White
    (0.080, 0.080, 0.080),  # Black
    (0.500, 0.130, 0.050),  # Brown / Dark Orange
    (1.000, 0.600, 0.100),  # Orange
    (0.680, 0.100, 0.680),  # Purple
    (0.500, 0.500, 0.500),  # Dark Grey
    (0.780, 0.780, 0.780),  # Light Grey
    (0.100, 0.700, 0.700),  # Cyan
    (1.000, 0.700, 0.700),  # Pink / Light Red
    (0.400, 0.200, 0.050),  # Reddish Brown
    (0.200, 0.500, 0.200),  # Sand Green
    (0.000, 0.400, 0.600),  # Dark Blue
    (0.900, 0.700, 0.400),  # Tan
    (0.050, 0.300, 0.050),  # Dark Green
    (0.600, 0.800, 1.000),  # Light Blue
    (1.000, 0.900, 0.600),  # Cream / Light Yellow
]


# ---------------------------------------------------------------------------
# ImportLDraw add-on interface
# ---------------------------------------------------------------------------

# Module name registered by the TobyLobster ImportLDraw add-on.
# If your add-on registers under a different name, change this constant.
IMPORTLDRAW_MODULE = "io_scene_importldraw"


def _enable_importldraw_addon(ldraw_root: str) -> bool:
    """
    Attempt to enable the ImportLDraw add-on and point it at the library root.
    Returns True if the add-on is ready to use.
    """
    # Already enabled?
    if IMPORTLDRAW_MODULE in bpy.context.preferences.addons:
        _set_addon_ldraw_path(ldraw_root)
        return True

    # Try to enable it
    try:
        result = bpy.ops.preferences.addon_enable(module=IMPORTLDRAW_MODULE)
        if result == {"FINISHED"} and IMPORTLDRAW_MODULE in bpy.context.preferences.addons:
            _set_addon_ldraw_path(ldraw_root)
            print(f"[INFO] ImportLDraw add-on enabled ({IMPORTLDRAW_MODULE})", flush=True)
            return True
    except Exception as e:
        print(f"[INFO] Could not enable ImportLDraw add-on: {e}", flush=True)

    return False


def _set_addon_ldraw_path(ldraw_root: str):
    """Push the LDraw library path into the add-on's preferences."""
    try:
        prefs = bpy.context.preferences.addons[IMPORTLDRAW_MODULE].preferences
        # The preference key differs slightly between add-on versions; try both.
        for attr in ("ldrawPath", "ldraw_path", "LDRAW_DIRECTORY"):
            if hasattr(prefs, attr):
                setattr(prefs, attr, ldraw_root)
                break
    except Exception as e:
        print(f"[WARN] Could not set LDraw path in add-on prefs: {e}", flush=True)


def import_ldraw_part(dat_path: str, ldraw_root: str,
                      use_addon: bool = True) -> list:
    """
    Import a single LDraw .dat file and return the list of Blender mesh objects
    that were added to the scene.

    Strategy:
      1. Try bpy.ops.import_scene.importldraw  (requires add-on installed)
      2. Fall back to the custom recursive parser below
    """
    objects_before = set(bpy.data.objects)

    if use_addon and _enable_importldraw_addon(ldraw_root):
        try:
            bpy.ops.import_scene.importldraw(filepath=dat_path)
            new_objs = [o for o in bpy.data.objects if o not in objects_before]
            mesh_objs = [o for o in new_objs if o.type == "MESH"]
            if mesh_objs:
                print(f"[INFO] Imported via add-on: {len(mesh_objs)} mesh object(s)", flush=True)
                return mesh_objs
            else:
                print("[WARN] Add-on produced no mesh objects; falling back to parser", flush=True)
        except Exception as e:
            print(f"[WARN] Add-on import failed ({e}); falling back to parser", flush=True)
            # Clean up any partial import
            for o in bpy.data.objects:
                if o not in objects_before:
                    bpy.data.objects.remove(o, do_unlink=True)

    # --- Fallback: custom recursive parser ---
    part_id = os.path.splitext(os.path.basename(dat_path))[0]
    triangles = parse_ldraw_file(dat_path, ldraw_root)
    if not triangles:
        return []
    mesh_obj = build_mesh_from_triangles(triangles, part_id)
    return [mesh_obj]


# ---------------------------------------------------------------------------
# LDraw recursive importer  (fallback — used when add-on is unavailable)
# ---------------------------------------------------------------------------

def resolve_ldraw_file(ref_name: str, ldraw_root: str) -> str | None:
    """
    Resolve a type-1 filename reference to an actual path.
    LDraw files use backslashes; we normalise to forward slashes.
    Search order: parts/ → parts/s/ → p/ → p/48/
    """
    norm = ref_name.replace("\\", "/").strip()
    base = os.path.basename(norm)

    candidates = [
        os.path.join(ldraw_root, "parts",    norm),
        os.path.join(ldraw_root, "parts", "s", base),
        os.path.join(ldraw_root, "p",        norm),
        os.path.join(ldraw_root, "p", "48",  base),
        os.path.join(ldraw_root, "p",  "8",  base),
        os.path.join(ldraw_root, "p",  "4",  base),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _apply_transform(verts, mat: Matrix):
    """Apply a 4x4 Matrix to a list of Vector(x,y,z)."""
    return [(mat @ v) for v in verts]


def parse_ldraw_file(filepath: str, ldraw_root: str,
                     parent_transform: Matrix | None = None,
                     invert_winding: bool = False,
                     depth: int = 0) -> list:
    """
    Recursively parse a .dat file and return a flat list of triangles.
    Each triangle is a list of three mathutils.Vector (world-space coords).
    Coordinate system is converted to Blender (Y-up) inside this function.

    Uses a depth limit (not a visited set) so that the same primitive file
    (e.g. stud.dat) can be included many times with different transforms —
    which is exactly how LDraw studs, tubes and edges work.
    """
    if depth > 24:  # guard against genuinely circular references
        return []

    if parent_transform is None:
        # LDraw Y is down; flip it to Blender Y-up.
        # Also scale from LDU (1 LDU = 0.4 mm) to Blender metres.
        LDU_TO_M = 0.0004
        flip_y = Matrix.Scale(-1, 4, (0, 1, 0))
        scale  = Matrix.Scale(LDU_TO_M, 4)
        parent_transform = scale @ flip_y

    triangles = []
    invert_next = False

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                tokens = line.split()
                line_type = tokens[0]

                # --- BFC winding inversion flag ---
                if line_type == "0":
                    upper = line.upper()
                    if "BFC" in upper and "INVERTNEXT" in upper:
                        invert_next = True
                    continue

                # --- Subfile reference ---
                elif line_type == "1":
                    if len(tokens) < 15:
                        invert_next = False
                        continue
                    try:
                        vals = [float(t) for t in tokens[1:14]]
                    except ValueError:
                        invert_next = False
                        continue

                    # color (ignored for geometry), then position + 3x3 matrix
                    # tokens: 1 color x y z a b c d e f g h i filename...
                    x, y, z = vals[1], vals[2], vals[3]
                    a, b, c = vals[4], vals[5], vals[6]
                    d, e, f = vals[7], vals[8], vals[9]
                    g, h, i = vals[10], vals[11], vals[12]

                    local_mat = Matrix([
                        [a, b, c, x],
                        [d, e, f, y],
                        [g, h, i, z],
                        [0, 0, 0, 1],
                    ])
                    combined = parent_transform @ local_mat

                    ref_name = " ".join(tokens[14:])
                    child_path = resolve_ldraw_file(ref_name, ldraw_root)
                    if child_path:
                        child_invert = invert_winding ^ invert_next
                        triangles += parse_ldraw_file(
                            child_path, ldraw_root,
                            parent_transform=combined,
                            invert_winding=child_invert,
                            depth=depth + 1,
                        )
                    invert_next = False

                # --- Triangle ---
                elif line_type == "3":
                    if len(tokens) < 11:
                        continue
                    try:
                        coords = [float(t) for t in tokens[2:11]]
                    except ValueError:
                        continue
                    v1 = Vector((coords[0], coords[1], coords[2]))
                    v2 = Vector((coords[3], coords[4], coords[5]))
                    v3 = Vector((coords[6], coords[7], coords[8]))
                    tri = _apply_transform([v1, v2, v3], parent_transform)
                    if invert_winding:
                        tri = tri[::-1]
                    triangles.append(tri)

                # --- Quad (split into 2 triangles) ---
                elif line_type == "4":
                    if len(tokens) < 14:
                        continue
                    try:
                        coords = [float(t) for t in tokens[2:14]]
                    except ValueError:
                        continue
                    v1 = Vector((coords[0],  coords[1],  coords[2]))
                    v2 = Vector((coords[3],  coords[4],  coords[5]))
                    v3 = Vector((coords[6],  coords[7],  coords[8]))
                    v4 = Vector((coords[9],  coords[10], coords[11]))
                    for tri_verts in [[v1, v2, v3], [v1, v3, v4]]:
                        tri = _apply_transform(tri_verts, parent_transform)
                        if invert_winding:
                            tri = tri[::-1]
                        triangles.append(tri)

    except OSError as e:
        print(f"[WARN] Could not read {filepath}: {e}")

    return triangles


def build_mesh_from_triangles(triangles: list, name: str) -> bpy.types.Object:
    """
    Given a flat list of triangle vertex-triplets (mathutils.Vector),
    build a Blender mesh object and link it to the active collection.
    Duplicate vertices are NOT merged here for speed; use merge_by_distance
    if needed.
    """
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()

    for tri in triangles:
        try:
            bm_verts = [bm.verts.new(v) for v in tri]
            bm.faces.new(bm_verts)
        except ValueError:
            # Degenerate face — skip
            pass

    bm.to_mesh(mesh)
    bm.free()

    mesh.validate()
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    return obj


# ---------------------------------------------------------------------------
# Scene utilities
# ---------------------------------------------------------------------------

def clear_scene():
    """
    Remove all objects without using bpy.ops (ops require an active viewport
    context which is unreliable in headless/background mode after the first
    iteration).  Directly removing via bpy.data works in any context.
    """
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for block in list(bpy.data.meshes):
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in list(bpy.data.lights):
        if block.users == 0:
            bpy.data.lights.remove(block)
    for block in list(bpy.data.cameras):
        if block.users == 0:
            bpy.data.cameras.remove(block)
    for block in list(bpy.data.materials):
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in list(bpy.data.images):
        if block.users == 0:
            bpy.data.images.remove(block)


def clear_scene_keep_lego(lego_objs: list):
    """
    Remove everything EXCEPT the LEGO part objects.
    Used between renders of the same part to avoid re-importing geometry —
    the mesh/BVH stays in VRAM and GPU utilisation stays high.
    """
    lego_set = set(lego_objs)
    for obj in list(bpy.data.objects):
        if obj not in lego_set:
            bpy.data.objects.remove(obj, do_unlink=True)
    for block in list(bpy.data.lights):
        if block.users == 0:
            bpy.data.lights.remove(block)
    for block in list(bpy.data.cameras):
        if block.users == 0:
            bpy.data.cameras.remove(block)
    for block in list(bpy.data.materials):
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in list(bpy.data.images):
        if block.users == 0:
            bpy.data.images.remove(block)


def get_scene_bounds(objects) -> tuple[Vector, float]:
    """
    Return (center, max_radius) of all mesh objects in Blender world space.
    Forces a dependency graph update first so matrix_world is current.
    """
    # Flush any pending transforms (critical after import or scale changes)
    bpy.context.view_layer.update()

    all_verts = []
    for obj in objects:
        if obj.type != "MESH":
            continue
        for corner in obj.bound_box:
            all_verts.append(obj.matrix_world @ Vector(corner))

    if not all_verts:
        return Vector((0, 0, 0)), 1.0

    center = sum(all_verts, Vector()) / len(all_verts)
    radius = max((v - center).length for v in all_verts)
    return center, max(radius, 0.01)


def normalize_part_to_unit_scale(mesh_objs, target_radius: float = 0.08):
    """
    Scale all LEGO part objects so the bounding radius equals target_radius.
    This makes the camera-distance formula work regardless of whether the
    ImportLDraw add-on imported at 1 LDU=1 BU or 1 LDU=0.0004 m.
    target_radius=0.08 m gives a nicely framed ~16 cm brick.
    """
    center, radius = get_scene_bounds(mesh_objs)
    if radius < 1e-6:
        print("[WARN] Part has zero bounding radius — skipping normalisation", flush=True)
        return

    scale_factor = target_radius / radius
    print(f"[INFO] Part radius={radius:.5f} → scaling by {scale_factor:.4f}", flush=True)

    for obj in mesh_objs:
        obj.scale = (
            obj.scale[0] * scale_factor,
            obj.scale[1] * scale_factor,
            obj.scale[2] * scale_factor,
        )
        # Re-centre at world origin so camera sphere maths is clean
        obj.location = (
            (obj.location[0] - center.x) * scale_factor,
            (obj.location[1] - center.y) * scale_factor,
            (obj.location[2] - center.z) * scale_factor,
        )

    bpy.context.view_layer.update()


# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------

def make_lego_material(color_rgb: tuple) -> bpy.types.Material:
    mat = bpy.data.materials.new("LEGO_Plastic")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")

    r, g, b = color_rgb
    bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
    bsdf.inputs["Roughness"].default_value  = 0.2
    bsdf.inputs["IOR"].default_value        = 1.46

    # Blender 4.x renamed Subsurface → Subsurface Weight
    for key in ("Subsurface Weight", "Subsurface"):
        if key in bsdf.inputs:
            bsdf.inputs[key].default_value = 0.1
            break

    return mat


def update_lego_material_color(mat: bpy.types.Material, color_rgb: tuple):
    """
    Update an existing LEGO material's base color in-place.
    This avoids creating a new material object each frame, which would
    trigger a full Cycles shader recompile and kill GPU utilisation.
    """
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        r, g, b = color_rgb
        bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)


def assign_material_to_all_meshes(mat: bpy.types.Material, objects):
    for obj in objects:
        if obj.type == "MESH":
            obj.data.materials.clear()
            obj.data.materials.append(mat)


# ---------------------------------------------------------------------------
# Camera  (create once per part, update transform each frame)
# ---------------------------------------------------------------------------

def setup_camera_once(center: Vector, radius: float) -> bpy.types.Object:
    """Create the scene camera using bpy.data (headless-safe). Call once per part."""
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 50.0
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    update_camera(cam_obj, center, radius)
    return cam_obj


def update_camera(cam_obj: bpy.types.Object, center: Vector, radius: float):
    """Randomise camera position/rotation in-place — no object creation."""
    yaw   = random.uniform(0.0, 2 * math.pi)
    pitch = random.uniform(math.radians(5), math.radians(80))
    dist  = radius * random.uniform(3.8, 6.0)

    x = center.x + dist * math.cos(pitch) * math.cos(yaw)
    y = center.y + dist * math.cos(pitch) * math.sin(yaw)
    z = center.z + dist * math.sin(pitch)

    cam_obj.location = (x, y, z)
    direction = center - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    roll = random.uniform(-math.radians(15), math.radians(15))
    cam_obj.rotation_euler.rotate_axis("Z", roll)


# ---------------------------------------------------------------------------
# Lighting  (create once per part, update transforms/energy each frame)
# ---------------------------------------------------------------------------

def _point_light_at(obj, target: Vector):
    """Rotate obj so its -Z axis points at target."""
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def setup_lighting_once(center: Vector, radius: float) -> list:
    """
    Create a 3-point lighting rig using bpy.data (headless-safe, no bpy.ops).
    Call once per part; use randomize_lighting() each frame to update values.
    """
    lights = []
    for name, ltype in [("Key", "AREA"), ("Fill", "AREA"), ("Rim", "POINT")]:
        ldata = bpy.data.lights.new(name, type=ltype)
        lobj  = bpy.data.objects.new(name, ldata)
        bpy.context.collection.objects.link(lobj)
        lights.append(lobj)
    randomize_lighting(lights, center, radius)
    return lights


def randomize_lighting(lights: list, center: Vector, radius: float):
    """Update light positions/energies in-place — no object creation."""
    key_obj, fill_obj, rim_obj = lights
    dist = radius * 3.0
    base = (radius ** 2) * 650.0

    key_loc = center + Vector((
        dist * random.uniform(0.5,  1.5),
        dist * random.uniform(-1.0, -0.3),
        dist * random.uniform(0.5,  1.5),
    ))
    key_obj.location      = key_loc
    key_obj.data.energy   = base * random.uniform(0.8, 1.8)
    key_obj.data.size     = radius * random.uniform(2.0, 4.0)
    key_obj.data.color    = (1.0, random.uniform(0.88, 0.96), random.uniform(0.75, 0.90))
    _point_light_at(key_obj, center)

    fill_loc = center + Vector((
        dist * random.uniform(-1.5, -0.5),
        dist * random.uniform(0.3,   1.0),
        dist * random.uniform(-0.3,  0.5),
    ))
    fill_obj.location     = fill_loc
    fill_obj.data.energy  = base * random.uniform(0.2, 0.55)
    fill_obj.data.size    = radius * random.uniform(3.0, 6.0)
    fill_obj.data.color   = (random.uniform(0.78, 0.88), random.uniform(0.88, 0.96), 1.0)
    _point_light_at(fill_obj, center)

    rim_loc = center + Vector((
        dist * random.uniform(-0.5, 0.5),
        dist * random.uniform(0.8,  1.5),
        dist * random.uniform(0.8,  1.5),
    ))
    rim_obj.location     = rim_loc
    rim_obj.data.energy  = base * random.uniform(0.3, 0.9)
    rim_obj.data.color   = (1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# Background / HDRI
# ---------------------------------------------------------------------------

def _ensure_world_nodes() -> tuple:
    """
    Ensure the world shader graph has the required nodes and return
    (bg_node, env_node_or_None).  Reuses existing nodes so Cycles
    does not recompile the world shader on every frame.
    """
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    # Locate or create required nodes
    bg_node  = nodes.get("Background")  or nodes.new("ShaderNodeBackground")
    bg_node.name = "Background"
    out_node = nodes.get("World Output") or nodes.new("ShaderNodeOutputWorld")
    out_node.name = "World Output"

    # Ensure Background → Output is connected
    if not out_node.inputs["Surface"].links:
        links.new(bg_node.outputs["Background"], out_node.inputs["Surface"])

    # Optional environment texture node
    env_node   = nodes.get("EnvTex")
    coord_node = nodes.get("TexCoord")

    if env_node is None:
        env_node = nodes.new("ShaderNodeTexEnvironment")
        env_node.name = "EnvTex"
    if coord_node is None:
        coord_node = nodes.new("ShaderNodeTexCoord")
        coord_node.name = "TexCoord"
    if not env_node.inputs["Vector"].links:
        links.new(coord_node.outputs["Generated"], env_node.inputs["Vector"])

    return bg_node, env_node


def setup_background(hdri_files: list):
    bg_node, env_node = _ensure_world_nodes()
    nodes = bpy.context.scene.world.node_tree.nodes
    links = bpy.context.scene.world.node_tree.links

    if hdri_files:
        try:
            # Swap the image in the existing env node — no node graph rebuild
            img_path = random.choice(hdri_files)
            env_node.image = bpy.data.images.load(img_path, check_existing=True)
            # Connect env → bg if not already linked
            if not bg_node.inputs["Color"].links:
                links.new(env_node.outputs["Color"], bg_node.inputs["Color"])
            bg_node.inputs["Strength"].default_value = random.uniform(0.5, 2.0)
            return
        except Exception as e:
            print(f"[WARN] Could not load HDRI: {e}")

    # Fallback: solid colour — disconnect env node so bg color is used
    for lnk in list(bg_node.inputs["Color"].links):
        links.remove(lnk)
    r, g, b = random.random(), random.random(), random.random()
    bg_node.inputs["Color"].default_value    = (r, g, b, 1.0)
    bg_node.inputs["Strength"].default_value = random.uniform(0.3, 1.5)


# ---------------------------------------------------------------------------
# Distractor objects  (fixed pool — create once, randomise each frame)
# ---------------------------------------------------------------------------

MAX_DISTRACTORS = 4


def setup_distractor_pool(n: int = MAX_DISTRACTORS) -> list:
    """
    Pre-create N distractor cube objects using bpy.data (headless-safe).
    All start hidden. Call randomize_distractors() each frame to show/move them.
    Using a fixed pool avoids creating/destroying mesh objects per frame, which
    would force a top-level BVH rebuild and stall the GPU.
    """
    pool = []
    for idx in range(n):
        me = bpy.data.meshes.new(f"Distractor_{idx}")
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        bm.to_mesh(me)
        bm.free()
        me.update()

        obj = bpy.data.objects.new(f"Distractor_{idx}", me)
        bpy.context.collection.objects.link(obj)

        mat = bpy.data.materials.new(f"DistMat_{idx}")
        mat.use_nodes = True
        me.materials.append(mat)

        obj.hide_render = True
        pool.append(obj)
    return pool


def randomize_distractors(pool: list, center: Vector, radius: float):
    """Show a random subset of the distractor pool and update their transforms."""
    n_active = random.randint(0, len(pool))
    for idx, obj in enumerate(pool):
        if idx < n_active:
            angle  = random.uniform(0, 2 * math.pi)
            spread = radius * random.uniform(2.5, 6.0)
            obj.location = (
                center.x + spread * math.cos(angle),
                center.y + spread * math.sin(angle),
                center.z + random.uniform(-radius, radius),
            )
            s = radius * random.uniform(0.2, 0.8)
            obj.scale = (s, s, s)
            bsdf = obj.data.materials[0].node_tree.nodes.get("Principled BSDF")
            if bsdf:
                bsdf.inputs["Base Color"].default_value = (
                    random.random(), random.random(), random.random(), 1.0
                )
                bsdf.inputs["Roughness"].default_value = random.uniform(0.1, 0.9)
            obj.hide_render = False
        else:
            obj.hide_render = True


# ---------------------------------------------------------------------------
# Render settings
# ---------------------------------------------------------------------------

def _setup_gpu(scene, preferred_type: str = "AUTO") -> bool:
    """
    Enable GPU rendering in Cycles.
    Tries backends in performance order: OPTIX → CUDA → HIP → METAL.
    Returns True if a GPU was successfully enabled.
    NOTE: For reliable GPU use in headless mode, open Blender GUI once,
    set Cycles GPU in Preferences → System, and Save Preferences.
    """
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
    except KeyError:
        print("[WARN] Cycles add-on not found — rendering on CPU.", flush=True)
        return False

    types_to_try = (
        ["OPTIX", "CUDA", "HIP", "METAL"] if preferred_type == "AUTO"
        else [preferred_type]
    )

    for dtype in types_to_try:
        if dtype == "CPU":
            break  # explicit CPU request — skip GPU search
        try:
            prefs.compute_device_type = dtype
            prefs.get_devices()  # must be called after changing type

            all_devices = list(prefs.devices)
            print(f"[INFO] {dtype} devices found: "
                  f"{[f'{d.name}({d.type})' for d in all_devices]}", flush=True)

            gpu_devices = [d for d in all_devices if d.type != "CPU"]
            if not gpu_devices:
                print(f"[INFO] No GPU devices under {dtype}, trying next backend.", flush=True)
                continue

            # Enable GPU devices only (pure GPU path is fastest)
            for d in all_devices:
                d.use = d.type != "CPU"
            scene.cycles.device = "GPU"
            print(
                f"[INFO] GPU rendering active — backend: {dtype}  "
                f"device(s): {', '.join(d.name for d in gpu_devices)}",
                flush=True,
            )
            # Save preferences so future headless runs pick up GPU automatically
            try:
                bpy.ops.wm.save_userpref()
                print("[INFO] GPU preferences saved to Blender user config.", flush=True)
            except Exception as e:
                print(f"[INFO] Could not save prefs (non-fatal): {e}", flush=True)
            return dtype          # ← return active backend string
        except Exception as e:
            print(f"[INFO] {dtype} unavailable: {e}", flush=True)

    scene.cycles.device = "CPU"
    print(
        "[WARN] No GPU backend initialised — rendering on CPU.\n"
        "[WARN] Fix: open Blender GUI → Edit → Preferences → System → "
        "Cycles Render Devices → select OptiX/CUDA → Save Preferences.",
        flush=True,
    )
    return "CPU"


def configure_render(W: int, H: int, samples: int, output_path: str,
                     gpu_backend: str = "CPU"):
    scene = bpy.context.scene
    scene.render.resolution_x   = W
    scene.render.resolution_y   = H
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath       = output_path
    scene.cycles.samples        = samples

    # Reuse compiled shaders/BVH across renders — critical for GPU efficiency
    scene.render.use_persistent_data = True

    # Denoising: use OptiX (GPU) denoiser when available so it doesn't stall on CPU
    scene.cycles.use_denoising = True
    try:
        scene.cycles.denoiser = "OPTIX" if gpu_backend == "OPTIX" else "OPENIMAGEDENOISE"
    except Exception:
        pass  # attribute absent in older Blender builds

    # Colour management
    scene.view_settings.exposure = random.uniform(-0.5, 0.5)
    scene.view_settings.gamma    = 1.0


# ---------------------------------------------------------------------------
# YOLO auto-labeling
# ---------------------------------------------------------------------------

def compute_yolo_label(mesh_objects, scene, camera, class_id: int) -> str | None:
    """
    Project all mesh bounding-box corners to camera space and compute the
    2-D axis-aligned bounding box in normalised [0,1] image coordinates.
    Returns a YOLO label string, or None if projection fails.
    """
    xs, ys = [], []
    for obj in mesh_objects:
        if obj.type != "MESH":
            continue
        for corner in obj.bound_box:
            world_pt = obj.matrix_world @ Vector(corner)
            co_2d = world_to_camera_view(scene, camera, world_pt)
            xs.append(co_2d.x)
            # world_to_camera_view returns y=0 at bottom; YOLO y=0 is top
            ys.append(1.0 - co_2d.y)

    if not xs:
        return None

    x_min = max(0.0, min(xs))
    x_max = min(1.0, max(xs))
    y_min = max(0.0, min(ys))
    y_max = min(1.0, max(ys))

    w = x_max - x_min
    h = y_max - y_min

    if w < 1e-4 or h < 1e-4:
        return None

    xc = (x_min + x_max) / 2.0
    yc = (y_min + y_max) / 2.0
    return f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


# ---------------------------------------------------------------------------
# Main render loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    parts  = [p.strip() for p in args.parts.split(",") if p.strip()]
    W, H   = (int(v) for v in args.resolution.split("x"))
    output = os.path.abspath(args.output)
    ldraw  = os.path.abspath(args.ldraw)

    # --- One-time render engine + GPU setup ---
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    gpu_backend = _setup_gpu(scene, args.device_type)

    # Collect HDRI files
    hdri_files = []
    if args.hdri_dir and os.path.isdir(args.hdri_dir):
        for fname in os.listdir(args.hdri_dir):
            if fname.lower().endswith((".hdr", ".exr")):
                hdri_files.append(os.path.join(args.hdri_dir, fname))

    total_renders = len(parts) * args.count
    done = 0

    for class_id, part_id in enumerate(parts):
        dat_path = os.path.join(ldraw, "parts", f"{part_id}.dat")
        if not os.path.isfile(dat_path):
            print(f"[ERROR] Part not found: {dat_path}", flush=True)
            continue

        img_dir   = os.path.join(output, part_id, "images")
        label_dir = os.path.join(output, part_id, "labels")
        os.makedirs(img_dir,   exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        # --- Import part ONCE per part_id ---
        # Keeping the mesh in the scene avoids re-importing and re-building
        # the BVH acceleration structure every frame, which is the main reason
        # GPU utilisation is low (GPU idles while CPU rebuilds the scene).
        clear_scene()
        lego_objs = import_ldraw_part(dat_path, ldraw)
        if not lego_objs:
            print(f"[WARN] No geometry for part {part_id}", flush=True)
            continue

        normalize_part_to_unit_scale(lego_objs)
        center, radius = get_scene_bounds(lego_objs)
        print(f"[INFO] Part {part_id} ready — rendering {args.count} images", flush=True)

        # --- Build persistent scene objects ONCE per part ---
        # Nothing is created or destroyed between frames; only values are updated.
        # This keeps the BVH fully cached so the GPU stays busy every render.
        part_mat    = make_lego_material(random.choice(LEGO_COLORS))
        assign_material_to_all_meshes(part_mat, lego_objs)
        cam_obj     = setup_camera_once(center, radius)
        lights      = setup_lighting_once(center, radius)
        distractors = setup_distractor_pool()

        # Set up render settings once (output path updated per-frame below)
        configure_render(W, H, args.samples, "", gpu_backend)

        for i in range(args.count):
            img_name   = f"{part_id}_{i:04d}.png"
            label_name = f"{part_id}_{i:04d}.txt"
            img_path   = os.path.join(img_dir,   img_name)
            label_path = os.path.join(label_dir, label_name)

            # Pure in-place updates — ZERO object creation/destruction this frame
            update_lego_material_color(part_mat, random.choice(LEGO_COLORS))
            update_camera(cam_obj, center, radius)
            randomize_lighting(lights, center, radius)
            setup_background(hdri_files)
            randomize_distractors(distractors, center, radius)

            bpy.context.scene.render.filepath       = img_path
            bpy.context.scene.view_settings.exposure = random.uniform(-0.5, 0.5)

            bpy.ops.render.render(write_still=True)

            # --- Label ---
            scene     = bpy.context.scene
            label_str = compute_yolo_label(lego_objs, scene, cam_obj, class_id)
            if label_str:
                with open(label_path, "w") as lf:
                    lf.write(label_str + "\n")
            else:
                print(f"[WARN] No valid 2D bbox for {part_id} render {i}", flush=True)

            done += 1
            progress = done / total_renders
            print(f"PROGRESS:{part_id}:{i + 1}/{args.count} ({progress:.1%})", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
