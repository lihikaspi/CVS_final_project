import blenderproc as bproc
from blenderproc.python.camera import CameraUtility
import numpy as np
import json
import csv
import random
import cv2
import bpy
from bpy_extras.object_utils import world_to_camera_view
from pathlib import Path

bproc.init()

# ==========================================
# 1. INITIALIZATION & PATHS
# ==========================================
BASE_DIR = Path.cwd()
MODELS_DIR = BASE_DIR / "data" / "surgical_tools_models"
CAM_FILE = BASE_DIR / "data" / "camera.json"
ROOT_OUTPUT_DIR = BASE_DIR / "output_synthetic"
ROOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# COCO2017 background pool (adjust to wherever /datashare/project points on your VM)
COCO_DIR = BASE_DIR / "data" / "train2017"
COCO_EXTS = {".jpg", ".jpeg", ".png"}

# Path for the global statistics CSV file
CSV_STATS_PATH = ROOT_OUTPUT_DIR / "generation_stats.csv"

# Set camera intrinsics
with open(CAM_FILE, "r") as file:
    camera_params = json.load(file)

cam_w, cam_h = camera_params["width"], camera_params["height"]
K = np.array([
    [camera_params["fx"], 0, camera_params["cx"]],
    [0, camera_params["fy"], camera_params["cy"]],
    [0, 0, 1]
])
bproc.camera.set_resolution(cam_w, cam_h)
CameraUtility.set_intrinsics_from_K_matrix(K, cam_w, cam_h)
bproc.renderer.set_output_format(enable_transparency=True)
bproc.renderer.set_max_amount_of_samples(100)

# Load the COCO background pool once (not per-sample) for speed
coco_files = sorted([p for p in COCO_DIR.glob("*") if p.suffix.lower() in COCO_EXTS])
if not coco_files:
    raise FileNotFoundError(
        f"No background images found in {COCO_DIR}. "
        f"Point COCO_DIR at your COCO2017 image folder before running."
    )


def project_pt_to_2d(pt_3d):
    """Projects 3D world coordinates to 2D pixel coordinates and checks depth."""
    scene = bpy.context.scene
    cam = bpy.context.scene.camera
    co_ndc = world_to_camera_view(scene, cam, pt_3d)
    # co_ndc.z > 0 ensures the point is in front of the camera plane
    return [co_ndc.x * cam_w, (1.0 - co_ndc.y) * cam_h], (co_ndc.z > 0)


def is_valid_landmark(pt_2d, in_front, w, h, margin=15):
    """Ensures keypoint is inside image bounds and in front of camera."""
    if not in_front:
        return False
    x, y = pt_2d
    return (margin <= x < (w - margin)) and (margin <= y < (h - margin))


def load_random_background(w, h):
    """Loads a random COCO image and resizes/crops it to (w, h), 3-channel BGR."""
    bg_path = random.choice(coco_files)
    bg = cv2.imread(str(bg_path))
    if bg is None:
        # Corrupt/unreadable file - fall back to a neutral gray background
        return np.full((h, w, 3), 127, dtype=np.uint8), bg_path.name

    bg_h, bg_w = bg.shape[:2]
    scale = max(w / bg_w, h / bg_h)
    new_w, new_h = int(np.ceil(bg_w * scale)), int(np.ceil(bg_h * scale))
    bg_resized = cv2.resize(bg, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center-crop to exactly (w, h)
    x0 = (new_w - w) // 2
    y0 = (new_h - h) // 2
    bg_cropped = bg_resized[y0:y0 + h, x0:x0 + w]
    return bg_cropped, bg_path.name


def composite_over_background(rgba_img, bg_bgr):
    """Alpha-composites a BGRA render over a BGR background. Returns a BGR image."""
    rgb = rgba_img[..., :3].astype(np.float32)
    alpha = (rgba_img[..., 3:4].astype(np.float32)) / 255.0
    bg = bg_bgr.astype(np.float32)

    out = rgb * alpha + bg * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


# ==========================================
# 2. MAIN GENERATION LOOP
# ==========================================
def generate_dataset(tool_name, num_samples=250):
    tool_dir = MODELS_DIR / tool_name
    obj_files = sorted(list(tool_dir.glob("*.obj")))

    tool_output_dir = ROOT_OUTPUT_DIR / tool_name
    tool_output_dir.mkdir(parents=True, exist_ok=True)

    # Directory for visual inspection deliverable
    vis_output_dir = ROOT_OUTPUT_DIR / f"{tool_name}_visualizations"
    vis_output_dir.mkdir(parents=True, exist_ok=True)

    json_path = tool_output_dir / f"{tool_name}_annotations.json"

    # Load existing annotations if continuing a previous run
    if json_path.exists():
        with open(json_path, "r") as f:
            annotations = json.load(f)
    else:
        annotations = []

    # Find starting index based on existing images
    existing_images = list(tool_output_dir.glob(f"{tool_name}_*.png"))
    if existing_images:
        indices = []
        for img_path in existing_images:
            try:
                indices.append(int(img_path.stem.split("_")[-1]))
            except ValueError:
                continue
        start_idx = max(indices) + 1 if indices else 0
    else:
        start_idx = 0

    # Verified Blender Vertex IDs across deformed meshes
    if tool_name == "tweezers":
        idx_tip1 = 183922
        idx_tip2 = 113535
        idx_joint = 245256
    else:  # needle_holder
        idx_tip1 = 146985
        idx_tip2 = 78113
        idx_joint = 157620

    csv_exists = CSV_STATS_PATH.exists()
    csv_file = open(CSV_STATS_PATH, mode="a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)

    if not csv_exists:
        csv_writer.writerow([
            "image_name", "tool", "source_obj", "background_source", "light_energy",
            "metallic", "roughness", "blur_kernel", "noise_std"
        ])

    samples_generated = 0
    max_attempts = num_samples * 20  # safety net against runaway resampling
    attempts = 0

    while samples_generated < num_samples:
        attempts += 1
        if attempts > max_attempts:
            print(f"[{tool_name}] WARNING: exceeded max attempts ({max_attempts}). "
                  f"Stopping early at {samples_generated}/{num_samples} samples.")
            break

        bproc.clean_up(clean_up_camera=False)
        current_idx = start_idx + samples_generated

        # 1. Articulation state selection
        selected_obj_path = random.choice(obj_files)
        tool_obj = bproc.loader.load_obj(str(selected_obj_path))[0]
        tool_obj.set_location([0, 0, 0])

        # 2. Material Randomization
        roughness_val = random.uniform(0.05, 0.4)
        mat = tool_obj.get_materials()[0]
        mat.set_principled_shader_value("Roughness", roughness_val)
        mat.set_principled_shader_value("Metallic", 1.0)

        # 3. Lighting Randomization
        light_energy = random.uniform(100, 1000)
        light = bproc.types.Light()
        light.set_type("POINT")
        light.set_location(bproc.sampler.shell(
            center=tool_obj.get_location(),
            radius_min=1.0, radius_max=5.0,
            elevation_min=1, elevation_max=89
        ))
        light.set_energy(light_energy)

        # 4. Camera Sampling
        location = bproc.sampler.shell(
            center=tool_obj.get_location(),
            radius_min=4.0, radius_max=7.5,
            elevation_min=-75, elevation_max=75
        )
        lookat_point = tool_obj.get_location() + np.random.uniform([-0.05, -0.05, -0.05], [0.05, 0.05, 0.05])
        rotation_matrix = bproc.camera.rotation_from_forward_vec(
            lookat_point - location,
            inplane_rot=np.random.uniform(-0.5, 0.5)
        )
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        bproc.camera.add_camera_pose(cam2world_matrix)

        bpy.context.view_layer.update()

        # 5. Extract and Validate Landmarks
        bpy_obj = tool_obj.blender_obj
        p3d_tip1 = bpy_obj.matrix_world @ bpy_obj.data.vertices[idx_tip1].co
        p3d_tip2 = bpy_obj.matrix_world @ bpy_obj.data.vertices[idx_tip2].co
        p3d_joint = bpy_obj.matrix_world @ bpy_obj.data.vertices[idx_joint].co

        pt1, in_front1 = project_pt_to_2d(p3d_tip1)
        pt2, in_front2 = project_pt_to_2d(p3d_tip2)
        pt_j, in_front_j = project_pt_to_2d(p3d_joint)

        # Re-sample pose if any landmark falls out of camera FOV
        if not (is_valid_landmark(pt1, in_front1, cam_w, cam_h) and
                is_valid_landmark(pt2, in_front2, cam_w, cam_h) and
                is_valid_landmark(pt_j, in_front_j, cam_w, cam_h)):
            continue

        # 6. Render (transparent RGBA)
        data = bproc.renderer.render()
        color_img = data["colors"][0]

        if color_img.shape[-1] == 4:
            color_img = cv2.cvtColor(color_img, cv2.COLOR_RGBA2BGRA)
        else:
            # Renderer didn't return alpha for some reason - can't composite, skip
            print(f"[{tool_name}] WARNING: render has no alpha channel, skipping sample.")
            continue

        # 7. Image-space augmentations (RGB channels only - alpha must stay clean
        #    for compositing, otherwise noise leaks into the "empty" region as
        #    visible speckle once blended onto a background)
        blur_kernel = 0
        if random.random() > 0.5:
            blur_kernel = random.choice([3, 5])
            color_img[..., :3] = cv2.GaussianBlur(color_img[..., :3], (blur_kernel, blur_kernel), 0)

        noise_std = 0.0
        if random.random() > 0.5:
            noise_std = random.uniform(5, 15)
            noise = np.random.normal(0, noise_std, color_img[..., :3].shape).astype(np.int16)
            color_img[..., :3] = np.clip(
                color_img[..., :3].astype(np.int16) + noise, 0, 255
            ).astype(np.uint8)

        # 8. Composite onto a random COCO background
        bg_img, bg_name = load_random_background(cam_w, cam_h)
        composited_img = composite_over_background(color_img, bg_img)

        # 9. Save Generated Image (final training sample - has a background)
        img_filename = f"{tool_name}_{current_idx:04d}.png"
        cv2.imwrite(str(tool_output_dir / img_filename), composited_img)

        # 10. Record Annotations
        annotations.append({
            "image_id": img_filename,
            "tool": tool_name,
            "keypoints": {
                "tip_1": [round(pt1[0], 2), round(pt1[1], 2)],
                "tip_2": [round(pt2[0], 2), round(pt2[1], 2)],
                "joint_or_base": [round(pt_j[0], 2), round(pt_j[1], 2)]
            }
        })

        # 11. Record CSV Stats
        csv_writer.writerow([
            img_filename,
            tool_name,
            selected_obj_path.name,
            bg_name,
            round(light_energy, 2),
            1.0,
            round(roughness_val, 4),
            blur_kernel,
            round(noise_std, 2)
        ])

        # 12. Visual Inspection Deliverable (first 15 samples per tool)
        #     Drawn on the composited image, since that's the actual training sample.
        if samples_generated < 15:
            vis_img = composited_img.copy()
            p1_int = (int(pt1[0]), int(pt1[1]))
            p2_int = (int(pt2[0]), int(pt2[1]))
            pj_int = (int(pt_j[0]), int(pt_j[1]))

            # Red: Tip 1, Blue: Tip 2, Green: Joint/Base
            cv2.circle(vis_img, p1_int, 6, (0, 0, 255), -1)
            cv2.circle(vis_img, p2_int, 6, (255, 0, 0), -1)
            cv2.circle(vis_img, pj_int, 6, (0, 255, 0), -1)

            cv2.imwrite(str(vis_output_dir / f"vis_{img_filename}"), vis_img)

        samples_generated += 1
        print(f"[{tool_name}] Generated sample {samples_generated}/{num_samples} -> {img_filename}")

    csv_file.close()
    with open(json_path, "w") as f:
        json.dump(annotations, f, indent=4)


if __name__ == "__main__":
    generate_dataset("tweezers", 250)
    generate_dataset("needle_holder", 250)