#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from tqdm import tqdm


import torch.backends.cuda
if not hasattr(torch.backends.cuda, "is_flash_attention_available"):
    torch.backends.cuda.is_flash_attention_available = lambda: False


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from metrics.gld_metrics import compute_abs_rel, compute_delta, compute_depth_rmse, compute_lpips, compute_psnr, compute_ssim


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RGB, depth, and pose metrics between two image folders.")
    parser.add_argument("--reference_dir", type=Path, help="Reference scene directory. Supports an images/ subdirectory and transforms.json.")
    parser.add_argument("--generated_dir", type=Path, help="Generated image directory. Supports direct image files or an images/ subdirectory.")
    parser.add_argument(
        "--vggt-repo",
        type=Path,
        default=Path("./thirdparty/vggt"),
        help="Path to the local VGGT repository.",
    )
    parser.add_argument("--vggt-model", type=str, default="facebook/VGGT-1B", help="VGGT checkpoint name or path.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--image-batch-size", type=int, default=8, help="Batch size for RGB metrics.")
    parser.add_argument("--preprocess-mode", choices=["crop", "pad"], default="crop", help="VGGT preprocessing mode.")
    parser.add_argument(
        "--gt-pose-is-w2c",
        action="store_true",
        help="Use GT poses as world-to-camera directly. By default, transform(s).json poses are assumed to be camera-to-world and will be inverted.",
    )
    parser.add_argument(
        "--droid-weights",
        type=Path,
        default=None,
        help="Optional path to DROID-SLAM weights for reprojection_error_metrics.py.",
    )
    parser.add_argument("--reprojection-stride", type=int, default=1, help="Frame stride for reprojection error.")
    parser.add_argument("--skip-reprojection", action="store_true", help="Skip reprojection error computation.")
    parser.add_argument(
        "--disable-first-pose-alignment",
        action="store_true",
        help="Disable canonicalizing GT/predicted poses by the first frame before error computation.",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optional path to save the aggregated metrics JSON.")
    return parser.parse_args()


def resolve_image_dir(folder: Path) -> Path:
    images_dir = folder / "images"
    if images_dir.is_dir():
        return images_dir
    return folder


def collect_images(folder: Path) -> List[Path]:
    image_dir = resolve_image_dir(folder)
    paths = [path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(paths)


def match_image_pairs(reference_dir: Path, generated_dir: Path) -> Tuple[List[Path], List[Path]]:
    reference_images = collect_images(reference_dir)
    generated_images = collect_images(generated_dir)

    if not reference_images:
        raise ValueError(f"No images found in reference directory: {reference_dir}")
    if not generated_images:
        raise ValueError(f"No images found in generated directory: {generated_dir}")

    if len(reference_images) != len(generated_images):
        raise ValueError(
            f"Frame count mismatch: reference has {len(reference_images)} frames, generated has {len(generated_images)} frames"
        )

    return reference_images, generated_images


def pil_to_tensor(path: Path) -> Tensor:
    image = Image.open(path)
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)
    image = image.convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def resize_tensor_image(image: Tensor, size_hw: Tuple[int, int]) -> Tensor:
    return F.interpolate(image.unsqueeze(0), size=size_hw, mode="bilinear", align_corners=False).squeeze(0)


def compute_rgb_metrics(reference_paths: Sequence[Path], generated_paths: Sequence[Path], device: torch.device, batch_size: int) -> Dict[str, float]:
    psnr_values: List[Tensor] = []
    ssim_values: List[Tensor] = []
    lpips_values: List[Tensor] = []

    for start in tqdm(range(0, len(reference_paths), batch_size), desc="RGB metrics"):
        ref_batch: List[Tensor] = []
        gen_batch: List[Tensor] = []
        batch_pairs = zip(reference_paths[start : start + batch_size], generated_paths[start : start + batch_size])

        for ref_path, gen_path in batch_pairs:
            ref_image = pil_to_tensor(ref_path)
            gen_image = pil_to_tensor(gen_path)
            if gen_image.shape[-2:] != ref_image.shape[-2:]:
                gen_image = resize_tensor_image(gen_image, ref_image.shape[-2:])
            ref_batch.append(ref_image)
            gen_batch.append(gen_image)

        ref_tensor = torch.stack(ref_batch, dim=0).to(device)
        gen_tensor = torch.stack(gen_batch, dim=0).to(device)

        psnr_values.append(compute_psnr(ref_tensor, gen_tensor).cpu())
        ssim_values.append(compute_ssim(ref_tensor, gen_tensor).cpu())
        lpips_values.append(compute_lpips(ref_tensor, gen_tensor).cpu())

    psnr_mean = torch.cat(psnr_values).mean().item()
    ssim_mean = torch.cat(ssim_values).mean().item()
    lpips_mean = torch.cat(lpips_values).mean().item()
    return {
        "psnr": psnr_mean,
        "ssim": ssim_mean,
        "lpips": lpips_mean,
    }


def load_vggt_components(vggt_repo: Path):
    repo_path = str(vggt_repo.resolve())
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    return VGGT, load_and_preprocess_images, pose_encoding_to_extri_intri


def resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    if device.type == "cuda":
        capability = torch.cuda.get_device_capability(device)
        if capability[0] >= 8:
            return torch.bfloat16
        return torch.float16
    return torch.float32


def infer_vggt_depth_and_pose(
    image_paths: Sequence[Path],
    model,
    load_and_preprocess_images,
    pose_encoding_to_extri_intri,
    device: torch.device,
    dtype: torch.dtype,
    preprocess_mode: str,
) -> Tuple[Tensor, Tensor, Tensor]:
    images = load_and_preprocess_images([str(path) for path in image_paths], mode=preprocess_mode).to(device)
    autocast_enabled = device.type == "cuda" and dtype != torch.float32

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=autocast_enabled, dtype=dtype):
            predictions = model(images)

    pose_enc = predictions["pose_enc"]
    extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
    depth = predictions["depth"][0, ..., 0]

    extrinsics = extrinsics[0].detach().cpu()
    intrinsics = intrinsics[0].detach().cpu()
    depth = depth.detach().cpu()
    return depth, extrinsics, intrinsics


def maybe_resize_depth(predicted: Tensor, ground_truth: Tensor) -> Tuple[Tensor, Tensor]:
    if predicted.shape[-2:] == ground_truth.shape[-2:]:
        return predicted, ground_truth

    resized_pred = F.interpolate(
        predicted.unsqueeze(1),
        size=ground_truth.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    return resized_pred, ground_truth


def compute_depth_metrics(predicted_depth: Tensor, ground_truth_depth: Tensor) -> Dict[str, float]:
    predicted_depth, ground_truth_depth = maybe_resize_depth(predicted_depth, ground_truth_depth)
    valid_mask = ground_truth_depth > 0
    eps = 1e-6
    predicted_depth = predicted_depth.clamp_min(eps)
    ground_truth_depth = ground_truth_depth.clamp_min(eps)

    abs_rel = compute_abs_rel(predicted_depth, ground_truth_depth, valid_mask).mean().item()
    rmse = compute_depth_rmse(predicted_depth, ground_truth_depth, valid_mask).mean().item()
    delta = compute_delta(predicted_depth, ground_truth_depth, threshold=1.25, valid_mask=valid_mask).mean().item()
    return {
        "depth_abs_rel": abs_rel,
        "depth_rmse": rmse,
        "depth_delta1": delta,
    }


def get_scaled_intrinsics_for_generated(
    intrinsics: Dict[str, float],
    generated_paths: Sequence[Path],
) -> Optional[List[float]]:
    required_keys = ["w", "h", "fx", "fy", "cx", "cy"]
    if not all(key in intrinsics for key in required_keys):
        return None

    first_generated = Image.open(generated_paths[0]).convert("RGB")
    generated_width, generated_height = first_generated.size
    gt_width = float(intrinsics["w"])
    gt_height = float(intrinsics["h"])

    scale_x = generated_width / gt_width
    scale_y = generated_height / gt_height

    return [
        float(intrinsics["fx"]) * scale_x,
        float(intrinsics["fy"]) * scale_y,
        float(intrinsics["cx"]) * scale_x,
        float(intrinsics["cy"]) * scale_y,
    ]


def compute_reprojection_metric(
    generated_paths: Sequence[Path],
    intrinsics: Dict[str, float],
    droid_weights: Optional[Path],
    stride: int,
) -> Dict[str, float]:
    droid_path = str(THIS_DIR / "thirdparty" / "DROID-SLAM" / "droid_slam")
    if droid_path not in sys.path:
        sys.path.insert(0, droid_path)

    from metrics.reprojection_error_metrics import ReprojectionErrorMetric

    metric = ReprojectionErrorMetric()
    metric._args.stride = stride

    scaled_intrinsics = get_scaled_intrinsics_for_generated(intrinsics, generated_paths)
    if scaled_intrinsics is not None:
        metric._args.calib = scaled_intrinsics

    reprojection_error = metric._compute_scores([str(path) for path in generated_paths])
    return {"reprojection_error": reprojection_error}


def load_gt_poses(reference_dir: Path, ordered_image_names: Sequence[str], gt_pose_is_w2c: bool) -> Tuple[Tensor, Dict[str, float]]:
    json_candidates = [reference_dir / "transforms.json", reference_dir / "transform.json"]
    json_path = next((path for path in json_candidates if path.is_file()), None)
    if json_path is None:
        raise FileNotFoundError(f"Could not find transforms.json or transform.json under {reference_dir}")

    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    intrinsics = {
        key: payload[key]
        for key in ["w", "h", "fx", "fy", "cx", "cy"]
        if key in payload
    }

    frames = payload.get("frames")
    if frames is None:
        raise ValueError(f"No frames field found in {json_path}")

    matrices: List[np.ndarray]
    if frames and isinstance(frames[0], dict):
        by_name = {}
        for frame in frames:
            matrix = frame.get("transform_matrix") or frame.get("matrix")
            file_path = frame.get("file_path") or frame.get("image_path") or frame.get("name")
            if matrix is None or file_path is None:
                raise ValueError(f"Frame entry missing matrix or file path in {json_path}")
            by_name[Path(file_path).name] = np.asarray(matrix, dtype=np.float32)
        matrices = [by_name[name] for name in ordered_image_names]
    else:
        matrices = [np.asarray(frame, dtype=np.float32) for frame in frames]
        if len(matrices) != len(ordered_image_names):
            raise ValueError(
                f"Pose count {len(matrices)} does not match image count {len(ordered_image_names)} in {reference_dir}"
            )

    poses = torch.from_numpy(np.stack(matrices, axis=0)).float()
    if not gt_pose_is_w2c:
        poses = torch.inverse(poses)
    return poses, intrinsics


def to_homogeneous_extrinsics(extrinsics_3x4: Tensor) -> Tensor:
    batch = extrinsics_3x4.shape[0]
    eye_row = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=extrinsics_3x4.dtype).view(1, 1, 4).repeat(batch, 1, 1)
    return torch.cat([extrinsics_3x4, eye_row], dim=1)


def canonicalize_w2c_poses(poses_w2c: Tensor) -> Tensor:
    first_inv = torch.inverse(poses_w2c[0])
    return torch.matmul(poses_w2c, first_inv.unsqueeze(0))


def compute_rotation_error_degrees(gt_rotations: Tensor, pred_rotations: Tensor) -> Tensor:
    relative = torch.matmul(gt_rotations, pred_rotations.transpose(1, 2))
    trace = relative[:, 0, 0] + relative[:, 1, 1] + relative[:, 2, 2]
    cosine = ((trace - 1.0) / 2.0).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def compute_translation_scale(gt_translations: Tensor, pred_translations: Tensor) -> float:
    numerator = (gt_translations * pred_translations).sum().item()
    denominator = (pred_translations * pred_translations).sum().item()
    if denominator < 1e-8:
        return 1.0
    return numerator / denominator


def compute_pose_metrics(gt_poses_w2c: Tensor, pred_extrinsics_w2c_3x4: Tensor, align_first_pose: bool) -> Dict[str, float]:
    pred_poses_w2c = to_homogeneous_extrinsics(pred_extrinsics_w2c_3x4.float())
    gt_poses_w2c = to_homogeneous_extrinsics(gt_poses_w2c.float())

    if align_first_pose:
        gt_poses_w2c = canonicalize_w2c_poses(gt_poses_w2c.float())
        pred_poses_w2c = canonicalize_w2c_poses(pred_poses_w2c.float())

    gt_rotations = gt_poses_w2c[:, :3, :3]
    pred_rotations = pred_poses_w2c[:, :3, :3]
    gt_translations = gt_poses_w2c[:, :3, 3]
    pred_translations = pred_poses_w2c[:, :3, 3]

    rotation_error = compute_rotation_error_degrees(gt_rotations, pred_rotations)
    scale = compute_translation_scale(gt_translations, pred_translations)
    translation_error = torch.linalg.norm(gt_translations - scale * pred_translations, dim=-1)

    return {
        "rotation_error_deg": rotation_error.mean().item(),
        "translation_error": translation_error.mean().item(),
        "translation_scale": scale,
    }


def build_summary(
    reference_dir: Path,
    generated_dir: Path,
    num_images: int,
    intrinsics: Dict[str, float],
    rgb_metrics: Dict[str, float],
    depth_metrics: Dict[str, float],
    pose_metrics: Dict[str, float],
    reprojection_metrics: Dict[str, float],
) -> Dict[str, object]:
    return {
        "reference_dir": str(reference_dir),
        "generated_dir": str(generated_dir),
        "num_images": num_images,
        "gt_intrinsics": intrinsics,
        **rgb_metrics,
        **depth_metrics,
        **pose_metrics,
        **reprojection_metrics,
    }


def print_summary(summary: Dict[str, object]) -> None:
    print("\nEvaluation summary")
    print(f"reference_dir: {summary['reference_dir']}")
    print(f"generated_dir: {summary['generated_dir']}")
    print(f"num_images: {summary['num_images']}")
    print(f"PSNR: {summary['psnr']:.6f}")
    print(f"SSIM: {summary['ssim']:.6f}")
    print(f"LPIPS: {summary['lpips']:.6f}")
    print(f"Depth AbsRel: {summary['depth_abs_rel']:.6f}")
    print(f"Depth RMSE: {summary['depth_rmse']:.6f}")
    print(f"Depth Delta1: {summary['depth_delta1']:.6f}")
    print(f"Rotation error (deg): {summary['rotation_error_deg']:.6f}")
    print(f"Translation error: {summary['translation_error']:.6f}")
    print(f"Translation scale: {summary['translation_scale']:.6f}")
    if "reprojection_error" in summary:
        print(f"Reprojection error: {summary['reprojection_error']:.6f}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    reference_paths, generated_paths = match_image_pairs(args.reference_dir, args.generated_dir)
    gt_poses_w2c, intrinsics = load_gt_poses(args.reference_dir, [path.name for path in reference_paths], args.gt_pose_is_w2c)

    rgb_metrics = compute_rgb_metrics(reference_paths, generated_paths, device, args.image_batch_size)

    VGGT, load_and_preprocess_images, pose_encoding_to_extri_intri = load_vggt_components(args.vggt_repo)
    model = VGGT.from_pretrained(args.vggt_model).to(device)
    model.eval()

    print("Running VGGT on reference images...")
    reference_depth, pesudo_GT_extrinsics, _ = infer_vggt_depth_and_pose(
        reference_paths,
        model,
        load_and_preprocess_images,
        pose_encoding_to_extri_intri,
        device,
        dtype,
        args.preprocess_mode,
    )

    print("Running VGGT on generated images...")
    generated_depth, generated_extrinsics, _ = infer_vggt_depth_and_pose(
        generated_paths,
        model,
        load_and_preprocess_images,
        pose_encoding_to_extri_intri,
        device,
        dtype,
        args.preprocess_mode,
    )

    depth_metrics = compute_depth_metrics(generated_depth, reference_depth)
    # pose_metrics = compute_pose_metrics(
    #     gt_poses_w2c,
    #     generated_extrinsics,
    #     align_first_pose=not args.disable_first_pose_alignment,
    # )

    pose_metrics = compute_pose_metrics(
        pesudo_GT_extrinsics,
        generated_extrinsics,
        align_first_pose=not args.disable_first_pose_alignment,
    )
    reprojection_metrics: Dict[str, float] = {}
    if not args.skip_reprojection:
        reprojection_metrics = compute_reprojection_metric(
            generated_paths,
            intrinsics,
            args.droid_weights,
            args.reprojection_stride,
        )

    summary = build_summary(
        args.reference_dir,
        args.generated_dir,
        len(reference_paths),
        intrinsics,
        rgb_metrics,
        depth_metrics,
        pose_metrics,
        reprojection_metrics,
    )
    print_summary(summary)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        print(f"Saved metrics to {args.output_json}")


if __name__ == "__main__":
    main()
