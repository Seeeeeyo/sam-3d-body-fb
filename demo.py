# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
SAM 3D Body Demo - Single Image Human Mesh Recovery

This demo supports various optimization modes for Jetson deployment:
- body-only inference (faster, no hand refinement)
- depth-based segmentation (instead of SAM)
- direct camera intrinsics (instead of MoGe2 FOV estimation)
- shape caching (calibration + runtime modes)
"""
import argparse
import os
from glob import glob
from typing import Optional, Tuple

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".sl"],
    pythonpath=True,
    dotenv=True,
)

import cv2
import numpy as np
import torch
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from tools.vis_utils import visualize_sample, visualize_sample_together
from tqdm import tqdm


def parse_camera_intrinsics(intrinsics_str: str) -> Optional[Tuple[float, ...]]:
    """Parse camera intrinsics from comma-separated string 'fx,fy,cx,cy'."""
    if not intrinsics_str:
        return None
    try:
        values = [float(v.strip()) for v in intrinsics_str.split(",")]
        if len(values) != 4:
            raise ValueError(f"Expected 4 values (fx,fy,cx,cy), got {len(values)}")
        return tuple(values)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid intrinsics format: {e}")


def main(args):
    if args.output_folder == "":
        output_folder = os.path.join("./output", os.path.basename(args.image_folder))
    else:
        output_folder = args.output_folder

    os.makedirs(output_folder, exist_ok=True)

    # Use command-line args or environment variables
    mhr_path = args.mhr_path or os.environ.get("SAM3D_MHR_PATH", "")
    detector_path = args.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
    segmentor_path = args.segmentor_path or os.environ.get("SAM3D_SEGMENTOR_PATH", "")
    fov_path = args.fov_path or os.environ.get("SAM3D_FOV_PATH", "")

    # Initialize sam-3d-body model and other optional modules
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, model_cfg = load_sam_3d_body(
        args.checkpoint_path, device=device, mhr_path=mhr_path
    )

    human_detector, human_segmentor, fov_estimator = None, None, None
    
    # Human detector
    if args.detector_name:
        from tools.build_detector import HumanDetector

        human_detector = HumanDetector(
            name=args.detector_name, device=device, path=detector_path
        )
    
    # Human segmentor - support depth-based for Jetson optimization
    if args.segmentor_name == "depth":
        # Use lightweight depth-based segmentation
        from tools.build_depth_segmentor import DepthBasedSegmentor
        
        human_segmentor = DepthBasedSegmentor(
            depth_threshold_min=args.depth_min,
            depth_threshold_max=args.depth_max,
            device=device,
        )
        print("Using depth-based segmentation (lightweight mode)")
    elif len(segmentor_path):
        from tools.build_sam import HumanSegmentor

        human_segmentor = HumanSegmentor(
            name=args.segmentor_name, device=device, path=segmentor_path
        )
    
    # FOV estimator - support direct intrinsics for Jetson optimization
    camera_intrinsics = parse_camera_intrinsics(args.camera_intrinsics)
    if camera_intrinsics is not None:
        # Use direct camera intrinsics (bypasses heavy MoGe2 model)
        from tools.build_fov_estimator import FOVEstimator
        
        fov_estimator = FOVEstimator(
            name="direct",
            device=device,
            intrinsics=camera_intrinsics,
        )
        print(f"Using direct camera intrinsics: fx={camera_intrinsics[0]:.1f}, "
              f"fy={camera_intrinsics[1]:.1f}, cx={camera_intrinsics[2]:.1f}, "
              f"cy={camera_intrinsics[3]:.1f}")
    elif args.fov_name:
        from tools.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(name=args.fov_name, device=device, path=fov_path)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )

    image_extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.gif",
        "*.bmp",
        "*.tiff",
        "*.webp",
    ]
    images_list = sorted(
        [
            image
            for ext in image_extensions
            for image in glob(os.path.join(args.image_folder, ext))
        ]
    )

    # Calibration mode: run first frame to cache shape parameters
    cached_shape = False
    if args.calibration_mode and len(images_list) > 0:
        print("\n=== CALIBRATION MODE ===")
        print("Running calibration on first image to cache shape parameters...")
        
        calibration_outputs = estimator.process_one_image(
            images_list[0],
            bbox_thr=args.bbox_thresh,
            use_mask=args.use_mask,
            inference_type="full",  # Full inference for calibration
        )
        
        if calibration_outputs:
            # Cache shape and scale from calibration
            shape_params = torch.tensor(
                [out["shape_params"] for out in calibration_outputs],
                device=device,
            )
            scale_params = torch.tensor(
                [out["scale_params"] for out in calibration_outputs],
                device=device,
            )
            
            # Set cached parameters in the model
            estimator.model.head_pose.set_cached_shape(shape_params, scale_params)
            estimator.model.head_pose.enable_shape_cache(True)
            
            print(f"Shape cached from {len(calibration_outputs)} person(s)")
            print("Subsequent frames will use cached shape for faster inference\n")
            cached_shape = True
    
    # Determine inference type
    inference_type = args.inference_type
    if inference_type == "auto":
        inference_type = "full"  # Default to full
    
    print(f"Inference type: {inference_type}")
    if cached_shape:
        print("Using cached shape parameters (runtime mode)")

    for image_path in tqdm(images_list):
        outputs = estimator.process_one_image(
            image_path,
            bbox_thr=args.bbox_thresh,
            use_mask=args.use_mask,
            inference_type=inference_type,
        )

        img = cv2.imread(image_path)
        rend_img = visualize_sample_together(img, outputs, estimator.faces)
        cv2.imwrite(
            f"{output_folder}/{os.path.basename(image_path)[:-4]}.jpg",
            rend_img.astype(np.uint8),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAM 3D Body Demo - Single Image Human Mesh Recovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard demo
  python demo.py --image_folder ./images --checkpoint_path ./checkpoints/model.ckpt

  # Jetson-optimized: body-only with direct camera intrinsics
  python demo.py --image_folder ./images --checkpoint_path ./model.ckpt \\
      --inference_type body --camera_intrinsics "1000,1000,640,360" --fov_name ""

  # Jetson-optimized: with calibration mode for shape caching
  python demo.py --image_folder ./images --checkpoint_path ./model.ckpt \\
      --calibration_mode --inference_type body

  # Depth-based segmentation (requires depth images)
  python demo.py --image_folder ./images --checkpoint_path ./model.ckpt \\
      --segmentor_name depth --depth_min 0.5 --depth_max 3.0

Environment Variables:
  SAM3D_MHR_PATH: Path to MHR asset
  SAM3D_DETECTOR_PATH: Path to human detection model folder
  SAM3D_SEGMENTOR_PATH: Path to human segmentation model folder
  SAM3D_FOV_PATH: Path to fov estimation model folder
        """,
    )
    parser.add_argument(
        "--image_folder",
        required=True,
        type=str,
        help="Path to folder containing input images",
    )
    parser.add_argument(
        "--output_folder",
        default="",
        type=str,
        help="Path to output folder (default: ./output/<image_folder_name>)",
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        type=str,
        help="Path to SAM 3D Body model checkpoint",
    )
    parser.add_argument(
        "--detector_name",
        default="vitdet",
        type=str,
        help="Human detection model for demo (Default `vitdet`, add your favorite detector if needed).",
    )
    parser.add_argument(
        "--segmentor_name",
        default="sam2",
        type=str,
        help="Human segmentation model: 'sam2' (default) or 'depth' for lightweight depth-based.",
    )
    parser.add_argument(
        "--fov_name",
        default="moge2",
        type=str,
        help="FOV estimation model: 'moge2' (default) or '' to disable (use --camera_intrinsics instead).",
    )
    parser.add_argument(
        "--detector_path",
        default="",
        type=str,
        help="Path to human detection model folder (or set SAM3D_DETECTOR_PATH)",
    )
    parser.add_argument(
        "--segmentor_path",
        default="",
        type=str,
        help="Path to human segmentation model folder (or set SAM3D_SEGMENTOR_PATH)",
    )
    parser.add_argument(
        "--fov_path",
        default="",
        type=str,
        help="Path to fov estimation model folder (or set SAM3D_FOV_PATH)",
    )
    parser.add_argument(
        "--mhr_path",
        default="",
        type=str,
        help="Path to MoHR/assets folder (or set SAM3D_mhr_path)",
    )
    parser.add_argument(
        "--bbox_thresh",
        default=0.8,
        type=float,
        help="Bounding box detection threshold",
    )
    parser.add_argument(
        "--use_mask",
        action="store_true",
        default=False,
        help="Use mask-conditioned prediction (segmentation mask is automatically generated from bbox)",
    )
    # Jetson optimization arguments
    parser.add_argument(
        "--inference_type",
        default="full",
        type=str,
        choices=["full", "body", "hand", "auto"],
        help="Inference type: 'full' (body + hand refinement), 'body' (faster, no hand), 'hand' (hand only)",
    )
    parser.add_argument(
        "--calibration_mode",
        action="store_true",
        default=False,
        help="Enable calibration mode: use first frame to cache shape, then use cached shape for runtime",
    )
    parser.add_argument(
        "--camera_intrinsics",
        default="",
        type=str,
        help="Direct camera intrinsics as 'fx,fy,cx,cy' (bypasses MoGe2 FOV estimation for speed)",
    )
    parser.add_argument(
        "--depth_min",
        default=0.3,
        type=float,
        help="Minimum depth threshold in meters for depth-based segmentation",
    )
    parser.add_argument(
        "--depth_max",
        default=4.0,
        type=float,
        help="Maximum depth threshold in meters for depth-based segmentation",
    )
    args = parser.parse_args()

    main(args)
