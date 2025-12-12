# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
FOV (Field of View) estimation module with support for:
1. MoGe2 neural network-based estimation (default, slower)
2. Direct camera intrinsics from calibrated cameras like Zed 2i (fast)

For Jetson deployment, use DirectIntrinsics to bypass the heavy MoGe2 model.
"""

import torch
import numpy as np
from typing import Optional, Union, Tuple


class FOVEstimator:
    """
    FOV estimator supporting multiple backends.
    
    Args:
        name: Estimator type - "moge2" for neural estimation or "direct" for
              using pre-calibrated camera intrinsics
        device: Torch device
        intrinsics: For "direct" mode, provide camera intrinsics as (fx, fy, cx, cy)
                   or a 3x3 numpy/torch array
        **kwargs: Additional arguments passed to the estimator
    """
    
    def __init__(
        self,
        name: str = "moge2",
        device: str = "cuda",
        intrinsics: Optional[Union[Tuple[float, ...], np.ndarray, torch.Tensor]] = None,
        **kwargs,
    ):
        self.device = device
        self.name = name

        if name == "moge2":
            print("########### Using fov estimator: MoGe2...")
            self.fov_estimator = load_moge(device, **kwargs)
            self.fov_estimator_func = run_moge
            self.fov_estimator.eval()
        elif name == "direct":
            print("########### Using direct camera intrinsics (no FOV estimation)...")
            if intrinsics is None:
                raise ValueError(
                    "For 'direct' mode, must provide camera intrinsics as "
                    "(fx, fy, cx, cy) tuple or 3x3 matrix"
                )
            self.fov_estimator = DirectIntrinsics(intrinsics, device)
            self.fov_estimator_func = run_direct
        else:
            raise NotImplementedError(f"FOV estimator '{name}' not implemented")

    def get_cam_intrinsics(self, img, **kwargs):
        return self.fov_estimator_func(self.fov_estimator, img, self.device, **kwargs)


class DirectIntrinsics:
    """
    Direct camera intrinsics provider for calibrated cameras.
    
    This is a lightweight alternative to neural FOV estimation that uses
    pre-calibrated camera parameters. Ideal for deployment with cameras
    like Zed 2i that provide accurate intrinsics.
    
    Args:
        intrinsics: Camera intrinsics as:
            - Tuple (fx, fy, cx, cy)
            - 3x3 numpy array or torch tensor (K matrix)
            - Zed SDK camera parameters object (auto-extracted)
    """
    
    def __init__(
        self,
        intrinsics: Union[Tuple[float, ...], np.ndarray, torch.Tensor],
        device: str = "cuda",
    ):
        self.device = device
        self._cam_int = self._parse_intrinsics(intrinsics)
    
    def _parse_intrinsics(
        self, intrinsics: Union[Tuple[float, ...], np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """Parse various intrinsics formats into 3x3 K matrix."""
        if isinstance(intrinsics, (list, tuple)):
            if len(intrinsics) == 4:
                fx, fy, cx, cy = intrinsics
                K = torch.tensor([
                    [fx, 0.0, cx],
                    [0.0, fy, cy],
                    [0.0, 0.0, 1.0],
                ], dtype=torch.float32)
            else:
                raise ValueError(
                    f"Expected 4 values (fx, fy, cx, cy), got {len(intrinsics)}"
                )
        elif isinstance(intrinsics, np.ndarray):
            K = torch.from_numpy(intrinsics.astype(np.float32))
        elif isinstance(intrinsics, torch.Tensor):
            K = intrinsics.float()
        else:
            raise TypeError(
                f"Unsupported intrinsics type: {type(intrinsics)}. "
                "Expected tuple, numpy array, or torch tensor."
            )
        
        if K.shape != (3, 3):
            raise ValueError(f"Intrinsics must be 3x3, got shape {K.shape}")
        
        return K
    
    def get_intrinsics(self, height: int, width: int) -> torch.Tensor:
        """
        Get camera intrinsics matrix.
        
        Args:
            height: Image height (for validation, not used for scaling)
            width: Image width (for validation, not used for scaling)
            
        Returns:
            3x3 intrinsics matrix
        """
        return self._cam_int.clone()
    
    @classmethod
    def from_zed(cls, zed_camera, device: str = "cuda") -> "DirectIntrinsics":
        """
        Create DirectIntrinsics from Zed SDK camera object.
        
        Args:
            zed_camera: pyzed.sl.Camera instance
            device: Torch device
            
        Returns:
            DirectIntrinsics instance
            
        Example:
            import pyzed.sl as sl
            zed = sl.Camera()
            # ... initialize camera ...
            intrinsics = DirectIntrinsics.from_zed(zed)
            
        Note:
            pyzed.sl is imported inside this method to avoid requiring it as a 
            hard dependency. Users who don't use Zed cameras don't need to 
            install the SDK.
        """
        # Import inside method to keep pyzed as optional dependency
        try:
            import pyzed.sl as sl  # noqa: F401 - verify import works
            
            calibration = zed_camera.get_camera_information().camera_configuration.calibration_parameters
            left_cam = calibration.left_cam
            
            fx = left_cam.fx
            fy = left_cam.fy
            cx = left_cam.cx
            cy = left_cam.cy
            
            return cls((fx, fy, cx, cy), device)
        except ImportError:
            raise ImportError(
                "pyzed.sl not found. Install Zed SDK Python bindings."
            )
    
    @classmethod
    def from_fov(
        cls,
        fov_degrees: float,
        width: int,
        height: int,
        device: str = "cuda",
    ) -> "DirectIntrinsics":
        """
        Create DirectIntrinsics from field of view angle.
        
        Args:
            fov_degrees: Horizontal field of view in degrees
            width: Image width in pixels
            height: Image height in pixels
            device: Torch device
            
        Returns:
            DirectIntrinsics instance
        """
        fov_rad = np.deg2rad(fov_degrees)
        fx = width / (2 * np.tan(fov_rad / 2))
        fy = fx  # Assume square pixels
        cx = width / 2
        cy = height / 2
        
        return cls((fx, fy, cx, cy), device)


def run_direct(intrinsics_provider: DirectIntrinsics, input_image, device):
    """
    Get intrinsics from DirectIntrinsics provider.
    
    Args:
        intrinsics_provider: DirectIntrinsics instance
        input_image: Input image (only used to get dimensions)
        device: Torch device
        
    Returns:
        Camera intrinsics with batch dimension (1, 3, 3)
    """
    H, W = input_image.shape[:2]
    cam_int = intrinsics_provider.get_intrinsics(H, W)
    # Add batch dimension
    return cam_int[None].to(device)


def load_moge(device, path=""):
    from moge.model.v2 import MoGeModel

    if path == "":
        path = "Ruicheng/moge-2-vitl-normal"
    moge_model = MoGeModel.from_pretrained(path).to(device)
    return moge_model


def run_moge(model, input_image, device):
    # We expect the image to be RGB already
    H, W, _ = input_image.shape
    input_image = torch.tensor(
        input_image / 255, dtype=torch.float32, device=device
    ).permute(2, 0, 1)

    # Infer w/ MoGe2
    moge_data = model.infer(input_image)

    # get intrinsics
    intrinsics = denormalize_f(moge_data["intrinsics"].cpu().numpy(), H, W)
    v_focal = intrinsics[1, 1]

    # override hfov with v_focal
    intrinsics[0, 0] = v_focal
    # add batch dim
    cam_intrinsics = intrinsics[None]

    return cam_intrinsics


def denormalize_f(norm_K, height, width):
    # Extract cx and cy from the normalized K matrix
    cx_norm = norm_K[0][2]  # c_x is at K[0][2]
    cy_norm = norm_K[1][2]  # c_y is at K[1][2]

    fx_norm = norm_K[0][0]  # Normalized fx
    fy_norm = norm_K[1][1]  # Normalized fy
    # s_norm = norm_K[0][1]   # Skew (usually 0)

    # Scale to absolute values
    fx_abs = fx_norm * width
    fy_abs = fy_norm * height
    cx_abs = cx_norm * width
    cy_abs = cy_norm * height
    # s_abs = s_norm * width
    s_abs = 0

    # Construct absolute K matrix
    abs_K = torch.tensor(
        [[fx_abs, s_abs, cx_abs], [0.0, fy_abs, cy_abs], [0.0, 0.0, 1.0]]
    )
    return abs_K
