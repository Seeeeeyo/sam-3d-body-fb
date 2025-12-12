# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Depth-based human segmentation for Jetson optimization.

This module provides a lightweight alternative to SAM-based segmentation
by using depth data from cameras like Zed 2i to create human masks.
This is significantly faster than neural network-based segmentation.

Usage:
    from tools.build_depth_segmentor import DepthBasedSegmentor
    
    segmentor = DepthBasedSegmentor(
        depth_threshold_min=0.5,  # meters
        depth_threshold_max=3.0,  # meters
    )
    masks, scores = segmentor.run_segmentation(depth_image, boxes)
"""

import numpy as np
from typing import Optional, Tuple


class DepthBasedSegmentor:
    """
    Lightweight depth-based human segmentation for real-time applications.
    
    Replaces heavy SAM-based segmentation with simple depth thresholding,
    providing massive speedup for Jetson deployment scenarios where depth
    cameras (e.g., Zed 2i) are available.
    
    Args:
        depth_threshold_min: Minimum depth in meters to consider as foreground
        depth_threshold_max: Maximum depth in meters to consider as foreground
        erode_kernel_size: Size of erosion kernel to clean up mask edges
        dilate_kernel_size: Size of dilation kernel to fill mask holes
    """
    
    def __init__(
        self,
        depth_threshold_min: float = 0.3,
        depth_threshold_max: float = 4.0,
        erode_kernel_size: int = 3,
        dilate_kernel_size: int = 5,
        device: str = "cuda",
    ):
        self.depth_threshold_min = depth_threshold_min
        self.depth_threshold_max = depth_threshold_max
        self.erode_kernel_size = erode_kernel_size
        self.dilate_kernel_size = dilate_kernel_size
        self.device = device
        
        # Pre-create morphological kernels
        self._erode_kernel = np.ones(
            (erode_kernel_size, erode_kernel_size), np.uint8
        )
        self._dilate_kernel = np.ones(
            (dilate_kernel_size, dilate_kernel_size), np.uint8
        )
    
    def run_segmentation(
        self,
        depth_image: np.ndarray,
        boxes: np.ndarray,
        rgb_image: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment humans using depth thresholding within detected bounding boxes.
        
        Args:
            depth_image: Depth image (H, W) in meters
            boxes: Bounding boxes (N, 4) in [x1, y1, x2, y2] format
            rgb_image: Optional RGB image (unused, for API compatibility)
            
        Returns:
            masks: Binary masks (N, H, W)
            scores: Confidence scores (N,) - always 1.0 for depth-based
        """
        if len(boxes) == 0:
            return np.array([]), np.array([])
        
        height, width = depth_image.shape[:2]
        num_boxes = boxes.shape[0]
        
        # Create depth-based foreground mask
        foreground_mask = (
            (depth_image >= self.depth_threshold_min) & 
            (depth_image <= self.depth_threshold_max) &
            (depth_image > 0)  # Valid depth readings
        )
        
        all_masks = []
        all_scores = []
        
        for i in range(num_boxes):
            box = boxes[i].astype(int)
            x1, y1, x2, y2 = (
                max(0, box[0]),
                max(0, box[1]),
                min(width, box[2]),
                min(height, box[3]),
            )
            
            # Create mask for this bounding box
            box_mask = np.zeros((height, width), dtype=np.uint8)
            box_mask[y1:y2, x1:x2] = 1
            
            # Combine with depth foreground
            person_mask = (foreground_mask & box_mask.astype(bool)).astype(np.uint8)
            
            # Apply morphological operations to clean up the mask
            person_mask = self._refine_mask(person_mask)
            
            all_masks.append(person_mask)
            all_scores.append(1.0)  # High confidence for depth-based
        
        return np.stack(all_masks), np.array(all_scores)
    
    def run_sam(
        self,
        img: np.ndarray,
        boxes: np.ndarray,
        depth_image: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        API-compatible method matching HumanSegmentor interface.
        
        If depth_image is provided, uses depth-based segmentation.
        Otherwise, falls back to simple bounding box masks.
        
        Args:
            img: RGB image (H, W, 3)
            boxes: Bounding boxes (N, 4)
            depth_image: Optional depth image (H, W) in meters
            
        Returns:
            masks: Binary masks (N, H, W)
            scores: Confidence scores (N,)
        """
        if depth_image is not None:
            return self.run_segmentation(depth_image, boxes, rgb_image=img)
        else:
            # Fallback: use bounding box as mask
            return self._bbox_to_mask(img, boxes)
    
    def _refine_mask(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological operations to refine the mask."""
        try:
            import cv2
            # Erode to remove noise
            mask = cv2.erode(mask, self._erode_kernel, iterations=1)
            # Dilate to fill holes
            mask = cv2.dilate(mask, self._dilate_kernel, iterations=2)
            # Final erosion to restore approximate size
            mask = cv2.erode(mask, self._erode_kernel, iterations=1)
        except ImportError:
            # If cv2 not available, return as-is
            pass
        return mask
    
    def _bbox_to_mask(
        self, img: np.ndarray, boxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create simple bounding box masks as fallback."""
        height, width = img.shape[:2]
        num_boxes = boxes.shape[0]
        
        if num_boxes == 0:
            return np.array([]), np.array([])
        
        masks = np.zeros((num_boxes, height, width), dtype=np.uint8)
        scores = np.ones(num_boxes, dtype=np.float32)
        
        for i in range(num_boxes):
            box = boxes[i].astype(int)
            x1, y1, x2, y2 = (
                max(0, box[0]),
                max(0, box[1]),
                min(width, box[2]),
                min(height, box[3]),
            )
            masks[i, y1:y2, x1:x2] = 1
        
        return masks, scores


class AdaptiveDepthSegmentor(DepthBasedSegmentor):
    """
    Adaptive depth segmentation that estimates depth thresholds from the scene.
    
    Useful when the human distance from camera varies across the sequence.
    """
    
    def __init__(
        self,
        percentile_min: float = 10,
        percentile_max: float = 90,
        fallback_threshold_min: float = 0.5,
        fallback_threshold_max: float = 3.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.percentile_min = percentile_min
        self.percentile_max = percentile_max
        self.fallback_threshold_min = fallback_threshold_min
        self.fallback_threshold_max = fallback_threshold_max
    
    def run_segmentation(
        self,
        depth_image: np.ndarray,
        boxes: np.ndarray,
        rgb_image: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment with adaptive depth thresholds based on box regions.
        """
        if len(boxes) == 0:
            return np.array([]), np.array([])
        
        height, width = depth_image.shape[:2]
        all_masks = []
        all_scores = []
        
        for i in range(boxes.shape[0]):
            box = boxes[i].astype(int)
            x1, y1, x2, y2 = (
                max(0, box[0]),
                max(0, box[1]),
                min(width, box[2]),
                min(height, box[3]),
            )
            
            # Get depth values within box
            box_depths = depth_image[y1:y2, x1:x2]
            valid_depths = box_depths[(box_depths > 0) & np.isfinite(box_depths)]
            
            if len(valid_depths) > 0:
                # Adaptive thresholds based on depth distribution in box
                d_min = np.percentile(valid_depths, self.percentile_min)
                d_max = np.percentile(valid_depths, self.percentile_max)
                # Add margin
                margin = (d_max - d_min) * 0.2
                d_min = max(0.1, d_min - margin)
                d_max = d_max + margin
            else:
                d_min = self.fallback_threshold_min
                d_max = self.fallback_threshold_max
            
            # Create mask with adaptive thresholds
            foreground_mask = (
                (depth_image >= d_min) & 
                (depth_image <= d_max) &
                (depth_image > 0)
            )
            
            # Intersect with bounding box
            box_mask = np.zeros((height, width), dtype=bool)
            box_mask[y1:y2, x1:x2] = True
            person_mask = (foreground_mask & box_mask).astype(np.uint8)
            
            # Refine mask
            person_mask = self._refine_mask(person_mask)
            
            all_masks.append(person_mask)
            all_scores.append(1.0)
        
        return np.stack(all_masks), np.array(all_scores)
