from typing import Tuple, Optional
import cv2
import numpy as np
import numpy.typing as npt
from collections import deque


class ViewTransformer:
    def __init__(
            self,
            filter_window: int = 5,
            max_reproj_error: float = 300.0,
            min_spread_x: float = 2500.0,
            min_spread_y: float = 1500.0
    ) -> None:
        """
        Initialize the ViewTransformer.

        Args:
            filter_window:    Controls EWA smoothing. Higher = smoother. Same API as before.
            max_reproj_error: Reject homographies where mean reprojection error in config
                              space exceeds this value (cm). Prevents bad frames polluting
                              the smoothed result.
            min_spread_x:     Minimum required keypoint spread in the config-space X axis
                              (pitch length direction, cm) before a homography update is
                              accepted. Default 2500cm = 25m.
            min_spread_y:     Minimum required keypoint spread in config-space Y axis
                              (pitch width direction, cm). Default 1500cm = 15m.

        Why spread thresholds matter for full-pitch projection:
            A homography estimated from keypoints clustered in one region (e.g. only the
            left goal box, x=0–2015cm) extrapolates unreliably to the rest of the pitch.
            When keypoints are sparse/clustered, it is better to HOLD the last good
            homography (computed from a well-spread set) than to overwrite it with a
            poorly constrained new estimate. This is the root cause of the deformation
            seen when only indices [0,1,6,8,9,10,11] are detected.
        """
        self.alpha = max(0.0, min(0.9, 1.0 - 1.0 / filter_window))
        self.max_reproj_error = max_reproj_error
        self.min_spread_x = min_spread_x
        self.min_spread_y = min_spread_y
        self.m = None           # Current smoothed homography (camera → config space)
        self.last_spread = (0.0, 0.0)   # Diagnostic: spread of last accepted update

    def update(
            self,
            source: npt.NDArray[np.float32],
            target: npt.NDArray[np.float32]
    ) -> None:
        """
        Update the homography from source (camera) to target (config) points.
        Updates are silently skipped when keypoints are too clustered or the
        computed homography has high reprojection error — preserving the last
        reliable estimate instead.

        Args:
            source: Camera-space keypoint coordinates, shape (N, 2).
            target: Config-space coordinates, shape (N, 2).
        """
        if source.size == 0 or target.size == 0:
            return

        if source.shape != target.shape:
            raise ValueError("Source and target must have the same shape.")
        if source.shape[1] != 2:
            raise ValueError("Source and target points must be 2D coordinates.")

        source = source.astype(np.float32)
        target = target.astype(np.float32)

        # --- Spread gate -------------------------------------------------------
        # Measure how well the detected keypoints span the pitch in both axes.
        # A homography needs geometric diversity in the control points to be
        # reliable across the full pitch. If the spread is too low, the current
        # estimate is not trustworthy for full-pitch projection — keep the last
        # good one unchanged.
        spread_x = float(np.max(target[:, 0]) - np.min(target[:, 0]))
        spread_y = float(np.max(target[:, 1]) - np.min(target[:, 1]))
        self.last_spread = (spread_x, spread_y)

        if spread_x < self.min_spread_x and spread_y < self.min_spread_y:
            # Keypoints too clustered — hold current homography
            return
        # -----------------------------------------------------------------------

        m, mask = cv2.findHomography(source, target, cv2.LMEDS)

        if m is None:
            return

        # --- Reprojection error gate -------------------------------------------
        # Even with good spread, the homography might be a poor fit (e.g. a few
        # keypoints were detected at wrong positions). Check that the computed
        # matrix actually maps source close to target before accepting it.
        projected = cv2.perspectiveTransform(
            source.reshape(-1, 1, 2), m
        ).reshape(-1, 2)
        mean_error = float(np.mean(np.linalg.norm(projected - target, axis=1)))

        if mean_error > self.max_reproj_error:
            return
        # -----------------------------------------------------------------------

        # EWA smoothing — geometrically valid unlike element-wise mean
        if self.m is None:
            self.m = m
        else:
            self.m = self.alpha * self.m + (1.0 - self.alpha) * m

    def transform_points(
            self,
            points: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        Transform camera-space points to config space using the smoothed homography.
        """
        if points.size == 0 or self.m is None:
            return points

        if points.shape[1] != 2:
            raise ValueError("Points must be 2D coordinates.")

        reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(reshaped, self.m)
        return transformed.reshape(-1, 2).astype(np.float32)

    def transform_image(
            self,
            image: npt.NDArray[np.uint8],
            resolution_wh: Tuple[int, int]
    ) -> npt.NDArray[np.uint8]:
        """
        Warp the full image from camera space into config (top-down) space.
        """
        if self.m is None:
            return image

        if len(image.shape) not in {2, 3}:
            raise ValueError("Image must be either grayscale or colour.")

        return cv2.warpPerspective(image, self.m, resolution_wh)