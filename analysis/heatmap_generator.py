import numpy as np
import cv2
from scipy.ndimage import gaussian_filter


class HeatmapGenerator:
    """
    Generates per-team player heatmaps by accumulating tactical (pitch-projected)
    positions across all video frames and rendering a Gaussian-smoothed density
    overlay on top of the pitch/court image.

    Inspired by the football_ai.ipynb workflow:
        1. Track players per frame and project to pitch via ViewTransformer
        2. Accumulate positions into a 2D histogram
        3. Smooth with a Gaussian kernel
        4. Overlay on the pitch/court background image
    """

    def __init__(self, config, court_image_path: str):
        """
        Args:
            config: Pitch/court configuration object (SoccerPitchConfiguration or
                    BasketballPitchConfigurationV2). Must expose `length` and `width`
                    attributes in centimetres.
            court_image_path: Absolute path to the pitch/court background PNG image.
        """
        self.config = config
        self.court_image_path = court_image_path

        # Canonical dimensions in cm (X = length, Y = width)
        self.pitch_length = getattr(config, 'length', 2800)
        self.pitch_width = getattr(config, 'width', 1500)

        # Resolution of the internal histogram grid.
        # ~10 cm per bin gives good detail without excessive memory.
        self.bin_size_cm = 10
        self.grid_cols = max(1, int(self.pitch_length / self.bin_size_cm))
        self.grid_rows = max(1, int(self.pitch_width / self.bin_size_cm))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        tactical_positions: list,
        tracks: dict,
        team: int,
    ) -> bytes:
        """
        Generate a PNG-encoded heatmap image for a specific team.

        Args:
            tactical_positions: Per-frame list of dicts  {player_id: [x_cm, y_cm]}
                                as produced by TacticalViewConverter.
            tracks: Full player tracks dict (to look up each player's team).
            team: Team index (0 or 1) to generate the heatmap for.

        Returns:
            PNG-encoded image bytes.
        """
        # 1. Collect all (x, y) positions for the requested team
        points = self._collect_team_positions(tactical_positions, tracks, team)

        if len(points) == 0:
            # No data — return a blank pitch image
            court_img = cv2.imread(self.court_image_path)
            if court_img is None:
                court_img = np.zeros((self.grid_rows, self.grid_cols, 3), dtype=np.uint8)
            _, png_bytes = cv2.imencode('.png', court_img)
            return png_bytes.tobytes()

        points = np.array(points, dtype=np.float64)

        # 2. Build a 2D histogram on the grid
        histogram = self._build_histogram(points)

        # 3. Gaussian smoothing
        sigma = self._auto_sigma()
        smoothed = gaussian_filter(histogram, sigma=sigma)

        # 4. Render overlay on the court image
        heatmap_img = self._render_overlay(smoothed, team)

        # 5. Encode to PNG
        _, png_bytes = cv2.imencode('.png', heatmap_img)
        return png_bytes.tobytes()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_team_positions(self, tactical_positions, tracks, team):
        """Gather all (x, y) positions belonging to the specified team."""
        points = []
        player_tracks = tracks.get('players', [])

        for frame_idx, frame_positions in enumerate(tactical_positions):
            if frame_idx >= len(player_tracks):
                break
            for player_id, pos in frame_positions.items():
                # Look up which team this player belongs to
                player_info = player_tracks[frame_idx].get(player_id, {})
                player_team = player_info.get('team', -1)

                if player_team == team:
                    x_cm, y_cm = pos
                    # Only include in-bounds positions
                    if 0 <= x_cm <= self.pitch_length and 0 <= y_cm <= self.pitch_width:
                        points.append((x_cm, y_cm))

        return points

    def _build_histogram(self, points: np.ndarray) -> np.ndarray:
        """
        Map (x_cm, y_cm) positions into a 2D histogram grid.
        X maps to columns (pitch length), Y maps to rows (pitch width).
        """
        histogram = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float64)

        col_indices = np.clip(
            (points[:, 0] / self.bin_size_cm).astype(int),
            0, self.grid_cols - 1
        )
        row_indices = np.clip(
            (points[:, 1] / self.bin_size_cm).astype(int),
            0, self.grid_rows - 1
        )

        # Accumulate counts
        np.add.at(histogram, (row_indices, col_indices), 1)
        return histogram

    def _auto_sigma(self) -> float:
        """
        Choose a Gaussian sigma proportional to ~3% of the longer pitch dimension.
        This gives a nice spread without being too blurry or too spiky.
        """
        longer_dim_bins = max(self.grid_cols, self.grid_rows)
        return max(2.0, longer_dim_bins * 0.03)

    def _render_overlay(self, smoothed: np.ndarray, team: int) -> np.ndarray:
        """
        Render the smoothed density map as a colour overlay on the court image.
        Uses a team-specific colour gradient.
        """
        # Load the court background
        court_img = cv2.imread(self.court_image_path)
        if court_img is None:
            # Fallback: create a dark green / dark brown background
            court_img = np.full(
                (self.grid_rows, self.grid_cols, 3),
                (34, 139, 34),  # Forest green (BGR)
                dtype=np.uint8
            )

        target_h, target_w = court_img.shape[:2]

        # Normalise the density to [0, 255]
        max_val = smoothed.max()
        if max_val > 0:
            norm = (smoothed / max_val * 255).astype(np.uint8)
        else:
            norm = np.zeros_like(smoothed, dtype=np.uint8)

        # Resize the heatmap grid to match the court image size
        norm_resized = cv2.resize(norm, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        # Apply team-specific colormap
        if team == 0:
            # Team 0: blue gradient (matches #00BFFF)
            colored = cv2.applyColorMap(norm_resized, cv2.COLORMAP_OCEAN)
        else:
            # Team 1: pink/hot gradient (matches #FF1493)
            colored = cv2.applyColorMap(norm_resized, cv2.COLORMAP_HOT)

        # Create an alpha mask — only overlay where density > 0
        # Use a threshold to avoid colouring the entire pitch
        alpha = norm_resized.astype(np.float32) / 255.0
        # Boost contrast: apply a power curve so low-density areas are more transparent
        alpha = np.power(alpha, 0.6)
        # Cap max opacity
        alpha = np.clip(alpha, 0.0, 0.7)
        alpha_3ch = np.stack([alpha, alpha, alpha], axis=-1)

        # Alpha-blend: result = alpha * heatmap + (1 - alpha) * court
        blended = (alpha_3ch * colored.astype(np.float32)
                   + (1.0 - alpha_3ch) * court_img.astype(np.float32))
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        # Add a title label
        team_label = f"Team {team + 1} Heatmap"
        cv2.putText(
            blended, team_label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            (255, 255, 255), 3  # white outline
        )
        cv2.putText(
            blended, team_label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            (0, 0, 0), 2  # black text
        )

        return blended
