from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class BasketballPitchConfigurationV2:
    width: int = 1500
    length: int = 2800
    paint_width: int = 490
    three_point_distance: float = 675.0
    three_point_straight: int = 299
    three_point_margin: int = 90
    basket_distance: float = 157.5
    free_throw_distance: int = 579
    centre_circle_radius: int = 180  # Adding for compatibility with drawer
    penalty_spot_distance: int = 579 # Adding for compatibility with drawer

    @property
    def vertices(self) -> List[Optional[Tuple[float, float]]]:
        return [
            (0, 0),                                                          # 0: Top left corner
            (0, self.three_point_margin),                                    # 1: Top left 3-point line
            (0, (self.width - self.paint_width) / 2),                        # 2: Top left paint
            (0, (self.width + self.paint_width) / 2),                        # 3: Bottom left paint
            (0, self.width - self.three_point_margin),                       # 4: Bottom left 3-point line
            (0, self.width),                                                 # 5: Bottom left corner
            (self.free_throw_distance, (self.width - self.paint_width) / 2), # 6: Top left free throw corner
            (self.free_throw_distance, (self.width + self.paint_width) / 2), # 7: Bottom left free throw corner
            (self.length / 2, 0),                                            # 8: Top middle line
            (self.length / 2, self.width),                                   # 9: Bottom middle line
            (self.length, 0),                                                # 10: Top right corner
            (self.length, self.three_point_margin),                          # 11: Top right 3-point line
            (self.length, (self.width - self.paint_width) / 2),              # 12: Top right paint
            (self.length, (self.width + self.paint_width) / 2),              # 13: Bottom right paint
            (self.length, self.width - self.three_point_margin),             # 14: Bottom right 3-point line
            (self.length, self.width),                                       # 15: Bottom right corner
            (self.length - self.free_throw_distance, (self.width - self.paint_width) / 2), # 16: Top right free throw corner
            (self.length - self.free_throw_distance, (self.width + self.paint_width) / 2), # 17: Bottom right free throw corner
        ]

    # Note: edges are 1-indexed to be compatible with Drawer's line drawing logic
    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        (1, 9), (9, 11), # Top sideline
        (6, 10), (10, 16), # Bottom sideline
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), # Left baseline
        (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), # Right baseline
        (3, 7), (7, 8), (8, 4), # Left paint
        (13, 17), (17, 18), (18, 14), # Right paint
    ])

    labels: List[str] = field(default_factory=lambda: [
        str(i) for i in range(18)
    ])

    colors: List[str] = field(default_factory=lambda: [
        "#FF1493"] * 18
    )

    @property
    def active_vertices(self) -> List[Tuple[int, Tuple[float, float]]]:
        """Returns only non-deleted (index, coordinate) pairs. Index is 1-based."""
        return [
            (i + 1, v)
            for i, v in enumerate(self.vertices)
            if v is not None
        ]
