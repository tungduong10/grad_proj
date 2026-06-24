from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class BasketballPitchConfiguration:
    width: int = 1500
    length: int = 2800
    paint_width: int = 490
    three_point_distance: float = 675.0
    three_point_straight: int = 299
    three_point_margin: int = 90
    basket_distance: float = 157.5
    free_throw_distance: int = 579

    @property
    def vertices(self) -> List[Optional[Tuple[float, float]]]:
        return [
            (0, 0),                                                          # 01
            (0, self.three_point_margin),                                    # 02
            None,                                                            # 03 (Deleted)
            (0, (self.width - self.paint_width) / 2),                       # 04
            (0, (self.width + self.paint_width) / 2),                       # 05
            None,                                                            # 06 (Deleted)
            (0, self.width - self.three_point_margin),                      # 07
            (0, self.width),                                                 # 08
            (self.basket_distance, self.width / 2),                         # 09
            (self.three_point_straight, self.three_point_margin),           # 10
            (self.three_point_straight, self.width - self.three_point_margin), # 11
            (self.free_throw_distance, (self.width - self.paint_width) / 2),# 12
            (self.free_throw_distance, self.width / 2),                     # 13
            (self.free_throw_distance, (self.width + self.paint_width) / 2),# 14
            (self.free_throw_distance, 0),                                  # 15
            (self.basket_distance + self.three_point_distance, self.width / 2), # 16
            (self.free_throw_distance, self.width),                         # 17
            None,                                                            # 18 (Deleted)
            (self.length / 2, 0),                                           # 19
            None,                                                            # 20 (Deleted)
            (self.length / 2, self.width / 2),                              # 21
            None,                                                            # 22 (Deleted)
            (self.length / 2, self.width),                                  # 23
            None,                                                            # 24 (Deleted)
            (self.length - self.free_throw_distance, 0),                    # 25
            (self.length - (self.basket_distance + self.three_point_distance), self.width / 2), # 26
            (self.length - self.free_throw_distance, self.width),           # 27
            (self.length - self.free_throw_distance, (self.width - self.paint_width) / 2), # 28
            (self.length - self.free_throw_distance, self.width / 2),       # 29
            (self.length - self.free_throw_distance, (self.width + self.paint_width) / 2), # 30
            (self.length - self.three_point_straight, self.three_point_margin), # 31
            (self.length - self.three_point_straight, self.width - self.three_point_margin), # 32
            (self.length - self.basket_distance, self.width / 2),           # 33
            (self.length, 0),                                                # 34
            (self.length, self.three_point_margin),                         # 35
            None,                                                            # 36 (Deleted)
            (self.length, (self.width - self.paint_width) / 2),             # 37
            (self.length, (self.width + self.paint_width) / 2),             # 38
            None,                                                            # 39 (Deleted)
            (self.length, self.width - self.three_point_margin),            # 40
            (self.length, self.width),                                      # 41
        ]

    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        # Court outer bounds
        (1, 8), (34, 41), (1, 34), (8, 41),
        # Center line
        (19, 23),
        # Left Paint
        (4, 12), (12, 14), (14, 5),
        # Right Paint
        (37, 28), (28, 30), (30, 38),
        # Left 3pt line
        (2, 10), (10, 16), (16, 11), (11, 7),
        # Right 3pt line
        (35, 31), (31, 26), (26, 32), (32, 40)
    ])

    labels: List[str] = field(default_factory=lambda: [
        f"{i:02d}" for i in range(1, 42)
    ])

    colors: List[str] = field(default_factory=lambda: [
        "#FF1493"] * 18 +
        ["#00BFFF"] * 6 +
        ["#32CD32"] * 17
    )

    @property
    def active_vertices(self) -> List[Tuple[int, Tuple[float, float]]]:
        """Returns only non-deleted (index, coordinate) pairs. Index is 1-based."""
        return [
            (i + 1, v)
            for i, v in enumerate(self.vertices)
            if v is not None
        ]