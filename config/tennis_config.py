from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class TennisCourtConfiguration:
    # Standard doubles court dimensions (cm)
    length: int = 2377          # baseline to baseline
    width: int = 1097           # doubles sideline to sideline
    singles_margin: int = 137   # doubles to singles sideline (each side)
    service_line: int = 640     # baseline to service line

    @property
    def net_x(self) -> float:
        return self.length / 2  # 1188.5cm

    @property
    def center_y(self) -> float:
        return self.width / 2   # 548.5cm

    @property
    def singles_far_y(self) -> int:
        return self.width - self.singles_margin  # 960cm

    @property
    def vertices(self) -> List[Tuple[float, float]]:
        return [
            (self.length, 0),                                                   # 0: bottom-left corner
            (self.length, self.singles_margin),                                 # 1: bottom-left singles
            (self.length, self.singles_far_y),                                  # 2: bottom-right singles
            (self.length, self.width),                                          # 3: bottom-right corner
            (0, self.width),                                                    # 4: top-right corner
            (0, self.singles_far_y),                                            # 5: top-right singles
            (0, self.singles_margin),                                           # 6: top-left singles
            (0, 0),                                                             # 7: top-left corner
            (self.service_line, self.singles_margin),                           # 8: upper-left service
            (self.service_line, self.center_y),                                 # 9: upper service T
            (self.service_line, self.singles_far_y),                            # 10: upper-right service
            (self.length - self.service_line, self.singles_far_y),              # 11: lower-right service
            (self.length - self.service_line, self.center_y),                   # 12: lower service T
            (self.length - self.service_line, self.singles_margin),             # 13: lower-left service
            (self.net_x, self.center_y),                                        # 14: net center
        ]

    # 1-indexed to match pipeline convention (config.vertices[start - 1])
    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        # Outer baselines
        (8, 7), (7, 6), (6, 5),       # top baseline
        (1, 2), (2, 3), (3, 4),       # bottom baseline
        # Doubles sidelines
        (8, 1),                        # left sideline
        (5, 4),                        # right sideline
        # Singles sidelines
        (7, 9), (9, 14), (14, 2),     # left singles sideline
        (6, 11), (11, 12), (12, 3),   # right singles sideline
        # Service lines
        (9, 10), (10, 11),             # upper service line
        (14, 13), (13, 12),            # lower service line
        # Center service line (14 only connects to 9 and 12)
        (10, 15), (15, 13),
    ])

    labels: List[str] = field(default_factory=lambda: [
        str(i) for i in range(15)
    ])

    colors: List[str] = field(default_factory=lambda: [
        "#FF1493"] * 15
    )

    @property
    def active_vertices(self) -> List[Tuple[int, Tuple[float, float]]]:
        """All vertices active — no deleted points in tennis config."""
        return [(i + 1, v) for i, v in enumerate(self.vertices)]