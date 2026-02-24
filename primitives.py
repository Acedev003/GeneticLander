from dataclasses import dataclass

@dataclass
class Point:
    x: int
    y: int

@dataclass
class Segment:
    start_point: Point
    end_point: Point