import pydantic
from typing import List

class City(pydantic.BaseModel):
    name: str
    x: float
    y: float
    connections: List[str] = []

    def __str__(self):
        return f"{self.name}: ({self.x}, {self.y})"

    def __repr__(self):
        return str(self)
