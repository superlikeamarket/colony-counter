from dataclasses import dataclass, asdict, field
from typing import Literal
from datetime import datetime
import uuid


@dataclass
class Detection:
    id: str
    center: tuple[int, int]
    box: tuple[int, int, int, int] | None = None  # x1, y1, x2, y2
    confidence: float | None = None
    source: Literal["model", "user"] = "model"


@dataclass
class CorrectionState:
    image_id: str
    sample_type: Literal["petri_dish", "petrifilm", "unknown"] = "unknown"

    model_detections: list[Detection] = field(default_factory=list)
    added_points: list[Detection] = field(default_factory=list)
    removed_detection_ids: set[str] = field(default_factory=set)

    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def active_model_detections(self) -> list[Detection]:
        return [
            detection
            for detection in self.model_detections
            if detection.id not in self.removed_detection_ids
        ]

    def all_active_points(self) -> list[Detection]:
        return self.active_model_detections() + self.added_points

    def final_count(self) -> int:
        return len(self.all_active_points())

    def model_count(self) -> int:
        return len(self.model_detections)

    def add_user_point(self, x: int, y: int) -> Detection:
        point = Detection(
            id=f"user_{uuid.uuid4().hex[:8]}",
            center=(x, y),
            source="user",
        )
        self.added_points.append(point)
        self.updated_at = datetime.now().isoformat()
        return point

    def remove_detection(self, detection_id: str) -> None:
        self.removed_detection_ids.add(detection_id)
        self.updated_at = datetime.now().isoformat()

    def undo_remove_detection(self, detection_id: str) -> None:
        self.removed_detection_ids.discard(detection_id)
        self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> dict:
        data = asdict(self)
        data["removed_detection_ids"] = list(self.removed_detection_ids)
        data["model_count"] = self.model_count()
        data["final_count"] = self.final_count()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "CorrectionState":
        state = cls(
            image_id=data["image_id"],
            sample_type=data.get("sample_type", "unknown"),
            model_detections=[
                Detection(**det)
                for det in data.get("model_detections", [])
            ],
            added_points=[
                Detection(**point)
                for point in data.get("added_points", [])
            ],
            removed_detection_ids=set(data.get("removed_detection_ids", [])),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
        )
        return state