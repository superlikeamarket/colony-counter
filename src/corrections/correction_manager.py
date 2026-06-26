import json
from pathlib import Path
from math import dist

from src.corrections.correction_state import CorrectionState, Detection


class CorrectionManager:
    def __init__(self, corrections_dir: str | Path):
        self.corrections_dir = Path(corrections_dir)
        self.corrections_dir.mkdir(parents=True, exist_ok=True)

    def create_state_from_predictions(
        self,
        image_id: str,
        centers: list[tuple[int, int]],
        boxes: list[tuple[int, int, int, int]] | None = None,
        confidences: list[float] | None = None,
        sample_type: str = "unknown",
    ) -> CorrectionState:
        detections = []

        if boxes is not None and len(boxes) != len(centers):
            raise ValueError("boxes must have the same length as centers")

        if confidences is not None and len(confidences) != len(centers):
            raise ValueError("confidences must have the same length as centers")

        for i, center in enumerate(centers):
            detection = Detection(
                id=f"model_{i}",
                center=center,
                box=boxes[i] if boxes is not None else None,
                confidence=confidences[i] if confidences is not None else None,
                source="model",
            )
            detections.append(detection)

        return CorrectionState(
            image_id=image_id,
            sample_type=sample_type,
            model_detections=detections,
        )

    def add_point(
        self,
        state: CorrectionState,
        x: int,
        y: int,
    ) -> CorrectionState:
        state.add_user_point(x, y)
        return state

    def remove_nearest_detection(
        self,
        state: CorrectionState,
        x: int,
        y: int,
        max_distance: float = 20.0,
    ) -> CorrectionState:
        active_detections = state.active_model_detections()

        if not active_detections:
            return state

        click_point = (x, y)

        nearest_detection = min(
            active_detections,
            key=lambda detection: dist(click_point, detection.center),
        )

        nearest_distance = dist(click_point, nearest_detection.center)

        if nearest_distance <= max_distance:
            state.remove_detection(nearest_detection.id)

        return state

    def remove_nearest_added_point(
        self,
        state: CorrectionState,
        x: int,
        y: int,
        max_distance: float = 20.0,
    ) -> CorrectionState:
        if not state.added_points:
            return state

        click_point = (x, y)

        nearest_point = min(
            state.added_points,
            key=lambda point: dist(click_point, point.center),
        )

        nearest_distance = dist(click_point, nearest_point.center)

        if nearest_distance <= max_distance:
            state.added_points = [
                point
                for point in state.added_points
                if point.id != nearest_point.id
            ]

        return state

    def reset_corrections(self, state: CorrectionState) -> CorrectionState:
        state.added_points.clear()
        state.removed_detection_ids.clear()
        return state

    def save_state(self, state: CorrectionState) -> Path:
        save_path = self.corrections_dir / f"{state.image_id}.json"

        with open(save_path, "w", encoding="utf-8") as file:
            json.dump(state.to_dict(), file, indent=4)

        return save_path

    def load_state(self, image_id: str) -> CorrectionState:
        load_path = self.corrections_dir / f"{image_id}.json"

        if not load_path.exists():
            raise FileNotFoundError(f"No correction file found for image_id: {image_id}")

        with open(load_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        return CorrectionState.from_dict(data)

    def state_exists(self, image_id: str) -> bool:
        return (self.corrections_dir / f"{image_id}.json").exists()