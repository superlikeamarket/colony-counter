import json
import shutil
from pathlib import Path


class TrainingDataExporter:
    def __init__(
        self,
        corrections_dir: str | Path,
        images_dir: str | Path,
        export_dir: str | Path,
    ):
        self.corrections_dir = Path(corrections_dir)
        self.images_dir = Path(images_dir)
        self.export_dir = Path(export_dir)

        self.export_images_dir = self.export_dir / "images"
        self.export_labels_dir = self.export_dir / "labels"

        self.export_images_dir.mkdir(parents=True, exist_ok=True)
        self.export_labels_dir.mkdir(parents=True, exist_ok=True)

    def export_all(
        self,
        default_box_size: int = 20,
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> None:
        correction_files = list(self.corrections_dir.glob("*.json"))

        for correction_file in correction_files:
            self.export_one(
                correction_file=correction_file,
                default_box_size=default_box_size,
                image_width=image_width,
                image_height=image_height,
            )

    def export_one(
        self,
        correction_file: str | Path,
        default_box_size: int = 20,
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> None:
        correction_file = Path(correction_file)

        with open(correction_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        image_id = data["image_id"]

        image_path = self._find_image(image_id)

        if image_path is None:
            print(f"Skipping {image_id}: image not found")
            return

        if image_width is None or image_height is None:
            image_width, image_height = self._get_image_size(image_path)

        active_points = self._get_active_points(data)

        label_lines = []

        for point in active_points:
            x, y = point["center"]

            label_line = self._center_to_yolo_label(
                x=x,
                y=y,
                image_width=image_width,
                image_height=image_height,
                box_size=default_box_size,
            )

            label_lines.append(label_line)

        export_image_path = self.export_images_dir / image_path.name
        export_label_path = self.export_labels_dir / f"{image_path.stem}.txt"

        shutil.copy(image_path, export_image_path)

        with open(export_label_path, "w", encoding="utf-8") as file:
            file.write("\n".join(label_lines))

    def _get_active_points(self, data: dict) -> list[dict]:
        removed_ids = set(data.get("removed_detection_ids", []))

        active_model_points = [
            detection
            for detection in data.get("model_detections", [])
            if detection["id"] not in removed_ids
        ]

        added_points = data.get("added_points", [])

        return active_model_points + added_points

    def _center_to_yolo_label(
        self,
        x: int,
        y: int,
        image_width: int,
        image_height: int,
        box_size: int,
    ) -> str:
        class_id = 0

        box_width = box_size
        box_height = box_size

        x_center_norm = x / image_width
        y_center_norm = y / image_height
        width_norm = box_width / image_width
        height_norm = box_height / image_height

        return (
            f"{class_id} "
            f"{x_center_norm:.6f} "
            f"{y_center_norm:.6f} "
            f"{width_norm:.6f} "
            f"{height_norm:.6f}"
        )

    def _find_image(self, image_id: str) -> Path | None:
        possible_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]

        for extension in possible_extensions:
            image_path = self.images_dir / f"{image_id}{extension}"
            if image_path.exists():
                return image_path

        return None

    def _get_image_size(self, image_path: Path) -> tuple[int, int]:
        from PIL import Image

        with Image.open(image_path) as img:
            return img.size