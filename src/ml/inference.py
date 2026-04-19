import os
import cv2
import yaml
from pathlib import Path
from ultralytics import YOLO


def load_config(config_path="configs/inference.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return YOLO(model_path)


def get_image_paths(source):
    source = Path(source)

    if source.is_file():
        return [source]

    if source.is_dir():
        exts = [".jpg", ".jpeg", ".png", ".bmp"]
        return [p for p in source.iterdir() if p.suffix.lower() in exts]

    raise ValueError(f"Invalid source: {source}")


def draw_predictions(image, boxes, pred_count):
    vis = image.copy()

    # Draw colony centers
    for box in boxes:
        x1, y1, x2, y2 = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        cv2.circle(vis, (cx, cy), 3, (0, 255, 0), -1)

    # Draw count text
    cv2.putText(
        vis,
        f"Predicted: {pred_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return vis


def run_inference(model, image_path, config):
    results = model(
        str(image_path),
        conf=config["conf"],
        iou=config["iou"],
        imgsz=config["imgsz"],
        max_det=config.get("max_det", 1000),
        verbose=False,
    )

    r = results[0]

    if r.boxes is None:
        return 0, [], None

    boxes = r.boxes.xyxy.cpu().numpy()
    pred_count = len(boxes)

    return pred_count, boxes, r.orig_img


def run_inference_on_image(model, image, config):
    results = model(
        image,
        conf=config["conf"],
        iou=config["iou"],
        imgsz=config["imgsz"],
        max_det=config.get("max_det", 1000),
        verbose=False,
    )

    r = results[0]

    if r.boxes is None:
        return 0, [], image

    boxes = r.boxes.xyxy.cpu().numpy()
    count = len(boxes)

    return count, boxes, r.orig_img


def main():
    config = load_config()

    model = load_model(config["model"])

    source = config.get("source", "assets/example_inputs")
    save_dir = Path(config.get("save_dir", "outputs/inference"))
    save_dir.mkdir(parents=True, exist_ok=True)

    image_paths = get_image_paths(source)

    print(f"Running inference on {len(image_paths)} images...\n")

    for img_path in image_paths:
        pred_count, boxes, image = run_inference(model, img_path, config)

        print(f"{img_path.name}: pred={pred_count}")

        if image is None:
            continue

        vis = draw_predictions(image, boxes, pred_count)

        save_path = save_dir / img_path.name
        cv2.imwrite(str(save_path), vis)


if __name__ == "__main__":
    main()