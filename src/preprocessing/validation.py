from src.config import ValidationConfig
from pathlib import Path
import cv2
import numpy as np
import math


# =========================
# IMAGE LOADING
# =========================

def load_image(image_path: Path) -> np.ndarray:
    """
    Load an image from disk.

    Return:
        image as numpy array
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return image


# =========================
# PLATE DETECTION
# =========================

def detect_plate(image: np.ndarray):
    """
    Detect the Petri dish as a circle.

    Strategy:
    - detect multiple candidate circles with HoughCircles
    - reject circles that are clearly implausible
    - score remaining circles by:
        1. closeness to image center
        2. how much of the circle stays inside the frame
        3. radius size (prefer large but reasonable circles)

    Returns:
        plate_detected (bool)
        circle (x, y, r) or None
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)

    h, w = image.shape[:2]
    image_center_x = w / 2
    image_center_y = h / 2
    min_dim = min(h, w)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(min_dim // 3),
        param1=100,
        param2=30,
        minRadius=int(min_dim * 0.35),  # minimum radius (35% of smaller dimension)
        maxRadius=int(min_dim * 0.55)   # maximum radius (55% of smaller dimension)
    )

    if circles is None:
        return False, None
    
    circles = np.round(circles[0]).astype("int")

    best_circle = None
    best_score = float("-inf")

    for (x, y, r) in circles:
        # Reject obviously bad radii
        if r < int(min_dim * 0.35) or r > int(min_dim * 0.55):
            continue

        # Distance of circle center from image center
        center_distance = np.sqrt((x - image_center_x) ** 2 + (y - image_center_y) ** 2)

        # How far the circle extends outside the frame
        overflow_left = max(0, -(x - r))
        overflow_top = max(0, -(y - r))
        overflow_right = max(0, (x + r) - w)
        overflow_bottom = max(0, (y + r) - h)
        total_overflow = overflow_left + overflow_top + overflow_right + overflow_bottom

        # Score:
        # - prefer circles near image center
        # - strongly penalize overflow
        # - slightly reward larger radius
        score = (
            - center_distance
            - 5 * total_overflow
            + 0.5 * r
        )

        if score > best_score:
            best_score = score
            best_circle = (x, y, r)

    if best_circle is None:
        return False, None

    return True, best_circle


def debug_detected_circles(image: np.ndarray, save_path: Path):
    """
    Save an image showing all candidate circles from HoughCircles.
    Useful for debugging bad detections.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)

    h, w = image.shape[:2]
    min_dim = min(h, w)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dim // 3,
        param1=100,
        param2=30,
        minRadius=int(min_dim * 0.35),
        maxRadius=int(min_dim * 0.55),
    )

    preview = image.copy()

    if circles is not None:
        circles = np.round(circles[0]).astype(int)

        for i, (x, y, r) in enumerate(circles):
            cv2.circle(preview, (x, y), r, (255, 0, 0), 2)   # blue candidate circle
            cv2.circle(preview, (x, y), 4, (0, 255, 255), -1)  # yellow center
            cv2.putText(
                preview,
                str(i),
                (x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

    cv2.imwrite(str(save_path), preview)


# =========================
# FRAME CHECK
# =========================

def is_plate_fully_in_frame(
        image: np.ndarray, circle, config: ValidationConfig
    ) -> bool:
    """
    Check if detected plate is fully inside the image.

    Input:
        - circle = (x, y, r)
        - config
            - plate_in_frame_margin_fraction = allow small margin (as a
                fraction of radius) to account for minor detection errors

    Return:
        True / False

    Logic:
    - plate should not touch image borders
    """
    h, w = image.shape[:2]
    x, y, r = circle

    # allow small margin (% of plate radius) to account for minor detection errors
    margin = int(r * config.plate_in_frame_margin_fraction)

    return (
        x - r >= -margin and
        y - r >= -margin and
        x + r <= w + margin and
        y + r <= h + margin
    )


# =========================
# CREATE PLATE MASK
# =========================

def create_plate_mask(
        image: np.ndarray, circle, config: ValidationConfig
    ) -> np.ndarray:
    """
    Create a binary mask for the detected plate.

    Input:
        image: original image
        circle: (x, y, r) in original image coordinates
        config
            mask_shrink_factor: optional radius scaling factor, e.g. 0.95 to
                avoid bright rim

    Return:
        2D uint8 mask:
        - 255 inside plate
        - 0 outside plate
    """
    h, w = image.shape[:2]
    x, y, r = circle
    r = int(r * config.mask_shrink_factor)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)
    return mask


# =========================
# CROP THE PLATE
# =========================

def crop_plate(
        image: np.ndarray,
        circle,
        config: ValidationConfig
    ) -> tuple[np.ndarray, tuple[int, int, int], tuple[int, int]]:
    """
    Crop the image to the detected plate region.

    Input:
        circle = (x, y, r)
        config
            crop_padding_fraction = extra pixels to include around the plate

    Return:
        cropped image (square around the plate)
        new circle (x, y, r) in cropped image coordinates
        crop boundaries (left, top) for debugging
    """
    x, y, r = circle
    padding = int(r * config.crop_padding_fraction)
    left = max(0, x - r - padding)
    top = max(0, y - r - padding)
    right = min(image.shape[1], x + r + padding)
    bottom = min(image.shape[0], y + r + padding)

    new_circle = (x - left, y - top, r)

    return image[top:bottom, left:right], new_circle, (left, top)


# =========================
# MASK THE PLATE
# =========================
def mask_plate(
        cropped_image: np.ndarray, circle
    ) -> np.ndarray:
    """
    Mask the plate region in the cropped image.

    Input:
        cropped_image: image cropped around the plate
        circle = (x, y, r) in cropped image coordinates

    Return:
        masked image where only the plate region is visible and everything else
            is black
    """

    cropped_center_x, cropped_center_y, r = circle
    
    # Create a mask with the same dimensions as the cropped image
    mask = np.zeros(cropped_image.shape[:2], dtype=np.uint8)

    # Draw a white circle on the mask
    cv2.circle(mask, (cropped_center_x, cropped_center_y), r, 255, -1)

    # Apply the mask to the cropped image
    masked_image = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)

    return masked_image


# =========================
# PLATE RESIZING
# =========================

def resize_plate(
    cropped_image: np.ndarray,
    circle,
    config: ValidationConfig
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """
    Resize cropped plate to a fixed size and update circle coordinates.
    Needed so that we can apply the same blur/overexposure thresholds to all 
        images regardless of original resolution.

    Parameters:
        cropped_image: plate crop
        circle: (x, y, r) in cropped image coordinates
        config
            target_size: output image size (square)

    Returns:
        resized_image
        resized_circle (x, y, r)
    """
    h, w = cropped_image.shape[:2]

    # scale factor to fit the plate into target size
    scale = config.target_size / max(h, w)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(cropped_image, (new_w, new_h))

    # pad to exact square if needed
    canvas = np.zeros((config.target_size, config.target_size, 3), dtype=np.uint8)

    x_offset = (config.target_size - new_w) // 2
    y_offset = (config.target_size - new_h) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    # update circle coordinates
    x, y, r = circle

    new_x = int(x * scale) + x_offset
    new_y = int(y * scale) + y_offset
    new_r = int(r * scale)

    return canvas, (new_x, new_y, new_r)


# =========================
# BLUR DETECTION
# =========================

def compute_blur_score(image: np.ndarray, mask: np.ndarray | None = None) -> float:
    """
    Compute blur score using Laplacian variance.
    If a mask is provided, compute the score only on the masked region.

    Input:
    - image: input BGR image
    - mask: optional binary mask (255 inside plate, 0 outside)

    Return:
        float score
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    if mask is None:
        return laplacian.var()
    
    values = laplacian[mask == 255]
    if values.size == 0:
        return 0.0
    
    return values.var()


def is_blurry(
        image: np.ndarray,
        config: ValidationConfig,
        mask: np.ndarray | None = None
    ) -> tuple[bool, float]:
    """
    Determine if image is blurry.

    Input:
    - image: input BGR image
    config
        - blur_threshold: blur score threshold below which image is considered blurry
    - mask: optional binary mask (255 inside plate, 0 outside) to focus blur
        analysis on the plate region

    Return:
        (is_blurry: bool, score: float)
    """
    score = compute_blur_score(image, mask=mask)
    return score < config.blur_threshold, score


# =========================
# OVEREXPOSURE DETECTION
# =========================

def compute_overexposed_fraction(
    image: np.ndarray,
    config: ValidationConfig,
    mask: np.ndarray | None = None
) -> float:
    """
    Compute fraction of very bright pixels.

    If mask is provided, compute only on the masked region.

    Return:
        float (0 → 1)

    Strategy:
    - convert to grayscale
    - count pixels above threshold
    - divide by total pixels
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if mask is None:
        bright_pixels = np.sum(gray >= config.overexposed_pixel_threshold)
        total_pixels = gray.size
        return bright_pixels / total_pixels if total_pixels > 0 else 0.0
    
    plate_pixels = gray[mask == 255]
    if plate_pixels.size == 0:
        return 0.0

    bright_pixels = np.sum(plate_pixels >= config.overexposed_pixel_threshold)
    total_pixels = plate_pixels.size
    return bright_pixels / total_pixels if total_pixels > 0 else 0.0


def is_overexposed(
        image: np.ndarray,
        config: ValidationConfig,
        mask: np.ndarray | None = None
):
    """
    Determine if image is overexposed.

    Return:
        (is_overexposed: bool, fraction: float)
    """
    fraction = compute_overexposed_fraction(image, config, mask=mask)
    return fraction > config.overexposed_fraction_threshold, fraction


# =========================
# MARKER DETECTION (add later, move values to config)
# =========================

def detect_dark_regions(
        image: np.ndarray,
        threshold_value: int = 50,
        min_area: float = 100.0,
        max_circularity: float = 0.6
) -> list:
    """
    Detect dark irregular regions such as marker ink.

    Parameters:
        image: input BGR image
        threshold_value: grayscale threshold for dark pixels
        min_area: minimum contour area to keep
        max_circularity: maximum circularity allowed
            (lower values = more irregular shapes)

    Returns:
        List of contours corresponding to dark irregular regions.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold for dark pixels (e.g., < 50)
    # Dark pixels become white in the binary image, others become black
    _, thresh = cv2.threshold(
        gray,
        threshold_value,
        255,
        cv2.THRESH_BINARY_INV
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter by area (remove tiny noise)
    filtered_contours = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < min_area:
            continue

        perimeter = cv2.arcLength(cnt, True)

        # Avoid division by zero
        if perimeter == 0:
            continue

        circularity = 4 * math.pi * area / (perimeter ** 2)

        # Keep only irregular shapes (low circularity)
        if circularity <= max_circularity:
            filtered_contours.append(cnt)

    return filtered_contours


# =========================
# VISUALIZATION
# =========================

def draw_preview(image: np.ndarray, circle, save_path: Path) -> None:
    """
    Save a debug image with the detected plate drawn on it.

    Draw:
    - detected plate (circle)
    - plate center
    """
    preview = image.copy()

    if circle is not None:
        x, y, r = circle
        cv2.circle(preview, (x, y), r, (0, 255, 0), 2)  # green outer circle
        cv2.circle(preview, (x, y), 5, (0, 0, 255), -1)  # red center point

    cv2.imwrite(str(save_path), preview)


# =========================
# VALIDATION PIPELINE
# =========================

def validate_image(image_path: Path, config: ValidationConfig):
    """
    Run full validation pipeline on one image.

    Steps:
    1. load image
    2. detect plate
    3. check if plate is in frame
    4. crop the plate region (for blur and overexposure analysis)
    5. mask the plate region (for blur and overexposure analysis)
    6. check blur
    7. check overexposure
    8. check if the image is accepted or rejected based on criteria

    Return:
        dictionary with results
        image (for visualization)
        circle (for visualization)
        cropped_image (for visualization)
        cropped_circle (for visualization)
        resized_image (for visualization)
        resized_circle (for visualization)
        masked_image (for visualization)
    """
    image = load_image(image_path)

    plate_detected, circle = detect_plate(image)

    if not plate_detected:
        in_frame = False
        blurry = False
        blur_score = 0.0
        overexposed = False
        overexposed_fraction = 0.0
        cropped_image = image
        cropped_circle = None
        resized_image = image
        resized_circle = None
        masked_image = image
    else:
        in_frame = is_plate_fully_in_frame(image, circle, config)

        cropped_image, cropped_circle, crop_bounds = crop_plate(
            image, circle, config
        ) if circle is not None else (image, (0, 0, 0), (0, 0))
        
        # resize plate
        resized_image, resized_circle = resize_plate(
            cropped_image, cropped_circle, config
        ) if cropped_circle is not None else (cropped_image, None)
        
        # create masks on resized image for blur/overexposure analysis
        resized_plate_mask = create_plate_mask(
            resized_image, resized_circle, config
        ) if resized_circle is not None else None

        # compute blur on standardized image
        blurry, blur_score = is_blurry(resized_image, config, mask=resized_plate_mask)

        # create mask on original image for overexposure analysis (to avoid
        # bright rim affecting results)
        plate_mask = create_plate_mask(
            image, circle, config
        ) if circle is not None else None

        # compute overexposure on original image but only within the plate region
        overexposed, overexposed_fraction = is_overexposed(image, config, mask=plate_mask)

        # create masked image for visualization (only the plate region visible)
        masked_image = mask_plate(
            cropped_image, cropped_circle
        ) if cropped_circle is not None else cropped_image

    # Determine if image is accepted or rejected
    reasons_for_rejection = []

    if not plate_detected:
        reasons_for_rejection.append("no_plate")
    else:
        if not in_frame:
            reasons_for_rejection.append("out_of_frame")
        if blurry:
            reasons_for_rejection.append("blurry")
        if overexposed:
            reasons_for_rejection.append("overexposed")

    accepted = len(reasons_for_rejection) == 0

    result = {
        "filename": image_path.name,
        "plate_detected": plate_detected,
        "circle": circle,
        "in_frame": in_frame,
        "blurry": blurry,
        "blur_score": blur_score,
        "overexposed": overexposed,
        "overexposed_fraction": overexposed_fraction,
        "accepted": accepted,
        "reasons_for_rejection": reasons_for_rejection
    }

    return result, image, circle, cropped_image, cropped_circle, resized_image, resized_circle, masked_image


# =========================
# MAIN
# =========================

def main() -> None:
    """
    Main execution:

    1. make output dirs and collect all image paths
    2. loop over images
    3. validate each image
    4. print results
    5. save the result to a list (to be saved to CSV later)
    6. save preview images
    7. save cropped images
    8. save resized images
    9. save masked images
    10. save results to CSV

    Output example:

        img001.jpg
        plate_detected = True
        in_frame = True
        blurry = False (score=...)
        overexposed = False (fraction=...)
        accepted = True
    """
    config = ValidationConfig()

    # Step 1: ensure output dirs exist   
    config.ensure_validation_dirs()

    # Step 1.5: collect image paths
    image_paths = sorted(
        p for p in config.paths.input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in config.image_extensions
    )

    print(f"Found {len(image_paths)} images for validation.")

    results = []

    # Step 2: loop
    for image_path in image_paths:
        # Step 3: validate
        result, image, circle, cropped_image, cropped_circle, resized_image, resized_circle, masked_image = validate_image(image_path, config)

        # Debug: if plate detected but not in frame, save image with all
        # candidate circles drawn
        if result["plate_detected"] and not result["in_frame"]:
            debug_path = config.paths.preview_dir / f"{image_path.stem}_all_circles.jpg"
            debug_detected_circles(image, debug_path)

        # Step 4: print results
        print(f"\n{result['filename']}:")
        print(f"circle = {circle}")
        print(f"image size = {image.shape[:2]}")
        print(f"  plate_detected = {result['plate_detected']}")
        print(f"  in_frame = {result['in_frame']}")
        print(f"  blurry = {result['blurry']} (score={result['blur_score']:.2f})")
        print(
            f"  overexposed = {result['overexposed']} "
            f"(fraction={result['overexposed_fraction']:.4f})"
        )
        print(f"  accepted = {result['accepted']}")
        if not result["accepted"]:
            print(f"  reasons_for_rejection = {result['reasons_for_rejection']}")

        # Step 5: save the results to a list (to be saved to CSV later)
        results.append(result)

        # Step 6: save preview
        preview_path = config.paths.preview_dir / f"{image_path.stem}_preview.jpg"
        draw_preview(image, circle, preview_path)

        # Step 7: save cropped image
        cropped_path = config.paths.cropped_dir / f"{image_path.stem}_cropped.jpg"
        cv2.imwrite(str(cropped_path), cropped_image)

        # Step 8: save resized image
        resized_path = config.paths.resized_dir / f"{image_path.stem}_resized.jpg"
        cv2.imwrite(str(resized_path), resized_image)

        # Step 9: save masked image
        masked_path = config.paths.masked_dir / f"{image_path.stem}_masked.jpg"
        cv2.imwrite(str(masked_path), masked_image)

    # Step 10: save results to CSV
    with open(config.paths.validation_csv, "w") as f:
        f.write("filename,plate_detected,in_frame,blurry,blur_score,overexposed,overexposed_fraction,accepted,reasons_for_rejection\n")
        for r in results:
            reasons = ";".join(r["reasons_for_rejection"]) if r["reasons_for_rejection"] else ""
            f.write(
                f"{r['filename']},{r['plate_detected']},{r['in_frame']},"
                f"{r['blurry']},{r['blur_score']:.2f},"
                f"{r['overexposed']},{r['overexposed_fraction']:.4f},"
                f"{r['accepted']},\"{reasons}\"\n"
            )
    

if __name__ == "__main__":
    main()