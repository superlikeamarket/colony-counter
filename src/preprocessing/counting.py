from src.config import CountingConfig, ThresholdConfig, MorphologyConfig, ContourConfig, MaskConfig, WatershedConfig
from src.utils import ensure_csv_with_header
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import math


# =========================
# DATA LOADING
# =========================

# Load validation results csv
def load_accepted_filenames(config: CountingConfig) -> list:
    """
    Load validation results CSV and return list of accepted filenames.
    Assumes CSV has columns "filename" and "accepted" (boolean).
    """
    df = pd.read_csv(config.paths.validation_csv)
    accepted_filenames = df[df["accepted"] == True]["filename"].tolist()
    return accepted_filenames


# Select accepted images
def get_accepted_image_paths(accepted_filenames, config: CountingConfig):
    """
    For each accepted filename, check for existence of masked image with any of
        the accepted extensions.
    Return list of valid image paths.
    """
    image_paths = []
    for filename in accepted_filenames:
        stem = Path(filename).stem
        for ext in config.image_extensions:
            potential_path = config.paths.masked_dir / f"{stem}_masked{ext}"
            if potential_path.exists():
                image_paths.append(potential_path)
                break
    return image_paths


# =========================
# PROCESSING FUNCTIONS
# =========================

def save_debug_image(path: Path, image: np.ndarray):
    """
    Save image to disk.
    """
    cv2.imwrite(str(path), image)


# Load image
def load_masked_image(image_path: Path, config: CountingConfig) -> np.ndarray:
    """
    Load masked image from disk.
    """
    
    if image_path.suffix.lower() not in config.image_extensions:
        raise ValueError(f"Unsupported extension: {image_path.suffix}")
    
    if not image_path.exists():
        raise FileNotFoundError(f"Could not load image: {image_path}")

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    return image


# Grayscale conversion
def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert input image to grayscale.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


# Thresholding
def threshold_colonies(gray: np.ndarray, config: ThresholdConfig) -> np.ndarray:
    """
    Convert grayscale image to binary image where colonies are white.

    Input:
    - gray: Grayscale image (2D numpy array)
    - config
        - gaussian_kernel_size: Kernel size for Gaussian blur (tuple)
        - threshold_value: Threshold value for binary thresholding (int)
        - use_otsu: Whether to use Otsu's method for automatic thresholding (bool)

    Output:
    - binary: Binary image (2D numpy array) where colonies are white (255)
    """
    # 1. Blur image (reduce noise)
    blurred = cv2.GaussianBlur(gray, config.gaussian_kernel_size, 0)

    # 2. Apply threshold
    if config.use_otsu:
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        _, binary = cv2.threshold(
            blurred, config.threshold_value, 255, cv2.THRESH_BINARY
        )

    # 3. Return binary image
    return binary


# Cleaning (morphological operations)
def clean_binary(binary: np.ndarray, config: MorphologyConfig) -> np.ndarray:
    """
    Clean binary image using morphological operations.

    Steps:
    - opening: remove small white noise
    - optional closing: fill small gaps in colonies

    Input:
        binary: thresholded binary image
        config
            kernel_size: size of morphology kernel
            use_closing: whether to apply closing after opening

    Return:
        cleaned binary image
    """
    # Create a small square kernel
    kernel = np.ones(config.morph_kernel_size, dtype=np.uint8)

    # Opening = erosion followed by dilation
    if config.use_opening:
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    else:
        cleaned = binary.copy()

    # Optional closing = dilation followed by erosion
    if config.use_closing:
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    return cleaned


# Create plate mask to remove the outer edge of the plate
def create_counting_mask(image: np.ndarray, config: MaskConfig) -> np.ndarray:
    """
    Create a circular mask for counting.

    Assumes the masked plate image is centered and fills most of the image.
    The mask is slightly smaller than the full plate to remove the bright rim,
        while keeping most edge colonies.

    Input:
        image: masked plate image
        config
            shrink: fraction of radius to keep

    Return:
        2D uint8 mask:
        - 255 inside counting region
        - 0 outside
    """
    h, w = image.shape[:2]

    center_x = w // 2
    center_y = h // 2

    radius = min(h, w) // 2
    radius = int(radius * config.shrink_factor)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)

    return mask


# Apply the mask
def apply_counting_mask(binary: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply counting mask to binary image.

    Input:
        binary: thresholded or cleaned binary image
        mask: circular counting mask

    Return:
        masked binary image
    """
    masked_binary = cv2.bitwise_and(binary, binary, mask=mask)
    return masked_binary


# Contour finding
def find_contours(binary: np.ndarray) -> list:
    """
    Find contours in a binary image.

    Input:
        binary: binary image where blobs (colonies) are white and background is
            black

    Return:
        list of contours
    """
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    return contours


def get_watershed_markers(
    binary: np.ndarray,
    config: WatershedConfig
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build watershed markers from a binary colony mask.

    Input:
        binary: uint8 binary image, colonies white (255), background black (0)

    Return:
        dist_norm: normalized distance transform (float image scaled 0..1)
        markers: int32 connected-components markers for watershed
    """
    # Distance transform on white foreground
    dist = cv2.distanceTransform(
        binary, config.distance_metric, config.distance_mask_size
    )

    # Normalize only for debugging / thresholding convenience
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    # Keep only strong peaks = likely colony centers
    sure_fg = (dist_norm >= config.peak_threshold_fraction).astype(np.uint8) * 255

    # Remove tiny marker noise
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(sure_fg)

    cleaned_markers = np.zeros_like(labels, dtype=np.int32)
    next_id = 1

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= config.marker_min_area:
            cleaned_markers[labels == label_id] = next_id
            next_id += 1

    return dist_norm, cleaned_markers


def apply_watershed(
    original_image: np.ndarray,
    binary: np.ndarray,
    markers: np.ndarray
) -> np.ndarray:
    """
    Apply watershed using precomputed markers.

    Input:
        original_image: original color image
        binary: binary mask with colonies as white
        markers: int32 marker image, 0 means unknown/background

    Return:
        watershed_labels: int32 label image
            -1 = watershed boundary
             1..N = separated colony regions
    """
    # Watershed expects a 3-channel image
    color_image = original_image.copy()

    # Unknown region = foreground mask minus current markers
    unknown = np.where((binary > 0) & (markers == 0), 255, 0).astype(np.uint8)

    watershed_markers = markers.copy()

    # Background must be a valid positive label, not 0
    watershed_markers = watershed_markers + 1

    # Unknown must be 0
    watershed_markers[unknown == 255] = 0

    watershed_labels = cv2.watershed(color_image, watershed_markers)

    return watershed_labels


def watershed_labels_to_contours(labels: np.ndarray) -> list:
    """
    Convert watershed label image into contours.

    Keeps only positive object labels (>1).
    """
    contours = []

    unique_labels = np.unique(labels)

    for label_id in unique_labels:
        # skip watershed boundaries and background
        if label_id <= 1:
            continue

        region_mask = np.zeros(labels.shape, dtype=np.uint8)
        region_mask[labels == label_id] = 255

        found_contours, _ = cv2.findContours(
            region_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if found_contours:
            # there should usually be one contour per region
            largest = max(found_contours, key=cv2.contourArea)
            contours.append(largest)

    return contours

def draw_watershed_preview(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    preview = image.copy()

    # watershed boundaries are -1
    preview[labels == -1] = (0, 0, 255)

    return preview


# Contour filtering
def filter_contours(contours: list, config: ContourConfig):
    """
    Filter contours by area and circularity.

    Keeps contours that:
    - are not too small
    - are not too large
    - are sufficiently round

    Input:
        contours: list of detected contours
        config
            min_area: minimum contour area
            max_area: maximum contour area
            min_circularity: minimum circularity threshold

    Return:
        list of filtered contours
    """
    filtered_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)

        # Filter out contours that are too small or too big
        if not (config.min_area < area < config.max_area):
            continue

        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue

        circularity = 4 * math.pi * area / (perimeter ** 2)

        # Filter out contours that have smaller circularity than the threshold
        if circularity >= config.min_circularity:
            filtered_contours.append(contour)

    return filtered_contours


# Counting
def count_contours(contours: list) -> int:
    """
    Counts contours

    Input: 
        contours: a list of filtered contours
    
    Return:
        number of contours in the list
    """
    return len(contours)


# Preview drawing
def draw_count_preview(
        image: np.ndarray,
        contours: list,
        count: int
) -> np.ndarray:
    """
    Draw filtered contours and predicted colony count on a copy of the input
        image.

    Input:
        image: original masked color image
        contours: list of contours
        count: predicted number of colonies

    Return:
        preview image with contours and count label
    """
    preview = image.copy()

    # Draw contours
    cv2.drawContours(
        preview,
        contours,
        -1,
        (0, 255, 0),
        2
    )
    
    # Draw count text
    cv2.putText(
        preview,
        f"Count: {count}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3,
        cv2.LINE_AA
    )

    return preview


# Image processing scripts
# Process a single image
def process_image(image_path: Path, config: CountingConfig) -> dict:
    """
    Process an image all the way from loading to outputs.
    
    Input:
        image_path: path to the image that will be processed
        config
            csv_file_name: csv save file name
            save_to_csv: whether to save the counting results to csv, yes by default
            save_debug_images: whether to save the images to corresponding folders, yes by
                default
    
    Return:
        a dictionary of processing steps as keys and resulting images as values;
            also includes the lists of contours
    """
    # Load the image
    masked_image = load_masked_image(image_path, config)
    
    # Convert to grayscale
    gray = to_grayscale(masked_image)
    
    # Convert to binary and clean
    binary = threshold_colonies(gray, config.threshold)
    cleaned = clean_binary(binary, config.morphology)
    
    # Mask the cleaned binary image
    counting_mask = create_counting_mask(cleaned, config.mask)
    masked_for_counting = apply_counting_mask(cleaned, counting_mask)
    
    # Find and count raw contours
    if config.watershed.use_watershed:
        dist_norm, markers = get_watershed_markers(
            masked_for_counting,
            config.watershed
        )

        watershed_labels = apply_watershed(
            masked_image,
            masked_for_counting,
            markers
        )

        raw_contours = watershed_labels_to_contours(watershed_labels)
    else:
        dist_norm = None
        markers = None
        watershed_labels = None
        raw_contours = find_contours(masked_for_counting)

    num_raw_contours = count_contours(raw_contours)
    
    # Filter and count the contours
    filtered_contours = filter_contours(raw_contours, config.contour)
    num_filtered_contours = count_contours(filtered_contours)
    
    # Draw the count preview
    preview = draw_count_preview(masked_image, filtered_contours, num_filtered_contours)
    
    # Add every image and contours to the output dict
    outputs = {
        "image_path": image_path,
        "image_name": image_path.name.replace("_masked", ""),
        "masked_image": masked_image,
        "grayscale": gray,
        "thresholded": binary,
        "cleaned": cleaned,
        "counting_masked": masked_for_counting,
        "contour_preview": preview,
        "counting_mask": counting_mask,
        "distance_transform": dist_norm,
        "watershed_markers": markers,
        "watershed_labels": watershed_labels,
        "number_of_raw_contours": num_raw_contours,
        "number_of_filtered_contours": num_filtered_contours,
        "raw_contours": raw_contours,
        "filtered_contours": filtered_contours
    }
    
    # Save to CSV
    if config.save_to_csv:
        count_df = pd.DataFrame([{
            "image_name": outputs["image_name"],
            "colony_count": num_filtered_contours
        }])
        count_df.to_csv(config.paths.counting_csv, mode = "a", header = False, index=False)
    
    # Save images to corresponding directories if not asked otherwise
    if config.save_debug_images:
        save_debug_image(config.paths.grayscale_dir / f"{image_path.stem}_grayscale.jpg", gray)
        
        debug_images = {
            "thresholded": binary,
            "cleaned": cleaned,
            "counting_mask": counting_mask,
            "counting_masked": masked_for_counting,
        }
        
        for key, value in debug_images.items():
            save_debug_image(config.paths.threshold_dir / f"{image_path.stem}_{key}.jpg", value)

        if dist_norm is not None:
            dist_img = (dist_norm * 255).astype(np.uint8)
            save_debug_image(config.paths.threshold_dir / f"{image_path.stem}_distance.jpg", dist_img)

        if watershed_labels is not None:
            watershed_preview = draw_watershed_preview(masked_image, watershed_labels)
            save_debug_image(config.paths.threshold_dir / f"{image_path.stem}_watershed.jpg", watershed_preview)
        
        save_debug_image(config.paths.contour_preview_dir / f"{image_path.stem}_contours.jpg", preview)
    
    # Return the dict with outputs
    return outputs


# Process a batch of images
def run_batch(image_paths: list[Path], config: CountingConfig) -> pd.DataFrame:
    """
    Process a set of images all the way from loading to outputs.
    
    Input:
        image_paths: a list of paths to the images that will be processed
        config
    
    Return:
        a pandas dataframe with filename and predicted contour count
    """
    output_rows = []

    for image_path in image_paths:
        output_dict = process_image(image_path, config)
        output_rows.append({
            "image_name": output_dict["image_name"],
            "colony_count": output_dict["number_of_filtered_contours"]
        })
    
    return pd.DataFrame(output_rows)


# =========================
# MAIN LOOP
# =========================

def main():
    """
    Main execution:

    Output example:
    """

    # remove later
    # for i, contour in enumerate(contours):
    #     area = cv2.contourArea(contour)
    #     perimeter = cv2.arcLength(contour, True)

    #     if perimeter == 0:
    #         continue

    #     circularity = 4 * math.pi * area / (perimeter ** 2)
    #     print(f"Contour {i}: area={area:.1f}, circularity={circularity:.3f}")
    # end of remove-later block

    config = CountingConfig()
    config.ensure_counting_dirs()

    ensure_csv_with_header(
        config.paths.counting_csv,
        ["image_name", "colony_count"]
    )

    accepted = load_accepted_filenames(config)
    image_paths = get_accepted_image_paths(accepted, config)

    accepted_filenames = load_accepted_filenames(config)
    print(f"Accepted filenames in CSV: {len(accepted_filenames)}")

    image_paths = get_accepted_image_paths(accepted_filenames, config)
    print(f"Matched masked images: {len(image_paths)}")

    print(f"Processing {len(image_paths)} images...")

    for image_path in image_paths:
        outputs = process_image(image_path, config)
        print(f"{outputs['image_name']}: {outputs['number_of_filtered_contours']}")


if __name__ == "__main__":
    main()