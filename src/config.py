from dataclasses import dataclass, field
from pathlib import Path
import cv2


# TO-DO
# make config.py 
# configurate all scripts with config.py
# write docstrings for every function 
# counting.py: tune min area, max area,min Circularity
# compare fixed threshold vs Otsu
# consider adaptive THRESHOLDING
# test contour merging/splitting behavior
# support touching colonies
# include edge colonies
# create a test skeleton
# add logging instead of print debugging
# add cli arguments


# =========================
# GLOBAL CONFIG
# =========================

DEFAULT_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# =========================
# src/data/split.py
# =========================

@dataclass
class SplitPathConfig:
    source_dir: Path = Path("local/data/raw_dataset")
    output_dir: Path = Path("local/data/dataset_split")


@dataclass
class SplitConfig:
    paths: SplitPathConfig = field(default_factory=SplitPathConfig)
    random_seed: int = 42
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    image_extensions: set[str] = field(
        default_factory=lambda: set(DEFAULT_IMAGE_EXTENSIONS)
    )
    table_files: list[str] = field(
        default_factory=lambda: ["annot_tab.csv", "annot_tab.tsv", "images.xls"]
    )
    filename_column: str = "image_name"


    # Directory setup
    def ensure_split_dirs(self) -> None:
        """
        Create all output folders needed for:
        - images/train, images/val, images/test
        - labels/train, labels/val, labels/test
        - annotations_xml/train, annotations_xml/val, annotations_xml/test
        - splits/
        """
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        
        for category in ['images', 'labels', 'annotations_xml']:
            for split in ['train', 'val', 'test']:
                dir_path = self.paths.output_dir / category / split
                dir_path.mkdir(parents=True, exist_ok=True)

        (self.paths.output_dir / 'splits').mkdir(parents=True, exist_ok=True)


# ===============================
# src/preprocessing/validation.py
# ===============================

@dataclass
class ValidationPathConfig:
    input_dir: Path = Path("local/data/raw_dataset")
    output_dir: Path = Path("local/outputs/validation")
    preview_dir: Path = Path("local/outputs/validation/validation_previews")
    cropped_dir: Path = Path("local/outputs/validation/cropped_plates")
    masked_dir: Path = Path("local/outputs/validation/masked_plates")
    resized_dir: Path = Path("local/outputs/validation/resized_plates")
    validation_csv: Path = Path("local/outputs/validation/validation_results.csv")


@dataclass
class ValidationConfig:
    paths: ValidationPathConfig = field(default_factory=ValidationPathConfig)
    image_extensions: set[str] = field(
        default_factory=lambda: set(DEFAULT_IMAGE_EXTENSIONS)
    )
    # allow small margin (10% of radius) to account for minor detection errors
    plate_in_frame_margin_fraction: float = 0.1
    # shrink the mask slightly to avoid bright rim affecting blur/overexposure
    # analysis
    mask_shrink_factor: float = 1
    # padding around the plate when cropping (5% of radius)
    crop_padding_fraction: float = 0.05
    # target size for resized plates (for consistent blur/overexposure 
    # thresholds)
    target_size: int = 1024
    # thresholds for blur and overexposure (to be tuned based on data)
    blur_threshold: float = 40.0
    # pixel intensity threshold for overexposure
    overexposed_pixel_threshold: int = 245
    overexposed_fraction_threshold: float = 0.05  # 5%


    # Directory setup
    def ensure_validation_dirs(self) -> None:
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        self.paths.preview_dir.mkdir(parents=True, exist_ok=True)
        self.paths.cropped_dir.mkdir(parents=True, exist_ok=True)
        self.paths.masked_dir.mkdir(parents=True, exist_ok=True)
        self.paths.resized_dir.mkdir(parents=True, exist_ok=True)


# =============================
# src/preprocessing/counting.py
# =============================

@dataclass
class CountingPathConfig:
    input_dir: Path = Path("local/outputs/validation")
    masked_dir: Path = Path("local/outputs/validation/masked_plates")
    validation_csv: Path = Path("local/outputs/validation/validation_results.csv")

    output_dir: Path = Path("local/outputs/counting")
    grayscale_dir: Path = Path("local/outputs/counting/grayscale")
    threshold_dir: Path = Path("local/outputs/counting/thresholded")
    contour_preview_dir: Path = Path("local/outputs/counting/contour_previews")
    counting_csv: Path = Path("local/outputs/counting/counting_results.csv")


@dataclass
class ThresholdConfig:
    gaussian_kernel_size: tuple[int, int] = (7, 7)
    threshold_value: int = 100
    use_otsu: bool = False


@dataclass
class MorphologyConfig:
    morph_kernel_size: tuple[int, int] = (3, 3)
    use_opening: bool = False
    use_closing: bool = False


@dataclass
class WatershedConfig:
    use_watershed: bool = True
    distance_metric: int = cv2.DIST_L2
    distance_mask_size: int = 5
    peak_threshold_fraction: float = 0.35
    marker_min_area: int = 10


@dataclass
class ContourConfig:
    min_area: int = 10
    max_area: int = 80000
    min_circularity: float = 0.42


@dataclass
class MaskConfig:
    shrink_factor: float = 0.925


@dataclass
class CountingConfig:
    paths: CountingPathConfig = field(default_factory=CountingPathConfig)
    image_extensions: set[str] = field(
        default_factory=lambda: set(DEFAULT_IMAGE_EXTENSIONS)
    )
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    morphology: MorphologyConfig = field(default_factory=MorphologyConfig)
    contour: ContourConfig = field(default_factory=ContourConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    watershed: WatershedConfig = field(default_factory=WatershedConfig)
    save_to_csv: bool = True
    save_debug_images: bool = True


    # Create output directories if they don't exist function
    def ensure_counting_dirs(self) -> None:
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        self.paths.grayscale_dir.mkdir(parents=True, exist_ok=True)
        self.paths.threshold_dir.mkdir(parents=True, exist_ok=True)
        self.paths.contour_preview_dir.mkdir(parents=True, exist_ok=True)