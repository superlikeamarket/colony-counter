🧫 Colony Counter

A computer vision pipeline for automated bacterial colony counting from Petri dish images.

This project focuses on building a robust, production-style preprocessing and validation pipeline before implementing colony detection and counting.


🚧 Current Status

✅ Dataset splitting (stratified)
✅ Image validation pipeline (quality filtering)
🚧 Colony counting (in progress)


📁 Project Structure
colony-counter/
│
├── data/
│   ├── raw_dataset/         # original images + annotations
│   └── dataset_split/       # train/val/test splits
│
├── outputs/
│   └── validation/          # validation results + debug outputs
│
├── src/
│   ├── data/
│   │   └── split.py         # dataset splitting script
│   │
│   └── preprocessing/
│       ├── validation.py    # image validation pipeline
│       └── counting.py      # (WIP) colony counting
│
├── notebooks/               # experiments (optional)
├── tests/                   # tests (optional)
├── requirements.txt
└── README.md


📦 Features
1. Dataset Splitting

Implemented in:
📄

Stratified split by species (e.g. sp01, sp02, …)
Preserves class distribution across:
train
validation
test
Copies:
images
labels (.txt)
XML annotations
Splits metadata tables (.csv, .tsv, .xls/.xlsx)
Generates summary CSVs for each split

2. Image Validation Pipeline

Implemented in:
📄

Each image is automatically checked for:

🔍 Plate Detection
Hough Circle Transform
Best-circle selection based on:
center proximity
radius size
overflow penalty
📐 Frame Check
Ensures plate is fully inside image
Uses radius-based margin (resolution-independent)
✂️ Cropping + Masking
Crops around plate
Masks background to focus analysis on plate only
📏 Resolution Normalization
Resizes plates to fixed size (1024×1024)
Ensures consistent blur/brightness thresholds
🌫️ Blur Detection
Laplacian variance
Computed only inside plate region
☀️ Overexposure Detection
Fraction of very bright pixels
Computed only inside plate region
⚠️ Rejection Criteria

Images are rejected if:

no plate detected
plate out of frame
blurry
overexposed


📊 Outputs

Validation generates:

outputs/validation/
│
├── validation_results.csv
├── validation_previews/     # circle overlays
├── cropped_plates/
├── resized_plates/
└── masked_plates/

Example output:

filename,plate_detected,in_frame,blurry,blur_score,overexposed,overexposed_fraction,accepted,reasons_for_rejection
sp01_img02.jpg,True,True,True,22.71,False,0.0185,False,"blurry"


▶️ How to Run
1. Install dependencies
pip install -r requirements.txt
2. Split dataset
python src/data/split.py
3. Run validation
python src/preprocessing/validation.py


🧠 Design Philosophy

This project prioritizes:

Robust preprocessing before modeling
Resolution-invariant metrics
Clear modular structure
Debuggable outputs (visual + CSV)

Instead of jumping directly into ML, the pipeline ensures:

garbage in → garbage out is avoided


🚀 Next Steps
Colony Counting (in progress)

Planned pipeline:

Input: validated + masked plate
Preprocessing:
grayscale
thresholding
Colony detection:
contour detection / blob detection
Filtering:
size
circularity
Output:
colony count
annotated image


💡 Notes
Blur and exposure are computed only inside the plate, avoiding background bias
Resize step ensures thresholds are consistent across all images
Plate masking avoids edge artifacts (e.g. bright rim)


📌 Future Improvements
Adaptive blur thresholds
Illumination correction
Colony segmentation with ML (e.g. U-Net)
CLI interface
Unit tests


🧑‍💻 Author

Built as part of a microbiology + computer vision project.