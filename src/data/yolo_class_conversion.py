from pathlib import Path

src_root = Path("data/dataset_yolo_species/labels")
dst_root = Path("data/dataset_yolo_counting/labels")

for txt_path in src_root.rglob("*.txt"):
    relative = txt_path.relative_to(src_root)
    out_path = dst_root / relative
    out_path.parent.mkdir(parents=True, exist_ok=True)

    new_lines = []

    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            parts[0] = "0"
            new_lines.append(" ".join(parts))

    with open(out_path, "w") as f:
        if new_lines:
            f.write("\n".join(new_lines) + "\n")