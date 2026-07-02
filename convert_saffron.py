"""
==========================================================
Saffron COCO Dataset Splitter
----------------------------------------------------------
Splits one COCO annotation file into
train.json / val.json / test.json

Author : Shamayita Moitra
==========================================================
"""

import json
import random
from pathlib import Path

# ==========================================================
# Configuration
# ==========================================================

SEED = 42

TRAIN_RATIO = 0.75
VAL_RATIO = 0.125
TEST_RATIO = 0.125

ROOT = Path("saffron")
ANNOTATION_FILE = ROOT / "annotations" / "result.json"
OUTPUT_DIR = ROOT / "annotations"

random.seed(SEED)

# ==========================================================
# Load COCO
# ==========================================================

with open(ANNOTATION_FILE, "r") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]
categories = coco["categories"]
info = coco.get("info", {})

# ==========================================================
# Keep only images that actually have annotations
# ==========================================================

annotated_image_ids = {ann["image_id"] for ann in annotations}

images = [
    img for img in images
    if img["id"] in annotated_image_ids
]

print(f"Images       : {len(images)}")
print(f"Annotations  : {len(annotations)}")

# ==========================================================
# Shuffle images
# ==========================================================

random.shuffle(images)

n = len(images)

train_end = int(n * TRAIN_RATIO)
val_end = train_end + int(n * VAL_RATIO)

train_images = images[:train_end]
val_images = images[train_end:val_end]
test_images = images[val_end:]

# ==========================================================
# Image ID sets
# ==========================================================

train_ids = {img["id"] for img in train_images}
val_ids = {img["id"] for img in val_images}
test_ids = {img["id"] for img in test_images}

# ==========================================================
# Split annotations
# ==========================================================

train_annotations = [
    ann for ann in annotations
    if ann["image_id"] in train_ids
]

val_annotations = [
    ann for ann in annotations
    if ann["image_id"] in val_ids
]

test_annotations = [
    ann for ann in annotations
    if ann["image_id"] in test_ids
]

# ==========================================================
# Save helper
# ==========================================================

def save_json(filename, imgs, anns):

    output = {
        "info": info,
        "images": imgs,
        "annotations": anns,
        "categories": categories
    }

    with open(OUTPUT_DIR / filename, "w") as f:
        json.dump(output, f, indent=4)

# ==========================================================
# Save
# ==========================================================

save_json("train.json", train_images, train_annotations)
save_json("val.json", val_images, val_annotations)
save_json("test.json", test_images, test_annotations)

# ==========================================================
# Summary
# ==========================================================

print("\n==========================================")
print("SAFFRON COCO SPLIT COMPLETED")
print("==========================================")

print(f"Train Images      : {len(train_images)}")
print(f"Validation Images : {len(val_images)}")
print(f"Test Images       : {len(test_images)}")

print()

print(f"Train Annotations : {len(train_annotations)}")
print(f"Validation Annots : {len(val_annotations)}")
print(f"Test Annotations  : {len(test_annotations)}")

print()

print("Saved:")

print(OUTPUT_DIR / "train.json")
print(OUTPUT_DIR / "val.json")
print(OUTPUT_DIR / "test.json")

print("==========================================")