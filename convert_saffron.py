import json
import random
from copy import deepcopy

random.seed(42)

INPUT = "project.json"

with open(INPUT, "r") as f:
    tasks = json.load(f)

images = []
annotations = []
ann_id = 1
empty_images = 0
warnings = 0

for task in tasks:

    img_id = task["inner_id"]
    filename = task["file_upload"]

    if len(task["annotations"]) == 0:
        empty_images += 1

        images.append({
            "id": img_id,
            "width": 1280,
            "height": 720,
            "file_name": filename
        })
        continue

    results = task["annotations"][0]["result"]

    if len(results) == 0:
        empty_images += 1
        continue

    width = 1280
    height = 720

    images.append({
        "id": img_id,
        "width": width,
        "height": height,
        "file_name": filename
    })

    rectangles = []
    keypoints = []

    for r in results:
        if r["type"] == "rectanglelabels":
            rectangles.append(r)

        elif r["type"] == "keypointlabels":
            keypoints.append(r)

    used = set()

    for box in rectangles:

        matched = None

        # First try matching using parentID (future-proof)
        if "id" in box:
            for idx, kp in enumerate(keypoints):
                if idx in used:
                    continue

                if kp.get("parentID") == box["id"]:
                    matched = (idx, kp)
                    break

        # Fall back to nearest unused keypoint in order
        if matched is None:
            for idx, kp in enumerate(keypoints):
                if idx not in used:
                    matched = (idx, kp)
                    break

        if matched is None:
            continue

        idx, kp = matched
        used.add(idx)

        bx = box["value"]["x"] * width / 100
        by = box["value"]["y"] * height / 100
        bw = box["value"]["width"] * width / 100
        bh = box["value"]["height"] * height / 100

        kx = kp["value"]["x"] * width / 100
        ky = kp["value"]["y"] * height / 100

        if not (bx <= kx <= bx + bw and by <= ky <= by + bh):
            warnings += 1
            print(
                f"Warning: keypoint outside bbox in image {filename}"
            )

        annotations.append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": 1,
            "bbox": [bx, by, bw, bh],
            "area": bw * bh,
            "iscrowd": 0,
            "num_keypoints": 1,
            "keypoints": [kx, ky, 2]
        })

        ann_id += 1

random.shuffle(images)

n = len(images)

train = images[:int(0.75 * n)]
val = images[int(0.75 * n):int(0.875 * n)]
test = images[int(0.875 * n):]

train_ids = {x["id"] for x in train}
val_ids = {x["id"] for x in val}
test_ids = {x["id"] for x in test}

train_ann = [a for a in annotations if a["image_id"] in train_ids]
val_ann = [a for a in annotations if a["image_id"] in val_ids]
test_ann = [a for a in annotations if a["image_id"] in test_ids]

cats = [{
    "id": 1,
    "name": "flower",
    "supercategory": "flower",
    "keypoints": ["cut_point"],
    "skeleton": []
}]


def save(name, imgs, anns):

    with open(name, "w") as f:
        json.dump({
            "images": imgs,
            "annotations": anns,
            "categories": deepcopy(cats)
        }, f, indent=2)


save("train.json", train, train_ann)
save("val.json", val, val_ann)
save("test.json", test, test_ann)

print("\n========== Dataset Summary ==========")
print(f"Images           : {len(images)}")
print(f"Annotations      : {len(annotations)}")
print(f"Empty images     : {empty_images}")
print(f"Warnings         : {warnings}")
print("-------------------------------------")
print(f"Train : {len(train)} images | {len(train_ann)} annotations")
print(f"Val   : {len(val)} images | {len(val_ann)} annotations")
print(f"Test  : {len(test)} images | {len(test_ann)} annotations")
print("=====================================")