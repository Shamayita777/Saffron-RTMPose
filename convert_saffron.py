import json
import random
import math
from copy import deepcopy

random.seed(42)

DATASET = "saffron/annotations/result.json"

with open(DATASET, "r") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]
categories = coco["categories"]

# ---------------------------------------------------
# Remove images without annotations
# ---------------------------------------------------

annotated_ids = {a["image_id"] for a in annotations}
images = [img for img in images if img["id"] in annotated_ids]

print(f"Images      : {len(images)}")
print(f"Annotations : {len(annotations)}")

# ---------------------------------------------------
# Merge flower bbox + nearest cutting point
# ---------------------------------------------------

new_annotations = []
new_id = 1

for img in images:

    img_id = img["id"]

    anns = [a for a in annotations if a["image_id"] == img_id]

    flowers = [a for a in anns if a["category_id"] == 0]
    points = [a for a in anns if a["category_id"] == 1]

    used_points = set()

    for flower in flowers:

        if "bbox" not in flower:
            continue

        bx, by, bw, bh = flower["bbox"]

        cx = bx + bw / 2
        cy = by + bh / 2

        best = None
        best_dist = 1e18

        for i, pt in enumerate(points):

            if i in used_points:
                continue

            if "keypoints" not in pt:
                continue

            kp = pt["keypoints"]

            x = kp[0]
            y = kp[1]

            d = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)

            if d < best_dist:
                best_dist = d
                best = (i, pt)

        if best is None:
            continue

        idx, pt = best
        used_points.add(idx)

        kp = pt["keypoints"]

        ann = {
            "id": new_id,
            "image_id": img_id,
            "category_id": 1,
            "bbox": flower["bbox"],
            "area": flower.get("area", bw * bh),
            "iscrowd": 0,
            "num_keypoints": 1,
            "keypoints": kp
        }

        new_annotations.append(ann)
        new_id += 1

print("Merged annotations :", len(new_annotations))

# ---------------------------------------------------
# Shuffle images
# ---------------------------------------------------

random.shuffle(images)

n = len(images)

train_imgs = images[: int(0.75 * n)]
val_imgs = images[int(0.75 * n): int(0.875 * n)]
test_imgs = images[int(0.875 * n):]

train_ids = {i["id"] for i in train_imgs}
val_ids = {i["id"] for i in val_imgs}
test_ids = {i["id"] for i in test_imgs}

train_ann = [a for a in new_annotations if a["image_id"] in train_ids]
val_ann = [a for a in new_annotations if a["image_id"] in val_ids]
test_ann = [a for a in new_annotations if a["image_id"] in test_ids]

# ---------------------------------------------------
# Categories for RTMPose
# ---------------------------------------------------

rtm_categories = [{
    "id": 1,
    "name": "flower",
    "supercategory": "flower",
    "keypoints": ["cut_point"],
    "skeleton": []
}]

# ---------------------------------------------------
# Save
# ---------------------------------------------------

def save(path, imgs, anns):

    out = {
        "images": imgs,
        "annotations": anns,
        "categories": deepcopy(rtm_categories)
    }

    with open(path, "w") as f:
        json.dump(out, f)

save("saffron/annotations/train.json", train_imgs, train_ann)
save("saffron/annotations/val.json", val_imgs, val_ann)
save("saffron/annotations/test.json", test_imgs, test_ann)

print("\n==============================")
print("RTMPOSE DATASET CREATED")
print("==============================")

print("Train :", len(train_imgs), len(train_ann))
print("Val   :", len(val_imgs), len(val_ann))
print("Test  :", len(test_imgs), len(test_ann))

print("\nSaved:")
print("saffron/annotations/train.json")
print("saffron/annotations/val.json")
print("saffron/annotations/test.json")