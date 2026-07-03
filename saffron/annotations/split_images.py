import os
import json
import shutil

SOURCE_IMAGES = "../images"
OUTPUT_IMAGES = "../images"


def copy_split(json_file, split_name):
    with open(json_file, "r") as f:
        data = json.load(f)

    out_dir = os.path.join(OUTPUT_IMAGES, split_name)
    os.makedirs(out_dir, exist_ok=True)

    copied = 0
    missing = 0

    for img in data["images"]:
        filename = img["file_name"]

        src = os.path.join(SOURCE_IMAGES, filename)
        dst = os.path.join(out_dir, filename)

        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied += 1
        else:
            print(f"Missing: {filename}")
            missing += 1

    print(f"{split_name}: copied {copied} images ({missing} missing)")


copy_split("train.json", "train")
copy_split("val.json", "val")
copy_split("test.json", "test")

print("\nDone!")