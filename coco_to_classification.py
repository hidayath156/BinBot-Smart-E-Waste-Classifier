import json
import os
import shutil
from collections import Counter

# =========================================================
# CONFIGURATION (CHANGE ONLY IF YOUR FOLDER NAMES DIFFER)
# =========================================================
BASE_DIR = os.path.join(os.getcwd(), "dataset")
SPLITS = ["train", "valid", "test"]   # IMPORTANT: use 'valid', not 'validation'
ANNOTATION_FILE = "_annotations.coco.json"

# =========================================================
# FUNCTION TO CONVERT ONE SPLIT
# =========================================================
def convert_split(split):
    print(f"\n==============================")
    print(f"Processing split: {split}")
    print(f"==============================")

    split_dir = os.path.join(BASE_DIR, split)

    if not os.path.exists(split_dir):
        print(f"❌ Split folder not found: {split_dir}")
        return

    ann_path = os.path.join(split_dir, ANNOTATION_FILE)

    if not os.path.exists(ann_path):
        print(f"❌ Annotation file missing in {split_dir}")
        print("📂 Files found:", os.listdir(split_dir))
        return

    print(f"📄 Using annotation file: {ann_path}")

    # Images are stored in the SAME folder
    images_dir = split_dir

    # -----------------------------------------------------
    # Load COCO annotation
    # -----------------------------------------------------
    with open(ann_path, "r") as f:
        coco = json.load(f)

    categories = {c["id"]: c["name"] for c in coco["categories"]}
    images = {img["id"]: img["file_name"] for img in coco["images"]}

    # -----------------------------------------------------
    # Collect all categories per image
    # -----------------------------------------------------
    image_to_categories = {}

    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        cat_name = categories.get(ann["category_id"])
        if cat_name:
            image_to_categories.setdefault(img_id, []).append(cat_name)

    # -----------------------------------------------------
    # Assign ONE dominant label per image
    # -----------------------------------------------------
    image_labels = {}
    for img_id, cat_list in image_to_categories.items():
        dominant_label = Counter(cat_list).most_common(1)[0][0]
        image_labels[img_id] = dominant_label

    # -----------------------------------------------------
    # Copy images into class folders
    # -----------------------------------------------------
    copied = 0
    skipped = 0

    for img_id, label in image_labels.items():
        fname = images.get(img_id)

        if not fname:
            skipped += 1
            continue

        src = os.path.join(images_dir, fname)

        if not os.path.exists(src):
            print(f"⚠️ Missing image file: {fname}")
            skipped += 1
            continue

        target_dir = os.path.join(split_dir, label)
        os.makedirs(target_dir, exist_ok=True)

        dst = os.path.join(target_dir, fname)
        shutil.copy(src, dst)
        copied += 1

    print(f"✅ {split} split completed")
    print(f"   ➤ Images copied : {copied}")
    print(f"   ➤ Images skipped: {skipped}")

# =========================================================
# RUN FOR ALL SPLITS
# =========================================================
print("\n🚀 Starting COCO → Classification Conversion")

for split in SPLITS:
    convert_split(split)

print("\n🎉 All splits converted successfully!")
print("📁 Your dataset is now CNN-ready")
