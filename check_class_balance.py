import os

base = "dataset/train"

for item in os.listdir(base):
    class_path = os.path.join(base, item)

    # Only count folders (classes)
    if os.path.isdir(class_path):
        count = len([
            f for f in os.listdir(class_path)
            if os.path.isfile(os.path.join(class_path, f))
        ])
        print(item, count)
