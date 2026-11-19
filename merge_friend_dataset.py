import os
import shutil

# Friend dataset path
friend_dataset = r"C:\Users\Mohan Sai\Downloads\second dataset"

# Main dataset path
main_dataset = r"C:\Users\Mohan Sai\OneDrive\Desktop\OKMA\datasets\helmet_dataset"

for split in ["train", "valid", "test"]:

    friend_images = os.path.join(friend_dataset, split, "images")
    friend_labels = os.path.join(friend_dataset, split, "labels")

    main_images = os.path.join(main_dataset, split, "images")
    main_labels = os.path.join(main_dataset, split, "labels")

    # Copy images
    if os.path.exists(friend_images):
        for img in os.listdir(friend_images):
            shutil.copy2(
                os.path.join(friend_images, img),
                os.path.join(main_images, img)
            )

    # Copy labels after converting class 0 -> 1
    if os.path.exists(friend_labels):
        for txt in os.listdir(friend_labels):

            src = os.path.join(friend_labels, txt)
            dst = os.path.join(main_labels, txt)

            with open(src, "r") as f:
                lines = f.readlines()

            new_lines = []

            for line in lines:
                parts = line.strip().split()

                if len(parts) > 0 and parts[0] == "0":
                    parts[0] = "1"

                new_lines.append(" ".join(parts))

            with open(dst, "w") as f:
                f.write("\n".join(new_lines))

print("Friend dataset merged successfully!")