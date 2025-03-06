#python yolo_split_crop_data.py --main_dir ../../Mint_weed_data/Mint --output_dir ./drone_dslr_cropped_data/TallFescue --classes Weed --bbox_scaling_factor 1.75 --crop_size 800 --overlap_pct 0.2 --val_pct 0.1 --test_pct 0.1 --dslr_resize 1024


#!/usr/bin/env python
import os
import json
import cv2
from decimal import Decimal
import argparse
from sklearn.model_selection import train_test_split

# Function to convert polygon to bounding box
def convert_polygon_to_bbox(points, scaling_factor=1.0):
    x_coordinates = [point[0] for point in points]
    y_coordinates = [point[1] for point in points]
    x_min = min(x_coordinates)
    y_min = min(y_coordinates)
    x_max = max(x_coordinates)
    y_max = max(y_coordinates)
    # Here you could apply scaling_factor if needed
    return int(x_min), int(y_min), int(x_max), int(y_max)

# Function to convert bounding box to YOLO format
def convert_to_yolo_format(x_min, y_min, x_max, y_max, img_width, img_height):
    x_center = Decimal(x_min + x_max) / 2 / Decimal(img_width)
    y_center = Decimal(y_min + y_max) / 2 / Decimal(img_height)
    width_norm = Decimal(x_max - x_min) / Decimal(img_width)
    height_norm = Decimal(y_max - y_min) / Decimal(img_height)
    return float(x_center), float(y_center), float(width_norm), float(height_norm)

# Function to recursively find all images and corresponding JSON files
def find_images_and_jsons(folder_path):
    image_files = []
    json_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.jpg'):
                image_files.append(os.path.join(root, file))
                json_file = os.path.join(root, file.rsplit('.', 1)[0] + '.json')
                json_files.append(json_file)
    return image_files, json_files

# Function to crop an image based on bounding boxes from its JSON annotation,
# then save both the cropped image and its corresponding YOLO annotation.
def crop_bboxes_and_save_all(image_path, json_path, output_images_dir, output_labels_dir, crop_size=1000, classes=None):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return 0
    img_height, img_width, _ = image.shape

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON {json_path}: {e}")
        return 0

    annotations = data.get('shapes', [])
    crop_count = 0

    for ann in annotations:
        points = ann['points']
        # If a list of target classes is provided, skip annotations not in the list.
        if classes is not None:
            label = ann.get('label')
            if label not in classes:
                continue

        # Convert polygon to bounding box
        x_min, y_min, x_max, y_max = convert_polygon_to_bbox(points)
        bbox_center_x = (x_min + x_max) // 2
        bbox_center_y = (y_min + y_max) // 2

        # Calculate crop region centered at the bounding box (ensuring it stays within image boundaries)
        crop_x_min = max(0, bbox_center_x - crop_size // 2)
        crop_y_min = max(0, bbox_center_y - crop_size // 2)
        crop_x_max = min(img_width, crop_x_min + crop_size)
        crop_y_max = min(img_height, crop_y_min + crop_size)
        if crop_x_max - crop_x_min < crop_size:
            crop_x_min = max(0, crop_x_max - crop_size)
        if crop_y_max - crop_y_min < crop_size:
            crop_y_min = max(0, crop_y_max - crop_size)

        # Check if the bounding box center is within the crop region
        if crop_x_min <= bbox_center_x <= crop_x_max and crop_y_min <= bbox_center_y <= crop_y_max:
            crop_img = image[crop_y_min:crop_y_max, crop_x_min:crop_x_max].copy()
            crop_annotations = []
            # Process each annotation to see if it overlaps with the crop region.
            for ann_check in annotations:
                points_check = ann_check['points']
                label = ann_check.get('label')
                # Determine class id based on provided classes list or default behavior.
                if classes is not None:
                    if label in classes:
                        class_id = classes.index(label)
                    else:
                        continue
                else:
                    if label == "Weed":
                        class_id = 0
                    elif label == "Mint":
                        class_id = 1
                    else:
                        continue

                x_min_check, y_min_check, x_max_check, y_max_check = convert_polygon_to_bbox(points_check)
                # Check if the bounding box from the annotation overlaps the crop region.
                if x_max_check > crop_x_min and x_min_check < crop_x_max and \
                   y_max_check > crop_y_min and y_min_check < crop_y_max:
                    # Adjust the bounding box relative to the crop region.
                    new_x_min = max(0, x_min_check - crop_x_min)
                    new_y_min = max(0, y_min_check - crop_y_min)
                    new_x_max = min(crop_size, x_max_check - crop_x_min)
                    new_y_max = min(crop_size, y_max_check - crop_y_min)
                    x_center, y_center, width_norm, height_norm = convert_to_yolo_format(
                        new_x_min, new_y_min, new_x_max, new_y_max, crop_size, crop_size
                    )
                    annotation_str = f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}"
                    crop_annotations.append(annotation_str)
            if crop_annotations:
                os.makedirs(output_images_dir, exist_ok=True)
                os.makedirs(output_labels_dir, exist_ok=True)
                base_filename = os.path.splitext(os.path.basename(image_path))[0]
                crop_filename = f"{base_filename}_{crop_count}"
                crop_img_filepath = os.path.join(output_images_dir, f"{crop_filename}.jpg")
                crop_label_filepath = os.path.join(output_labels_dir, f"{crop_filename}.txt")
                cv2.imwrite(crop_img_filepath, crop_img)
                with open(crop_label_filepath, 'w') as f:
                    f.write("\n".join(crop_annotations))
                crop_count += 1

    print(f"Processed {crop_count} crops for {image_path}")
    return crop_count

# Function to split the dataset into train, valid, and test sets.
def split_dataset(all_images_dir, all_labels_dir, output_dir, train_pct=0.8, val_pct=0.1, test_pct=0.1):
    image_files = [f for f in os.listdir(all_images_dir) if f.lower().endswith('.jpg')]
    label_files = [f.rsplit('.', 1)[0] + '.txt' for f in image_files]

    image_paths = [os.path.join(all_images_dir, img) for img in image_files]
    label_paths = [os.path.join(all_labels_dir, lbl) for lbl in label_files]

    # First, split out the training set.
    train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
        image_paths, label_paths, test_size=(1 - train_pct), random_state=42
    )
    # From the remainder, split into validation and test sets.
    if temp_imgs:
        # Calculate the fraction for validation relative to the remainder.
        val_ratio = val_pct / (val_pct + test_pct)
    else:
        val_ratio = 0
    val_imgs, test_imgs, val_labels, test_labels = train_test_split(
        temp_imgs, temp_labels, test_size=(1 - val_ratio), random_state=42
    )

    splits = {
        "train": (train_imgs, train_labels),
        "valid": (val_imgs, val_labels),
        "test": (test_imgs, test_labels)
    }

    for split, (imgs, labels) in splits.items():
        split_images_dir = os.path.join(output_dir, split, "images")
        split_labels_dir = os.path.join(output_dir, split, "labels")
        os.makedirs(split_images_dir, exist_ok=True)
        os.makedirs(split_labels_dir, exist_ok=True)
        for img, lbl in zip(imgs, labels):
            # Here we use os.rename; alternatively, you can use shutil.copy.
            os.rename(img, os.path.join(split_images_dir, os.path.basename(img)))
            os.rename(lbl, os.path.join(split_labels_dir, os.path.basename(lbl)))
        print(f"{split} set: {len(imgs)} images")

def main():
    parser = argparse.ArgumentParser(description="YOLO Split and Crop Data Script")
    parser.add_argument('--main_dir', type=str, required=True,
                        help="Main directory containing images and JSON annotations")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Output directory for processed images and labels")
    parser.add_argument('--classes', type=str, default="",
                        help="Comma-separated list of target classes (optional)")
    parser.add_argument('--bbox_scaling_factor', type=float, default=1.0,
                        help="Bounding box scaling factor (currently not used in cropping)")
    parser.add_argument('--crop_size', type=int, default=1000,
                        help="Crop size for images")
    parser.add_argument('--overlap_pct', type=float, default=0.0,
                        help="Overlap percentage for cropping (currently not used)")
    parser.add_argument('--val_pct', type=float, default=0.1,
                        help="Validation set percentage")
    parser.add_argument('--test_pct', type=float, default=0.1,
                        help="Test set percentage")
    parser.add_argument('--dslr_resize', type=int, default=1024,
                        help="Resize dimension for DSLR images (currently not used)")
    args = parser.parse_args()

    # Process the classes argument into a list (if provided)
    if args.classes:
        classes = [c.strip() for c in args.classes.split(',')]
    else:
        classes = None

    # Directories for cropped images and labels
    cropped_images_dir = os.path.join(args.output_dir, "All_images")
    cropped_labels_dir = os.path.join(args.output_dir, "All_labels")
    os.makedirs(cropped_images_dir, exist_ok=True)
    os.makedirs(cropped_labels_dir, exist_ok=True)

    # Step 1: Crop images and save crops
    image_files, json_files = find_images_and_jsons(args.main_dir)
    total_crops = 0
    for img_file, json_file in zip(image_files, json_files):
        if os.path.exists(json_file):
            total_crops += crop_bboxes_and_save_all(
                img_file, json_file,
                cropped_images_dir, cropped_labels_dir,
                crop_size=args.crop_size, classes=classes
            )
        else:
            print(f"JSON file not found for image: {img_file}")

    print(f"Total crops generated: {total_crops}")

    # Step 2: Split dataset into train, valid, and test sets
    split_output_dir = os.path.join(args.output_dir, "split_data")
    os.makedirs(split_output_dir, exist_ok=True)
    # Compute training percentage as the remainder after subtracting validation and test percentages.
    train_pct = 1 - (args.val_pct + args.test_pct)
    split_dataset(cropped_images_dir, cropped_labels_dir, split_output_dir,
                  train_pct=train_pct, val_pct=args.val_pct, test_pct=args.test_pct)

if __name__ == '__main__':
    main()
