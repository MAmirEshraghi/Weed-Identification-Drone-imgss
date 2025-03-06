
# Weed Identification

This repository is designed to house cutting-edge computer vision models and tools for detecting weeds in agricultural fields using drone and DSLR imagery. Our aim is to enhance precision agriculture practices by providing an efficient and scalable solution for weed identification and management.

# Introduction


# Repository Structure
```plaintext
├── LICENSE
├── README.md              # This file
├── .gitmodules            # (Contains submodule info for Ultralytics repos if applicable)
├── ultralytics_yolov8/    # Code/modified version of Ultralytics YOLO v8 (regular folder)
├── runs/                  # Training and evaluation results (e.g., runs/16b_1gpu_50eps_03)
├── train_test_yolov8.py   # Script for training and testing the YOLO model
├── yolo_split_crop_data.py# Script for splitting and cropping image data for YOLO
├── yolov8n.pt             # Pre-trained model weights (YOLOv8 nano variant)
```
# Usage
## Data Preparation & Cropping
Use the yolo_split_crop_data.py script to convert annotations to YOLO format, split your dataset into training, validation, and test sets, and crop your images appropriately.

Example command:
```bash
python yolo_split_crop_data.py --main_dir path/to/your/images \
    --output_dir path/to/output_directory \
    --classes Weed,Mint \
    --bbox_scaling_factor 1.75 \
    --crop_size 800 \
    --overlap_pct 0.2 \
    --val_pct 0.1 \
    --test_pct 0.1 \
    --dslr_resize 1024
```
![01](https://github.com/user-attachments/assets/1d7e7771-1b4c-4725-acb6-3a7e8f73e768)


## Training & Testing
Run the train_test_yolov8.py script to train or test your model. For example:

```bash
python train_test_yolov8.py --data path/to/data.yaml --weights yolov8n.pt --epochs 50 --batch-size
```
![02](https://github.com/user-attachments/assets/7873f930-f9e8-447c-b072-5e3d242473e8)


# Results:

![dsmplr result](https://github.com/user-attachments/assets/fdda38a0-9905-4121-af6c-e4cdeefbdcca)
![Rec0031-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/cb68a907-a5fc-4c81-b167-198ca31c9319)

![results](https://github.com/user-attachments/assets/ab3e1e0e-1ebe-4709-a6d4-cfc58aeebe66)
=======
