# train_test_yolov8.py

import argparse
from ultralytics import YOLO

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train or Test YOLOv8 model.")
    subparsers = parser.add_subparsers(dest='command', required=True, help='Subcommand to run (train or test)')

    # Subparser for training
    train_parser = subparsers.add_parser('train', help='Train the YOLOv8 model')
    train_parser.add_argument('--model', type=str, required=True, help='Path to the trained model weights file.')
    train_parser.add_argument('--data', type=str, required=True, help='Path to the YAML file specifying the dataset.')
    train_parser.add_argument('--epochs', type=int, default=40, help='Number of epochs to train the model.')
    train_parser.add_argument('--batch', type=int, default=6, help='Number of images in each batch (needs to be a multiple of number of devices).')
    train_parser.add_argument('--imgsz', type=int, default=800, help='Image size to use.')
    train_parser.add_argument('--device', type=str, default='0,1,2', help='Comma-separated list of available GPUs in numerical order.')
    train_parser.add_argument('--workers', type=int, default=8, help='Number of worker threads for data loading (per RANK if Multi-GPU training).')
    train_parser.add_argument('--project', type=str, default='None', help='Name of the project directory where training outputs are saved. Allows for organized storage of different experiments.')
    train_parser.add_argument('--name', type=str, default='None', help='Name of the training run. Used for creating a subdirectory within the project folder, where training logs and outpus are stored.')

    # Subparser for testing
    test_parser = subparsers.add_parser('test', help='Test the YOLOv8 model')
    test_parser.add_argument('--model', type=str, required=True, help='Path to the model file to use for testing.')
    test_parser.add_argument('--data', type=str, required=True, help='Path to the YAML file specifying the dataset.')
    test_parser.add_argument('--split', type=str, default='test', help='Dataset split to use for validation (default is test).')
    test_parser.add_argument('--imgsz', type=int, default=800, help='Image size to use.')
    test_parser.add_argument('--batch', type=int, default=16, help='Batch size for validation.')
    test_parser.add_argument('--conf', type=float, default=0.001, help='Confidence threshold for predictions.')
    test_parser.add_argument('--iou', type=float, default=0.5, help='IoU threshold for NMS.')
    test_parser.add_argument('--device', type=str, default='0', help='Device to use for testing.')
    test_parser.add_argument('--save_json', type=bool, default=False, help='If True, saves the results to a JSON file for further analysis or integration with other tools.')
    test_parser.add_argument('--plots', type=bool, default=False, help='When set to True, generates and saves plots of predictions versus ground truth for visual evaluation of the model performance.')
    test_parser.add_argument('--print_metrics', type=bool, default=False, help='If True, Print mAP at a given IOU threshold.')
    return parser.parse_args()

def train_model(args):
    # Convert device string to list of integers
    device = [int(d) for d in args.device.split(',')]
    print(device)
    # Load a model
    model = YOLO(args.model)  # load a pretrained model (recommended for training)

    # Train the model
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        workers=args.workers,
        project = args.project,
        name = args.name,
    )

def test_model(args):
    # Load the trained model
    model = YOLO(args.model)
    device = [int(d) for d in args.device.split(',')]
    # Validate the model
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save_json=args.save_json,
        plots=args.plots,
    )
    if args.print_metrics:
        print(metrics.box.map50)

def main():
    args = parse_arguments()
    
    if args.command == 'train':
        train_model(args)
    elif args.command == 'test':
        test_model(args)

if __name__ == "__main__":
    main()
