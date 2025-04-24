# dataset.py - Custom dataset handling and training

import os
import shutil
import yaml
import logging
from pathlib import Path
from ultralytics import YOLO

logger = logging.getLogger('surveillance')

def prepare_custom_dataset(dataset_path):
    """
    Prepare a custom dataset for training
    
    Args:
        dataset_path (str): Path to the dataset directory
        
    Returns:
        str: Path to the prepared dataset
    """
    # Create dataset structure
    dataset_dir = Path(dataset_path)
    
    # Check if dataset directory exists
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory {dataset_dir} not found")
    
    # Create output directory
    output_dir = dataset_dir / "prepared"
    output_dir.mkdir(exist_ok=True)
    
    # Create train and val directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    # Create images and labels directories
    (train_dir / "images").mkdir(exist_ok=True)
    (train_dir / "labels").mkdir(exist_ok=True)
    (val_dir / "images").mkdir(exist_ok=True)
    (val_dir / "labels").mkdir(exist_ok=True)
    
    # Check if the dataset is already in YOLO format
    if (dataset_dir / "train" / "images").exists() and (dataset_dir / "train" / "labels").exists():
        logger.info("Dataset already in YOLO format, copying files...")
        
        # Copy train images and labels
        for img_file in (dataset_dir / "train" / "images").glob("*.*"):
            shutil.copy(img_file, train_dir / "images")
        
        for label_file in (dataset_dir / "train" / "labels").glob("*.txt"):
            shutil.copy(label_file, train_dir / "labels")
        
        # Copy validation images and labels if they exist
        if (dataset_dir / "val" / "images").exists() and (dataset_dir / "val" / "labels").exists():
            for img_file in (dataset_dir / "val" / "images").glob("*.*"):
                shutil.copy(img_file, val_dir / "images")
            
            for label_file in (dataset_dir / "val" / "labels").glob("*.txt"):
                shutil.copy(label_file, val_dir / "labels")
    else:
        logger.warning("Dataset not in YOLO format. Please ensure the dataset follows YOLO format")
        logger.info("Expected structure: dataset/train/images/, dataset/train/labels/, dataset/val/images/, dataset/val/labels/")
        
        # Check if at least images directory exists
        imgs_dir = dataset_dir / "images"
        if imgs_dir.exists():
            logger.info(f"Found images directory: {imgs_dir}")
            # Copy 80% of images to train and 20% to val
            img_files = list(imgs_dir.glob("*.*"))
            train_size = int(len(img_files) * 0.8)
            
            for i, img_file in enumerate(img_files):
                if i < train_size:
                    shutil.copy(img_file, train_dir / "images")
                else:
                    shutil.copy(img_file, val_dir / "images")
            
            # Check if labels directory exists
            labels_dir = dataset_dir / "labels"
            if labels_dir.exists():
                logger.info(f"Found labels directory: {labels_dir}")
                # Copy labels corresponding to images
                for i, img_file in enumerate(img_files):
                    label_file = labels_dir / f"{img_file.stem}.txt"
                    if label_file.exists():
                        if i < train_size:
                            shutil.copy(label_file, train_dir / "labels")
                        else:
                            shutil.copy(label_file, val_dir / "labels")
    
    # Create dataset.yaml file
    # Check if dataset.yaml exists in the dataset directory
    yaml_file = dataset_dir / "dataset.yaml"
    if yaml_file.exists():
        # Read existing yaml file
        with open(yaml_file, 'r') as f:
            dataset_config = yaml.safe_load(f)
    else:
        # Create default yaml file
        dataset_config = {
            'path': str(output_dir),
            'train': str(train_dir / "images"),
            'val': str(val_dir / "images"),
            'names': {
                0: 'person',
                1: 'bicycle',
                2: 'car',
                3: 'motorcycle',
                4: 'bus',
                5: 'truck',
                6: 'traffic light',
                7: 'fire hydrant'
            }
        }
    
    # Update paths in dataset config
    dataset_config['path'] = str(output_dir)
    dataset_config['train'] = str(train_dir / "images")
    dataset_config['val'] = str(val_dir / "images")
    
    # Write yaml file
    with open(output_dir / "dataset.yaml", 'w') as f:
        yaml.dump(dataset_config, f)
    
    logger.info(f"Dataset prepared successfully at {output_dir}")
    return str(output_dir)

def train_custom_model(dataset_path, epochs=10, batch_size=16, img_size=640):
    """
    Train a custom YOLOv8 model
    
    Args:
        dataset_path (str): Path to the prepared dataset
        epochs (int): Number of training epochs
        batch_size (int): Batch size
        img_size (int): Image size
        
    Returns:
        str: Path to the trained model
    """
    logger.info(f"Training custom model on {dataset_path}")
    
    # Load base model
    model = YOLO('yolov8n.pt')
    
    # Find dataset.yaml
    dataset_yaml = os.path.join(dataset_path, "dataset.yaml")
    if not os.path.exists(dataset_yaml):
        raise FileNotFoundError(f"Dataset config file {dataset_yaml} not found")
    
    # Train the model
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device='mps' if hasattr(model, 'device') and 'mps' in str(model.device) else 'cpu',
        save=True,
        project="runs/train",
        name="custom_model"
    )
    
    # Get the path to the trained model
    model_path = os.path.join("runs/train/custom_model/weights/best.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join("runs/train/custom_model/weights/last.pt")
    
    logger.info(f"Training completed. Model saved at {model_path}")
    return model_path
