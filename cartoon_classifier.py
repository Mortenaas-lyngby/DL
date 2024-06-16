from pathlib import Path
import shutil
import argparse
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model():
    """
    Load the pre-trained ResNet50 model with ImageNet weights.

    Returns:
        model (tensorflow.keras.Model): Loaded ResNet50 model.
    """
    return ResNet50(weights='imagenet')

def is_cartoon(img_path: Path, model, threshold: float = 0.5) -> bool:
    """
    Determine if an image is a cartoon using the ResNet50 model.

    Args:
        img_path (Path): Path to the image file.
        model (tensorflow.keras.Model): Pre-trained ResNet50 model.
        threshold (float): Confidence threshold for classification.

    Returns:
        bool: True if the image is classified as a cartoon, False otherwise.
    """
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)
        label = decode_predictions(preds, top=1)[0][0]

        # Check if the top prediction is related to 'cartoon' with confidence above threshold
        keywords = ['comic', 'cartoon', 'animated', 'animation']
        for keyword in keywords:
            if keyword in label[1] and label[2] > threshold:
                return True
        
        return False

    except Exception as e:
        logging.error(f"Error processing image {img_path}: {str(e)}")
        return False

def process_directory(dataset_path: Path, threshold: float):
    """
    Process the dataset directory to separate cartoon and photo images.

    Args:
        dataset_path (Path): Path to the dataset directory.
        threshold (float): Confidence threshold for classification.

    Raises:
        FileNotFoundError: If the dataset directory does not exist.
    """
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise FileNotFoundError(f"No dataset directory exists at {dataset_path.absolute()}")

    photo_only_path = dataset_path.parent / f"{dataset_path.name}_photo_only"
    cartoon_only_path = dataset_path.parent / f"{dataset_path.name}_cartoon_only"
    
    splits = ["train", "val", "test"]
    model = load_model()

    for split in splits:
        photo_images_path_split = photo_only_path / "images" / split
        photo_labels_path_split = photo_only_path / "labels" / split
        cartoon_images_path_split = cartoon_only_path / "images" / split
        cartoon_labels_path_split = cartoon_only_path / "labels" / split

        photo_images_path_split.mkdir(parents=True, exist_ok=True)
        photo_labels_path_split.mkdir(parents=True, exist_ok=True)
        cartoon_images_path_split.mkdir(parents=True, exist_ok=True)
        cartoon_labels_path_split.mkdir(parents=True, exist_ok=True)

        images_split_path = dataset_path / "images" / split
        labels_split_path = dataset_path / "labels" / split

        images = list(images_split_path.glob("*.jpg"))
        total_images = len(images)

        # Initialize tqdm progress bar
        progress_bar = tqdm(total=total_images, desc=f"Processing {split} split", unit="image")

        for image_path in images:
            corresponding_label_path = labels_split_path / f"{image_path.stem}.txt"
            photo_image_dest = photo_images_path_split / image_path.name
            photo_label_dest = photo_labels_path_split / corresponding_label_path.name
            cartoon_image_dest = cartoon_images_path_split / image_path.name
            cartoon_label_dest = cartoon_labels_path_split / corresponding_label_path.name

            # Check if the image already exists in the target directories
            if photo_image_dest.exists() or cartoon_image_dest.exists():
                progress_bar.update(1)
                continue

            if corresponding_label_path.exists():
                if is_cartoon(image_path, model, threshold):
                    shutil.copy(image_path, cartoon_image_dest)
                    shutil.copy(corresponding_label_path, cartoon_label_dest)
                else:
                    shutil.copy(image_path, photo_image_dest)
                    shutil.copy(corresponding_label_path, photo_label_dest)
            else:
                logging.warning(f"Label for {image_path.name} not found.")
                if is_cartoon(image_path, model, threshold):
                    shutil.copy(image_path, cartoon_image_dest)
                else:
                    shutil.copy(image_path, photo_image_dest)

            progress_bar.update(1)

        progress_bar.close()

    logging.info(f"Photo-only dataset created at: {photo_only_path}")
    logging.info(f"Cartoon-only dataset created at: {cartoon_only_path}")

def command_line_options():
    """
    Parse command-line options.

    Returns:
        dict: Dictionary containing parsed command-line arguments.
    """
    args = argparse.ArgumentParser(description="Process dataset to filter out cartoons and keep only photos.")
    args.add_argument("-t", "--threshold", type=float, help="Classification threshold (default: 0.5)", default=0.5)
    args.add_argument("dataset_path", type=Path, help="Path to the dataset directory")
    return vars(args.parse_args())

if __name__ == "__main__":
    options = command_line_options()
    dataset_path = options["dataset_path"]
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise FileNotFoundError(f"No dataset directory exists at {dataset_path.absolute()}")
    process_directory(dataset_path, options["threshold"])