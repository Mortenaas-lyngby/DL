#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Union
import shutil
import argparse
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image


def is_cartoon(image_path: Union[str, Path], threshold: float = 0.5) -> bool:
    model = ResNet50(weights='imagenet')

    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    label = decode_predictions(preds, top=1)[0][0]
    print(f'{image_path.name}: {label}')

    # Check if the top prediction is 'comic_book'
    if label[1] == 'comic_book' and label[2] > threshold:
        return True
    else:
        return False


def process_directory(dataset_path: Path, threshold: float):
    if dataset_path == Path('/'):
        raise ValueError("The dataset path cannot be the root directory.")

    if not (dataset_path / "images").exists() or not (dataset_path / "labels").exists():
        raise FileNotFoundError("The dataset directory must contain 'images' and 'labels' subdirectories.")

    photo_only_path = dataset_path.parent / f"{dataset_path.name}_photo_only"
    cartoon_only_path = dataset_path.parent / f"{dataset_path.name}_cartoon_only"
    photo_images_path = photo_only_path / "images"
    photo_labels_path = photo_only_path / "labels"
    cartoon_images_path = cartoon_only_path / "images"
    cartoon_labels_path = cartoon_only_path / "labels"
    
    missing_labels = []

    for split in ["train", "val", "test"]:
        photo_images_path_split = photo_images_path / split
        photo_labels_path_split = photo_labels_path / split
        cartoon_images_path_split = cartoon_images_path / split
        cartoon_labels_path_split = cartoon_labels_path / split

        photo_images_path_split.mkdir(parents=True, exist_ok=True)
        photo_labels_path_split.mkdir(parents=True, exist_ok=True)
        cartoon_images_path_split.mkdir(parents=True, exist_ok=True)
        cartoon_labels_path_split.mkdir(parents=True, exist_ok=True)

        images_split_path = dataset_path / "images" / split
        labels_split_path = dataset_path / "labels" / split

        for image_path in images_split_path.glob("*.jpg"):
            corresponding_label_path = labels_split_path / f"{image_path.stem}.txt"
            if not corresponding_label_path.exists():
                missing_labels.append(image_path.name)
                continue

            if is_cartoon(image_path, threshold):
                shutil.copy(image_path, cartoon_images_path_split / image_path.name)
                shutil.copy(corresponding_label_path, cartoon_labels_path_split / corresponding_label_path.name)
            else:
                shutil.copy(image_path, photo_images_path_split / image_path.name)
                shutil.copy(corresponding_label_path, photo_labels_path_split / corresponding_label_path.name)
    
    with open(photo_only_path / 'missing_labels.txt', 'w') as file:
        for label in missing_labels:
            file.write(f"{label}\n")

    print(f"Photo-only dataset created at: {photo_only_path}")
    print(f"Cartoon-only dataset created at: {cartoon_only_path}")


def command_line_options():
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