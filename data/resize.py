from PIL import Image
import os, glob
import random
import argparse
from pathlib import Path


def resize_images(directory, size):
    for img in os.listdir(directory):
        image = Image.open(directory + img)
        image_resized = image.resize(size, Image.ANTIALIAS)
        image_resized.save(directory + img)


# def shuffle_images(directory):
#     for img in os.listdir(directory):
#     image = Image.shuffle(img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize images")
    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        required=True,
        help="Directory containing the images",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        nargs=2,
        required=True,
        metavar=("width", "height"),
        help="Image size",
    )
    args = parser.parse_args()
    resize_images(args.directory, args.size)


# put your own path here
dataset_path = args.directory

# Percentage of images to be used for the validation set
percentage_test = 20

Path("./data").mkdir(parents=True, exist_ok=True)
Path("./data/images").mkdir(parents=True, exist_ok=True)
Path("./data/labels").mkdir(parents=True, exist_ok=True)
Path("./data/images/train").mkdir(parents=True, exist_ok=True)
Path("./data/images/valid").mkdir(parents=True, exist_ok=True)
Path("./data/labels/train").mkdir(parents=True, exist_ok=True)
Path("./data/labels/valid").mkdir(parents=True, exist_ok=True)


# Populate the folders
p = percentage_test / 100
for pathAndFilename in glob.iglob(os.path.join(dataset_path, "*.jpg")):
    title, ext = os.path.splitext(os.path.basename(pathAndFilename))
    if random.random() <= p:
        os.system(f"cp {dataset_path}/{title}.jpg data/images/valid")
        os.system(f"cp {dataset_path}/{title}.txt data/labels/valid")
    else:
        os.system(f"cp {dataset_path}/{title}.jpg data/images/train")
        os.system(f"cp {dataset_path}/{title}.txt data/labels/train")
