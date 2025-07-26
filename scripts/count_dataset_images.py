import os
from collections import defaultdict

# Dataset names and their folder paths
DATASETS = [
    'blowhole_dataset',
    'crack_dataset',
    'ExcessiveReinforcement_dataset',
    'porosity_dataset',
    'spatter_dataset',
]

# Image file extensions to count
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

# Base datasets directory
BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'Datasets')


def count_images_in_folder(folder_path):
    count = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in IMAGE_EXTENSIONS:
                count += 1
    return count


def main():
    summary = defaultdict(dict)
    for dataset in DATASETS:
        dataset_path = os.path.join(BASE_DIR, dataset)
        for split in ['train', 'valid']:
            split_path = os.path.join(dataset_path, split)
            if os.path.isdir(split_path):
                num_images = count_images_in_folder(split_path)
                summary[dataset][split] = num_images
            else:
                summary[dataset][split] = 0

    print("Image count summary:")
    for dataset in DATASETS:
        print(f"{dataset}:")
        for split in ['train', 'valid']:
            print(f"  {split}: {summary[dataset][split]} images")

    # Optionally, write to a file
    with open('dataset_image_counts.txt', 'w') as f:
        for dataset in DATASETS:
            f.write(f"{dataset}:\n")
            for split in ['train', 'valid']:
                f.write(f"  {split}: {summary[dataset][split]} images\n")


if __name__ == "__main__":
    main()
