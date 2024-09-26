import os
import zipfile

from pathlib import Path

import requests

def download_data(source: str, 
                  destination: str,
                  remove_source: bool = True) -> Path:
    """Downloads a zipped dataset from source and unzips to destination.

    Args:
        source (str): A link to a zipped file containing data.
        destination (str): A target directory to unzip data to.
        remove_source (bool): Whether to remove the source after downloading and extracting.
    
    Returns:
        pathlib.Path to downloaded data.
    
    Example usage:
        download_data(source="https://github.com/xyz/raw/main/data/dataset.zip",
                      destination="dataset_directory")
    """
    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / destination

    # If the image folder doesn't exist, download it and prepare it... 
    if image_path.is_dir():
        print(f"[INFO] {image_path} directory exists, skipping download.")
    else:
        print(f"[INFO] Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)
        
        # Download dataset
        target_file = Path(source).name
        with open(data_path / target_file, "wb") as f:
            request = requests.get(source)
            print(f"[INFO] Downloading {target_file} from {source}...")
            f.write(request.content)

        # Unzip dataset
        with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
            print(f"[INFO] Unzipping {target_file} data...") 
            zip_ref.extractall(image_path)

        # Remove .zip file
        if remove_source:
            os.remove(data_path / target_file)
    
    return image_path

# https://driveandact.com/dataset/inner_mirror.zip
# https://driveandact.com/dataset/kinect_color.zip
# https://driveandact.com/dataset/iccv_activities_3s.zip
#image_path = download_data(source="https://driveandact.com/dataset/iccv_openpose_3d.zip",
#                           destination="daa_3dpose_labels")

#image_path = download_data(source="https://driveandact.com/dataset/kinect_color.zip",
#                           destination="kinect_color")

image_path = download_data(source="https://driveandact.com/dataset/iccv_activities_3s.zip",
                           destination="kinect_color_annotation")
image_path