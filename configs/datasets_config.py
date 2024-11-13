import os

# Base directory for datasets, relative to this configuration file
base_dataset_path = os.path.join(os.path.dirname(__file__), '../Datasets')

datasets_config = {
    "Endoscapes": {
        "frames_path": os.path.join(base_dataset_path, "Endoscapes"),
        "masks_path": os.path.join(base_dataset_path, "Endoscapes"),
        "subdirs": ["test", "train", "val"]
    },
    "UD_Ureter": {
        "frames_path": os.path.join(base_dataset_path, "UD Ureter-Uterine Artery-Nerve Dataset", "Frames"),
        "masks_path": os.path.join(base_dataset_path, "UD Ureter-Uterine Artery-Nerve Dataset", "Masks"),
        "subdirs": []
    },
    "CholecSeg8k": {
        "frames_path": os.path.join(base_dataset_path, "CholecSeg8k", "Frames"),
        "masks_path": os.path.join(base_dataset_path, "CholecSeg8k", "Masks"),
        "subdirs": []  
    },
    "m2caiSeg": {
        "frames_path": os.path.join(base_dataset_path, "m2caiSeg"),
        "masks_path": os.path.join(base_dataset_path, "m2caiSeg"),
        "subdirs": ["train", "test", "trainval"]
    }
}