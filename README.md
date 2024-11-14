Create a new repository on GitHub without a README.

SAM2_Eval/
├── SAM2_Eval_Surgical_Datasets/
│   ├── All_Dataset_SAM_Eval.ipynb       # Notebook for testing SAM2
│   ├── configs/                         # Configuration files
│   ├── src/                             # Source code
│   ├── utils/                           # Utility scripts directory
│   │   └── path_utils.py                # Path utility script
│   ├── README.md                        # Documentation
│   └── requirements.txt                 # Python dependencies
└── Datasets/                            # Dataset directory (not part of the Git repository)




## Edit this further
SAM2 Surgical Dataset Evaluation
This project evaluates the segmentation performance of SAM2 (Segment Anything Model) across multiple surgical datasets. The model is applied to datasets with diverse anatomical structures, allowing comparison and validation of SAM2's effectiveness in medical image segmentation.

Project Description
The SAM2 model is a segmentation model designed to generalize across different object classes, making it particularly useful for complex medical images. This project aims to:

Evaluate SAM2's performance on various surgical datasets, including Endoscapes, UD Ureter, CholecSeg8k, and m2caiSeg.
Provide modular and reusable code to streamline evaluation across new datasets.
Enable flexible dataset access, both locally and through Google Drive, for seamless use in different environments.
The project structure is organized to keep code and datasets separate, ensuring that code is version-controlled without including large datasets.

Setup Instructions
Local Setup
Clone the Repository: Clone this repository to your local machine:

bash
Copy code
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
Install Dependencies: Install the required Python packages:

bash
Copy code
pip install -r requirements.txt
Prepare the Datasets: Place the datasets in a folder named Datasets within the parent directory of the project folder. The structure should look like this:

bash
Copy code
SAM2_Eval/
├── SAM2_Eval_Surgical_Datasets/     # Cloned repository with code
└── Datasets/                        # Folder containing datasets
    ├── Endoscapes/
    ├── UD Ureter-Uterine Artery-Nerve Dataset/
    ├── CholecSeg8k/
    └── m2caiSeg/
Run the Notebook: Open and run All_Dataset_SAM_Eval.ipynb to test SAM2's segmentation performance on the datasets.

Google Colab Setup
Open the Notebook in Google Colab: Upload or open All_Dataset_SAM_Eval.ipynb in Google Colab.

Mount Google Drive: The code automatically mounts Google Drive when it detects a Colab environment. Ensure your datasets are saved in a directory within Google Drive with the following structure:

mathematica
Copy code
MyDrive/
└── SAM2_Eval/
    └── Datasets/
        ├── Endoscapes/
        ├── UD Ureter-Uterine Artery-Nerve Dataset/
        ├── CholecSeg8k/
        └── m2caiSeg/
Run the Notebook: With Google Drive mounted, run the cells in the notebook. The get_dataset_path() function will handle path selection automatically, using the Google Drive path if in Colab.

Dataset Paths and Usage Instructions
Local Path: Place datasets in ../Datasets relative to the SAM2_Eval_Surgical_Datasets directory.
Google Drive Path: Store datasets in MyDrive/SAM2_Eval/Datasets in your Google Drive.
The project will use the appropriate paths based on the environment (local or Colab). Ensure the directory names match the above structure for seamless access.
