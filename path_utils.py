import os

def get_dataset_path():
    # Check if running in Googel Colab
    if 'COLAB_GPU' in os.environ:
        from google.colab import drive
        drive.mount('/content/drive')
        base_path = '/content/drive/MyDrive/SAM2_Eval/Datasets'

    else:
        # Local dataset path (relative to the Git repo root)
        base_path = os.path.join(os.path.dirname(__file__), '../Datasets')

    return base_path