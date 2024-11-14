import os
import gc
from src.sam2_eval_utils import initialize_model, load_image, load_mask, process_image
from configs.datasets_config import datasets_config

def main():
    sam_model = initialize_model()
    for dataset_name, config in datasets_config.items():
        frames_path = config["frames_path"]
        masks_path = config["masks_path"]
        subdirs = config["subdirs"]

        print(f"\nProcessing dataset: {dataset_name}")

        # Iterate through each subdir, if specified
        for subdir in subdirs or [""]:
            frames_folder = os.path.join(frames_path, subdir, "Frames")
            masks_folder = os.path.join(masks_path, subdir, "Masks")
            predicted_masks_folder = os.path.join(masks_path, subdir, "PredictedMasks")
            os.makedirs(predicted_masks_folder, exist_ok=True)

            image_names = sorted(os.listdir(frames_folder))
            for idx, image_name in enumerate(image_names):
                image_path = os.path.join(frames_folder, image_name)
                mask_path = os.path.join(masks_folder, image_name)
                predicted_mask_path = os.path.join(predicted_masks_folder, f"{idx:05d}.png")

                # Load and check image and mask
                image = load_image(image_path)
                mask = load_mask(mask_path)
                if image is None or mask is None:
                    print(f"Error loading image or mask: {image_path} or {mask_path}")
                    continue

                # Run SAM2 inference on the image
                inference_state = sam_model.init_state(frames_folder)
                process_image(sam_model, inference_state, image, mask, predicted_mask_path, idx)
                
                # Reset the inference state
                sam_model.reset_state(inference_state)
                
                gc.collect()  # Free memory

    print("Segmentation processing completed for all datasets.")

if __name__ == "__main__":
    main()