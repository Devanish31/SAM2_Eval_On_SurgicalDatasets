import os
import cv2
import numpy as np
import random
import torch
import gc
from sam2.build_sam import build_sam2_video_predictor

device = "cpu"
# Set the relative paths for the model configuration and checkpoint
model_cfg = r"C:\Users\devan\Downloads\SAM2_Eval\SAM2_model_and_cfg\sam2_hiera_l.yaml"
checkpoint_path = r"C:\Users\devan\Downloads\SAM2_Eval\SAM2_model_and_cfg\sam2_hiera_large.pt"


def initialize_model():
    """Initialize the SAM2 model"""
    return build_sam2_video_predictor(
        config_file=model_cfg,
        ckpt_path=checkpoint_path,
        device=device
    )

def load_image(image_path):
    """Load an image given its path"""
    return cv2.imread(image_path)

def load_mask(mask_path):
    """Load a mask image given its path."""
    return cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

def select_point_prompt(mask, num_points=1):
    """
    Select a specified number of random prompt points with value 255 from the mask.
    
    Args: 
        mask (np.ndarray): The mask image array.
        num_points (int): The number of prompt points to select. Default is 1.
        
    Returns:
        list: A list of selected prompt coordinates. If there are fewer points available than requested,
              returns all available points.
    """
    # Find all coordinates in the mask where the value is 255
    coords = np.column_stack(np.where(mask == 255))
    print(f"Total prompt points found in mask: {len(coords)}")

    # If no valid points are found, return None
    if coords.size == 0:
        print("No valid points with value 255 found in the mask.")
        return None
    
    # If the number of points requested is greater than available points, use all points
    if num_points >= len(coords):
        print(f"Requested {num_points} points, but only {len(coords)} available. Returning all points.")
        return coords.tolist()

    # Randomly select the specified number of points
    selected_points = random.sample(coords.tolist(), num_points)
    print(f"Selected points: {selected_points}")

    return selected_points

def save_predicted_mask(predicted_mask, predicted_mask_path):
    """Save the predicted mask to disk."""
    predicted_mask_bgr = cv2.cvtColor(predicted_mask, cv2.COLOR_GRAY2BGR)
    success = cv2.imwrite(predicted_mask_path, predicted_mask_bgr)
    if success:
        print(f"Predicted mask successfully saved at: {predicted_mask_path}")
    else:
        print(f"Failed to save the predicted mask at: {predicted_mask_path}")

def process_image(sam_model, inference_state, image, mask, output_path, idx):
    """Run the SAM2 inference on a single image."""
    print(f"Starting inference on image index {idx}")
    
    # Select prompt coordinates
    prompt_coord = select_point_prompt(mask)
    if prompt_coord is None:
        print("No valid prompt point found for mask; skipping this image.")
        return
    
    print(f"Selected prompt coordinates: {prompt_coord}")
    
    # Set up inference parameters
    obj_id = idx + 1
    frame_idx = 0
    points = [[prompt_coord[1], prompt_coord[0]]]
    labels = [1]

    # Perform inference
    print(f"Running SAM2 model inference on frame {frame_idx} with object ID {obj_id}")
    _, out_obj_ids, out_mask_logits = sam_model.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        points=points,
        labels=labels,
    )

    # Check if inference produced a valid mask
    if out_mask_logits is not None:
        print(f"Inference successful; generating predicted mask for output path {output_path}")
        predicted_mask = (out_mask_logits[0].squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255

        # Save the predicted mask
        saved = save_predicted_mask(predicted_mask, output_path)
        if saved:
            print(f"Predicted mask saved successfully at {output_path}")
        else:
            print(f"Failed to save predicted mask at {output_path}")
    else:
        print("No valid mask output from SAM2 model inference.")

