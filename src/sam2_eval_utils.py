import os
import cv2
import numpy as np
import random
import torch
import gc
import pandas as pd
from sam2.build_sam import build_sam2_video_predictor


device = "cpu"

# Set base_dir to the directory containing this script (SAM2_Eval_Surgical_Datasets)
base_dir = os.getcwd()

# Move up one level to reach SAM2_Eval directory
root_dir = os.path.abspath(os.path.join(base_dir, ".."))

# Define paths to model configuration and checkpoint files relative to SAM2_Eval
model_cfg = os.path.join(root_dir, "SAM2_model_and_cfg", "sam2_hiera_l.yaml")
checkpoint_path = os.path.join(root_dir, "SAM2_model_and_cfg", "sam2_hiera_large.pt")

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
        tuple: A tuple containing two lists:
               - A list of selected prompt coordinates in the format [[x, y], [x, y], ...].
               - A list of labels (1 for each selected point).
    """
    # Find all coordinates in the mask where the value is 255
    coords = np.column_stack(np.where(mask == 255))
    print(f"Total prompt points found in mask: {len(coords)}")

    # If no valid points are found, return None
    if coords.size == 0:
        print("No valid points with value 255 found in the mask.")
        return None, None

    # If the number of points requested is greater than available points, use all points
    if num_points >= len(coords):
        print(f"Requested {num_points} points, but only {len(coords)} available. Returning all points.")
        selected_points = coords
    else:
        # Randomly select the specified number of points
        selected_points = random.sample(coords.tolist(), num_points)
    
    # Format points as [[x, y], [x, y], ...] where x and y are in the correct order
    formatted_points = [[point[1], point[0]] for point in selected_points]
    print(f"Formatted points: {formatted_points}")

    # Create a list of labels, with a label of 1 for each selected point
    labels = [1] * len(formatted_points)

    return formatted_points, labels

def save_predicted_mask(predicted_mask, predicted_mask_path):
    """Save the predicted mask to disk."""
    predicted_mask_bgr = cv2.cvtColor(predicted_mask, cv2.COLOR_GRAY2BGR)
    success = cv2.imwrite(predicted_mask_path, predicted_mask_bgr)
    if success:
        print(f"Predicted mask successfully saved at: {predicted_mask_path}")
    else:
        print(f"Failed to save the predicted mask at: {predicted_mask_path}")

def process_image(sam_model, inference_state, image, mask, points, labels, output_path, idx):
    """Run the SAM2 inference on a single image."""
    print(f"Starting inference on image index {idx}")
    
    # Set up inference parameters
    obj_id = idx + 1
    frame_idx = 0

    # Perform inference with the selected prompt point
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


# Function to calculate metrics between predicted and ground truth masks
def calculate_metrics(predicted_mask, ground_truth):
    predicted_mask_bin = (predicted_mask > 127).astype(np.uint8)
    ground_truth_bin = (ground_truth > 127).astype(np.uint8)

    intersection = np.logical_and(predicted_mask_bin, ground_truth_bin).sum()
    union = np.logical_or(predicted_mask_bin, ground_truth_bin).sum()
    true_positive = intersection
    false_positive = np.logical_and(predicted_mask_bin, np.logical_not(ground_truth_bin)).sum()
    false_negative = np.logical_and(np.logical_not(predicted_mask_bin), ground_truth_bin).sum()

    iou = intersection / union if union > 0 else 0
    dice = (2 * intersection) / (predicted_mask_bin.sum() + ground_truth_bin.sum()) if (predicted_mask_bin.sum() + ground_truth_bin.sum()) > 0 else 0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

    return iou, dice, precision, recall

# Function to evaluate masks for a single organ
def evaluate_organ(predicted_masks_path, ground_truth_masks_path):
    results = []
    
    for file in os.listdir(predicted_masks_path):
        if file.endswith(".png"):
            predicted_mask_file = os.path.join(predicted_masks_path, file)
            ground_truth_file = os.path.join(ground_truth_masks_path, file)
            
            if not os.path.exists(ground_truth_file):
                print(f"Warning: Ground truth mask missing for {file}")
                continue
            
            predicted_mask = cv2.imread(predicted_mask_file, cv2.IMREAD_GRAYSCALE)
            ground_truth = cv2.imread(ground_truth_file, cv2.IMREAD_GRAYSCALE)

            if predicted_mask is None or ground_truth is None:
                print(f"Error loading predicted or ground truth mask for {file}")
                continue

            iou, dice, precision, recall = calculate_metrics(predicted_mask, ground_truth)
            
            results.append({
                "File": file,
                "IoU": iou,
                "Dice": dice,
                "Precision": precision,
                "Recall": recall
            })

    return results
