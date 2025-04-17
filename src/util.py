'''
utility functions are defined in this file
'''
import cv2
from PIL import Image
import pandas as pd

import albumentations as A
import numpy as np
import os
from tqdm import tqdm

def create_augmentation_pipeline():
    """Create an augmentation pipeline suitable for captcha images"""
    return A.Compose([
        # Color-based augmentations
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=20, p=0.5),
        ], p=0.5),
        
        # Noise and blur
        A.OneOf([
            # A.GaussNoise(std_range=(0.2, 0.5), p=0.5),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.5),
            A.GaussianBlur(blur_limit=(1, 3), p=0.5),
        ], p=0.5),
        
        # Image quality
        A.OneOf([
            A.ImageCompression(quality_range = (95,100), p=0.5),
            A.Posterize(num_bits=[4, 6], p=0.5),
        ], p=0.5),
    ])

def preprocess_captcha(image_path):
    """
    Preprocess a captcha image by converting to grayscale, thresholding, and cropping to text area.
    
    Args:
        image_path (str): Path to the input captcha image file
        
    Returns:
        PIL.Image: Processed image containing the cropped captcha text
    """
    # Read image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Binary threshold to separate text from background
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV) # + cv2.THRESH_OTSU # cv2.THRESH_BINARY_INV + + cv2.THRESH_OTSU
    
    # Find contours to detect text area
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find bounding box that contains all text
        x_min = float('inf')
        x_max = 0
        y_min = float('inf')
        y_max = 0
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
        
        # Add padding
        padding = 2
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(img.shape[1], x_max + padding)
        y_max = min(img.shape[0], y_max + padding)
        
        # Crop image to text area
        cropped = img[y_min:y_max, x_min:x_max]
        
        # Convert cropped image to grayscale
        cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        
        # Apply final threshold to make background white and text black
        _, binary_clean = cv2.threshold(cropped_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)



        
        # Convert to RGB (black text on white background)
        clean_rgb = cv2.cvtColor(binary_clean, cv2.COLOR_GRAY2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(clean_rgb)
    
    return pil_image


def augment_captcha(input_data, img_dir, label_dir, num_augmentations = 3): 
    """Augment images in input_data and return new combined dataset
    
    Args:
        input_data: containing original data before augmentation
        img_dir: folder to write the augmented images
        label_dir: folder to write the augmented images' labels
        num_augmentations: Number of augmentations to create per image (default: 3)
    """
    augmentor = create_augmentation_pipeline()
    augmented_data = []
    
    # Create output directory if it doesn't exist
    os.makedirs(img_dir, exist_ok=True)
    
    # Keep track of the largest key number to continue numbering
    max_idx = input_data["key"].astype(int).max()
    next_idx = max_idx + 1
    
    # Create augmentations for each image
    for _, data in tqdm(input_data.iterrows(), desc="Augmenting images"):
        image_path = data['input_file']
        label = data['ground_truth']
        key = data['key']
        
        # Read image with OpenCV
        image = cv2.imread(image_path)
        
        # Create multiple augmentations for each image
        for i in range(num_augmentations):
            # Apply augmentation
            augmented = augmentor(image=image)['image']
            
            # Save augmented image to new directory
            aug_filename = f"input{next_idx:02d}.jpg"
            aug_path = os.path.join(img_dir, aug_filename)
            cv2.imwrite(aug_path, augmented)

            # Save ground truth to text file
            label_filename = f"outputaug{next_idx:02d}.txt"
            label_path = os.path.join(label_dir, label_filename)
            with open(label_path, 'w') as f:
                f.write(str(label))
            
            # Add to augmented dataset

            augmented_data.append({
                'key' : next_idx,
                'input_file': aug_path,
                'ground_truth': label,
                'augmented': True,
                'original_key': key
            })
            next_idx += 1
    
    # Combine original and augmented data
    #final_data = {**input_data, **augmented_data}

    augmented_df = pd.DataFrame(augmented_data)
    
    # Add augmented column to original DataFrame if it doesn't exist
    if 'augmented' not in input_data.columns:
        input_data['augmented'] = False
    
    # Combine original and augmented data
    final_df = pd.concat([input_data, augmented_df], ignore_index=True)
    
    print(f"Dataset size increased from {len(input_data)} to {len(final_df)} images")
    print(f"Augmented images saved to: {img_dir}")
    return final_df