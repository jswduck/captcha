"""
This module provides a Captcha class for recognizing text in captcha images.

The Captcha class includes methods for loading a model, preprocessing images,
and fine-tuning the model on training data.
"""

from typing import List

import torch
import torch.nn as nn
from torchvision import transforms
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import pytesseract

import yaml
import os
import re
import pandas as pd
from PIL import Image


import numpy as np
import cv2
import util

class Captcha(object):
    """
    A class for performing OCR on captcha images using pre-trained TrOCR model.

    This class handles loading the model, preprocessing images, and generating text
    predictions for captcha images.
    """
    def __init__(self, device, config_path:str="config.yaml"):
        # Load configurations
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Get model configuration
        model_config = self.config['model']
        model_path = model_config['pretrained_path']
        self.preprocessing = model_config['preprocessing']
        print (f"Preprocess: {self.preprocessing}")
        
        # Load model and processor
        self.processor = TrOCRProcessor.from_pretrained(model_path)
        model = VisionEncoderDecoderModel.from_pretrained(model_path)
        self.model = model.to(device)
        
        # Store training configurations
        self.batch_size = self.config['training']['batch_size']
        
        # Store data paths
        data_config = self.config['data']
        self.input_dir = data_config['raw_dir']
        self.output_dir = data_config['output_dir']
        self.finetune_eval_dir = data_config['finetune_eval_dir']
        self.aug_dir = data_config['aug_dir']
        
        # Set model configurations
        self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size
        
        # Set beam search parameters
        if 'beam_search' in model_config:
            beam_config = model_config['beam_search']
            self.model.config.max_length = beam_config.get('max_length', 64)
            self.model.config.early_stopping = beam_config.get('early_stopping', True)
            self.model.config.no_repeat_ngram_size = beam_config.get('no_repeat_ngram_size', 3)
            self.model.config.length_penalty = beam_config.get('length_penalty', 2.0)
            self.model.config.num_beams = beam_config.get('num_beams', 4)
    
    def __call__(self, im_path = "", save_path = ""):

        """
        Perform inference on a single image or batch process all images in a directory.
        Load the data first, and then depending on the mode, we call different models to do inference.
        
        Args:
            im_path: Path to either a .jpg image file or a directory containing .jpg images
            save_path: Path to save output. If im_path is a directory, save_path should also be a directory
        
        Returns:
            dict: Generated text for single image or multiple images
        """


        # valid_modes = ["tesseract", "TrOCR"]
        # if mode not in valid_modes:
        #     print(f"Error: mode must be one of {valid_modes}")
        #     return pd.DataFrame()

        ## load the captcha images from the folders / file

        # Initialize lists to store data for DataFrame
        keys = []
        input_files = []
        output_files = []
        im_path = self.input_dir if im_path == "" else im_path
        
        if os.path.isfile(im_path):
            if im_path.lower().endswith('.jpg'):
                input_path = os.path.join(self.input_dir, im_path)
                if save_path == "":
                    save_path = os.path.basename(im_path).replace('.jpg', '.txt')
                output_path = os.path.join(self.output_dir, save_path)
                
                match = re.search(r'input(\d+)\.jpg', input_path)
                if match:
                    key = match.group(1)
                    keys.append(key)
                    input_files.append(input_path)
                    output_files.append(output_path)
                else:
                    print(f"Filename {im_path} does not match expected pattern")
                    return pd.DataFrame()

        elif os.path.isdir(im_path):
            for filename in sorted(os.listdir(im_path)):
                if filename.lower().endswith('.jpg'):
                    input_path = os.path.join(im_path, filename)
                    output_filename = filename.replace('input', 'pred').replace('.jpg', '.txt') # output to a file with predxx.txt
                    save_path = self.output_dir if save_path == "" else save_path
                    output_path = os.path.join(save_path, output_filename)
                    
                    match = re.search(r'input(\d+)\.jpg', filename)
                    if match:
                        key = match.group(1)
                        keys.append(key)
                        input_files.append(input_path)
                        output_files.append(output_path)
                    else:
                        print(f"Filename {filename} does not match expected pattern: it has to start with input and followed by a number")
                        continue

        else:
            raise FileNotFoundError(f"Path not found: {im_path}")

        if not keys:
            print(f"No jpg files found in {im_path}")
            return pd.DataFrame()

        # Create DataFrame
        batch_data = pd.DataFrame({
            'key': keys,
            'input_file': input_files,
            'output_file': output_files
        })
        
        # Sort by key
        batch_data = batch_data.sort_values('key')

        ## call inference functions 

    
        print(f"Using TrOCR for inference")
        preds = self.inference_with_TrOCR(batch_data)

        batch_data['prediction'] = preds

        # Write predictions to files
        for _, row in batch_data.iterrows():
            with open(row['output_file'], 'w') as f:
                f.write(row['prediction'])
            print(f"Generated text '{row['prediction']}' saved to {row['output_file']}")

        return batch_data
        

    def inference_with_tesseract(self, batch_data):
        """
        Perform inference on a single image or batch process all images in a directory, with tesseract
        
        Args:
            batch_data: data loaded from the folders / individual file. Stored in a DataFrame
        
        Returns:
            DataFrame: Identified text for the images
        """

        # Set Tesseract config
        os.environ['TESSDATA_PREFIX'] = '/opt/homebrew/share/tessdata'

        pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

        tesseract_results = [pytesseract.image_to_string(Image.open(path), config='--psm 7 --oem 1 -l eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789').strip()[:5] for path in batch_data['input_file']] # --oem 1 --tessdata-dir /opt/homebrew/share/tessdata


        return tesseract_results


    def inference_with_TrOCR(self, batch_data):
        """
        Perform inference on a single image or batch process all images in a directory, with TrOCR
        
        Args:
            batch_data: data loaded from the folders / individual file. Stored in a DataFrame
            
        
        Returns:
            DataFrame: Identified text for the images
        """

        
        
        # Process all images in batch
        if self.preprocessing:
            batch_images = [util.preprocess_captcha(path) for path in batch_data['input_file']]
            print(f"Preprocessing turned on")
        else:
            batch_images = [Image.open(path).convert("RGB") for path in batch_data['input_file']]
            print(f"Preprocessing turned off")

        pixel_values = self.processor(batch_images, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Add predictions to DataFrame
        # batch_data['prediction'] = generated_texts
        
        
        return generated_texts
    

    def add_gt_labels(self, preds:pd.DataFrame, label_dir="") -> pd.DataFrame:
        """
        Load labels from text files and merge with predictions DataFrame.
        
        Args:
            preds: DataFrame containing predictions with 'key' and 'prediction' columns
            label_dir: Directory containing label files. Defaults to self.output_dir if empty
        
        Returns:
            DataFrame with predictions and ground truth labels merged on key
        """
        # Use default directory if none provided
        label_dir = self.output_dir if label_dir == "" else label_dir
        
        # Initialize lists for labels DataFrame
        keys = []
        labels = []
        
        # Read all label files
        for filename in sorted(os.listdir(label_dir)):
            if filename.startswith('output') and filename.endswith('.txt'):
                # Extract key from filename (outputXX.txt)
                match = re.search(r'output(?:aug)?(\d+)\.txt', filename)
                if match:
                    key = match.group(1)
                    # Read label from file
                    with open(os.path.join(label_dir, filename), 'r') as f:
                        label = f.read().strip()
                    keys.append(key)
                    labels.append(label)
        
        # Create labels DataFrame
        labels_df = pd.DataFrame({
            'key': keys,
            'ground_truth': labels
        })
        
        # Merge predictions with labels on key
        merged_df = preds.merge(labels_df, on='key', how='left')
        
        # Calculate if prediction matches ground truth
        merged_df['correct'] = merged_df.apply(
            lambda row: row['prediction'] == row['ground_truth'], axis=1
        )
        
        return merged_df
