import argparse
import torch
from captcha_trocr import CaptchaTrOCR

def parse_args():
    parser = argparse.ArgumentParser(description='Captcha OCR using TrOCR')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--input', type=str, default='',
                        help='Input image file or directory path')
    parser.add_argument('--output', type=str, default='',
                        help='Output directory path for predictions')
    parser.add_argument('--labels', type=str, default='',
                        help='Directory containing ground truth label files')

    return parser.parse_args()

def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    
    # Initialize CaptchaTrOCR with device and config
    captcha_model = CaptchaTrOCR(device=device, config_path=args.config)
    
    # Run inference
    predictions = captcha_model(im_path=args.input, save_path=args.output)
    
    # Add ground truth labels if directory provided
    if args.labels:
        results = captcha_model.add_gt_labelbels(predictions, label_dir=args.labels)
        # Print WER if labels were provided
        accuracy = results['correct'].mean() * 100
        print(f"\nWord level accuracy: {accuracy:.2f}%")
        print(f"Correct predictions: {results['correct'].sum()}/{len(results)}")

if __name__ == '__main__':
    main()