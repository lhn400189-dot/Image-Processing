"""
Image Reasoning Test
Support testing individual images or randomly selecting test set images
"""
import os
import sys
import cv2
import numpy as np
import random
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict, Any, Optional

#====Configure Region=====
#Default model path
DEFAULT_MODEL = "yolo11n-obb.pt"
#List of image paths to be tested (if empty, randomly selected from the test set)
TEST_IMAGES = [
#Add the image path to be tested here, for example:
    # "data/test/images/P0006.png",
    # "data/val/images/P0003.png"
]
#If no image is specified, randomly select the number of images
NUM_RANDOM_IMAGES = 3
#Test set path
TEST_DIR = "data/test/images"
#Verification set path (alternative)
VAL_DIR = "data/val/images"
#Output directory
OUTPUT_DIR = "detection_results"
# ===================

def find_images(directory: str) -> List[str]:
#Search for all image files in the directory
    if not os.path.exists(directory):
        return []
    return [
        os.path.join(directory, f) 
        for f in os.listdir(directory) 
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ]

class YOLOInference:
    def __init__(self, model_path: str = DEFAULT_MODEL):
        """Initialize YOLO Model"""
        print(f" load model: {model_path}")
        self.model = YOLO(model_path)
        print(" Model loading completed")
        print(f" Model Information:")
        print(f"   - Number of categories: {len(self.model.names)}")
        print(f"   - Supported categories: {list(self.model.names.values())}")
    
    def process_images(self, image_paths: List[str], output_dir: str = OUTPUT_DIR) -> List[Dict[str, Any]]:
        """Processing multiple images"""
        results = []
        for img_path in image_paths:
            try:
                result = self._process_single_image(img_path, output_dir)
                results.append(result)
            except Exception as e:
                print(f" process images {os.path.basename(img_path)} Error occurred during: {str(e)}")
        return results
    
    def _process_single_image(self, image_path: str, output_dir: str) -> Dict[str, Any]:
        """Processing a single image"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"The image does not exist: {image_path}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Unable to read image: {image_path}")
        
        print(f"\n Processing: {os.path.basename(image_path)}")
        print(f"   - image path: {image_path}")
        print(f"   - image size: {img.shape[1]}x{img.shape[0]}")
        
        # reasoning
        results = self.model(img, conf=0.25, iou=0.7)
        
        # Processing result
        result_info = {
            'image_path': image_path,
            'detections': 0,
            'class_counts': {},
            'output_path': ''
        }
        
        for i, result in enumerate(results):
            # Draw Results
            result_img = result.plot(
                line_width=2,
                font_size=12,
                conf=True,
                labels=True,
                boxes=True
            )
            
            # Save result image
            output_path = os.path.join(output_dir, f"detection_{os.path.basename(image_path)}")
            cv2.imwrite(output_path, result_img)
            result_info['output_path'] = output_path
            
            # Statistical testing results
            if result.obb is not None:
                num_detections = len(result.obb)
                result_info['detections'] = num_detections
                print(f" Detected {num_detections} goals")
                
                if num_detections > 0:
                    classes = result.obb.cls.cpu().numpy()
                    class_names = [self.model.names[int(c)] for c in classes]
                    from collections import Counter
                    class_counts = Counter(class_names)
                    result_info['class_counts'] = dict(class_counts)
                    
                    # Print category statistics
                    for cls_name, count in class_counts.items():
                        print(f"   - {cls_name}: {count}个")
            else:
                print(" Target not detected")
        
        return result_info

def main():
    """main"""
    print("=" * 60)
    print(" YOLO11-OBB Image reasoning test")
    print("=" * 60)
    
    # Analyze command-line parameters
    model_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    
    try:
        # Obtain the image to be tested
        if TEST_IMAGES:
            print(f"\n Using configuration {len(TEST_IMAGES)} Test image")
            image_paths = TEST_IMAGES
        else:
            print(f"\n Randomly select from the test set {NUM_RANDOM_IMAGES} pictures")
            # Attempt to obtain images from the test set
            test_images = find_images(TEST_DIR)
            if not test_images:
                print(f" Test set directory  {TEST_DIR} No image found in, try using the validation set...")
                test_images = find_images(VAL_DIR)
                if not test_images:
                    raise FileNotFoundError(f"No test images were found, please check {TEST_DIR} or {VAL_DIR} content")
            
            random.seed(42)  
            image_paths = random.sample(test_images, min(NUM_RANDOM_IMAGES, len(test_images)))
        
        # Initialize the model
        detector = YOLOInference(model_path)
        
        # process images
        results = detector.process_images(image_paths)
        
        # Print summary information
        print("\n" + "=" * 60)
        print(" Detection completed!")
        print(" Summary results:")
        
        total_detections = 0
        total_images = len(results)
        class_totals = {}
        
        for result in results:
            total_detections += result['detections']
            for cls_name, count in result['class_counts'].items():
                class_totals[cls_name] = class_totals.get(cls_name, 0) + count
        
        print(f"   - Number of processed images: {total_images} ")
        print(f"   - Total number of targets detected: {total_detections} items")
        
        if class_totals:
            print("   - Category statistics:")
            for cls, count in sorted(class_totals.items()):
                print(f"      - {cls}: {count}个")
        
        print(f"\n The test results have been saved to: {os.path.abspath(OUTPUT_DIR)}/")
        
    except Exception as e:
        print(f"\n An error occurred: {str(e)}")
    
    print("\n" + "=" * 60)
    print(" Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
