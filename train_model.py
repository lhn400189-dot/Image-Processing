"""
training scripts
Only output the loss curve and the final recognition result graph
"""
import os
from ultralytics import YOLO
from model_final.gmvae import GMVAE
from model_final import utils
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import Path

def train_yolo11_obb():
    """Train model"""
    print(" start Train model...")
    
    #Load pre trained model
    model = YOLO("yolo11n-obb.pt")
    
    #Training configuration
    results = model.train(
        data="dota_custom.yaml",
        epochs=30,              
        imgsz=1024,              
        batch=4,                
        device=0,               # GPU
        project="yolo11_final",
        name="dota_training",
        save=True,
        plots=False,            
        verbose=False,         
        val=True,
        patience=5,            
        save_period=3          
    )
    
    print("Training completed!")
    return results

def plot_training_curves():
    """Draw Training Curve"""
    print("Generate training curve...")
    
    results_dir = "yolo11_final/dota_training"
    csv_file = os.path.join(results_dir, "results.csv")
    
    if not os.path.exists(csv_file):
        print("ERROR: Training result file not found")
        return
    
    #Read training results
    import pandas as pd
    df = pd.read_csv(csv_file)
    
    #Create Chart
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss Curve
    epochs = df['epoch']
    ax1.plot(epochs, df['train/box_loss'], label='Box Loss', color='blue')
    ax1.plot(epochs, df['train/cls_loss'], label='Cls Loss', color='red')
    ax1.plot(epochs, df['train/dfl_loss'], label='DFL Loss', color='green')
    ax1.set_title('Training Losses')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # mAP Curve
    if 'metrics/mAP50(B)' in df.columns:
        ax2.plot(epochs, df['metrics/mAP50(B)'], label='mAP@0.5', color='purple')
        ax2.plot(epochs, df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', color='orange')
        ax2.set_title('Validation mAP')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('mAP')
        ax2.legend()
        ax2.grid(True)
    
    # learning rate
    if 'lr/pg0' in df.columns:
        ax3.plot(epochs, df['lr/pg0'], label='Learning Rate', color='brown')
        ax3.set_title('Learning Rate')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('LR')
        ax3.legend()
        ax3.grid(True)

    # total Loss
    total_loss = df['train/box_loss'] + df['train/cls_loss'] + df['train/dfl_loss']
    ax4.plot(epochs, total_loss, label='Total Loss', color='black')
    ax4.set_title('Total Training Loss')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("The training curve has been saved: training_curves.png")

def test_inference_and_visualize():
    """Test reasoning and visualize results"""
    print("Test model inference...")
    
    #Load the trained model
    model_path = "yolo11_final/dota_training/weights/best.pt"
    if not os.path.exists(model_path):
        print("ERROR:No trained model found, use pre trained model")
        model_path = "yolo11n-obb.pt"
        model1 = GMVAE()    

    
    model = YOLO(model_path)
    #Search for test images (prioritize using the test dataset)
    test_dir = "./data/test/images"
    test_images = []
    
    if os.path.exists(test_dir):
        all_images = [f for f in os.listdir(test_dir) 
                     if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        import random
        random.seed(42)  
        test_images = [os.path.join(test_dir, img) for img in random.sample(all_images, min(5, len(all_images)))]
    
    if not test_images:
        print("WARNING:Test set image not found, validation set will be used...")
        test_dir = "./data/val/images"
        if os.path.exists(test_dir):
            all_images = [f for f in os.listdir(test_dir) 
                         if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            test_images = [os.path.join(test_dir, img) for img in random.sample(all_images, min(5, len(all_images)))]
    
    if not test_images:
        print("ERROR:Test image not found")
        return
    
    results_dir = "detection_results"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"for {len(test_images)} Image detection...")
    
    for i, img_path in enumerate(test_images):
        print(f" process image {i+1}/{len(test_images)}: {os.path.basename(img_path)}")
        
        # reasoning
        results = model(img_path, conf=0.25, iou=0.7)
        
        # Save results
        for j, result in enumerate(results):
            result_img = result.plot(
                line_width=2,
                font_size=12,
                conf=True,
                labels=True,
                boxes=True
            )
            
            output_path = os.path.join(results_dir, f"detection_{i+1}_{os.path.basename(img_path)}")
            cv2.imwrite(output_path, result_img)
            
            if result.obb is not None:
                num_detections = len(result.obb)
                print(f"    GOOD!s Detected {num_detections} goal")
                
                if num_detections > 0:
                    classes = result.obb.cls.cpu().numpy()
                    class_names = [model.names[int(c)] for c in classes]
                    from collections import Counter
                    class_counts = Counter(class_names)
                    for cls_name, count in class_counts.items():
                        print(f"      - {cls_name}: {count}个")
            else:
                print("    ERROR:Target not detected")
    
    print(f"The test results have been saved to: {results_dir}/")

def main():
    """main"""
    print("=" * 60)
    print("YOLO11-OBB training process")
    print("=" * 60)
    
    # 1. Train the model
    print("\n step 1: Training YOLO11-OBB model...")
    train_results = train_yolo11_obb()
    
    # 2. Plot the training curve
    print("\n step 2: Generate training curve...")
    plot_training_curves()
    
    # 3. Testing reasoning
    print("\n step 3: Test model inference...")
    test_inference_and_visualize()
    
    print("\n" + "=" * 60)
    print("all steps compelete！")
    print("check training_curves.png Understand the training process")
    print("check detection_results/ Understand the detection effect")
    print("=" * 60)

if __name__ == "__main__":
    main()
