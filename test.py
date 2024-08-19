import argparse
import os
import time


# Parse command-line arguments
parser = argparse.ArgumentParser(description='YOLOv8 Model Testing')
parser.add_argument('test_data_path', type=str, help='Path to the test data folder')
parser.add_argument('trained_model_path', type=str, help='Path to the trained model file (e.g., best.pt)')
parser.add_argument('result_path', type=str, help='Path to save the classification results')
args = parser.parse_args()


import torch
from ultralytics import YOLO
model = YOLO(args.trained_model_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


results = model.predict(source=args.test_data_path,save=False, save_txt=False, save_conf=False, save_crop=False)

count=1
with open(os.path.join(args.result_path, 'results.txt'), 'w') as f:
    for i,result in enumerate(results):
        label=result.names
        prob=result.probs.data
        
        output_line = f"image {i+1}/{len(results)} "
        for i, label in enumerate(label.values()):
            output_line += f"{label} {prob[i]:.2f}, "
        output_line = output_line[:-2] 
        count+=1
        f.write(output_line+"\n")
        

print(f'Classification results saved to {os.path.join(args.result_path, "results.txt")}')
