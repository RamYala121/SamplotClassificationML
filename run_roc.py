import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_curve, roc_auc_score
from torchvision import transforms

# =====================================================================
# 1. SETUP & PATHS
# =====================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import the 'Net' class from your net.py file
from net import Net

model = Net()

# Automatically find your saved weights (.pth file) in the folder
pth_files = [f for f in os.listdir('.') if f.endswith('.pth')]
if pth_files:
    print(f"Found weights file: {pth_files[0]}")
    model.load_state_dict(torch.load(pth_files[0], map_location=device))
else:
    print("Warning: No .pth file found in this folder. Make sure your model weights are in this directory!")

model.to(device)
model.eval()

# Image transformations matching your model's input size
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =====================================================================
# 2. GATHER IMAGES FROM THE TRAIN DIRECTORY
# =====================================================================
data_dir = "train" 
positive_dir = os.path.join(data_dir, "positive")
negative_dir = os.path.join(data_dir, "negative")

dataset = []

# Gather Positive cases (Label = 1)
if os.path.exists(positive_dir):
    for f in os.listdir(positive_dir):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            dataset.append({"file": os.path.join(positive_dir, f), "label": 1})

# Gather Negative cases (Label = 0)
if os.path.exists(negative_dir):
    for f in os.listdir(negative_dir):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            dataset.append({"file": os.path.join(negative_dir, f), "label": 0})

# =====================================================================
# 3. RUN PREDICTIONS
# =====================================================================
y_true = []
y_score = []

print(f"Running model predictions on {len(dataset)} images found in '{data_dir}' folder...")

for item in dataset:
    file_path = item["file"]
    label = item["label"]
    
    try:
        image = Image.open(file_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            
            # Convert multiclass logits into probabilities using Softmax
            probabilities = F.softmax(outputs, dim=1)
            
            # Grab the probability of class 1 (Positive / Gene Fusion)
            prob_positive = probabilities[0][1].item()
            
        y_score.append(prob_positive)
        y_true.append(label)
    except Exception as e:
        print(f"Skipping broken or unreadable image {file_path}: {e}")

# =====================================================================
# 4. COMPUTE & PLOT ROC CURVE
# =====================================================================
if len(y_true) > 0:
    # Save the exact TSV format your mentor requested
    df = pd.DataFrame({"score": y_score, "label": y_true})
    df.to_csv("samplot_scores.tsv", sep="\t", index=False, header=False)
    print("Saved evaluation data to 'samplot_scores.tsv'")

    # Calculate ROC metrics
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    # Plotting setup
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"Net Model (auROC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Baseline (0.50)")
    
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Gene Fusion Classification ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    output_image = "samplot_roc_curve.png"
    plt.savefig(output_image, dpi=300)
    print(f"Success! ROC Curve saved as '{output_image}'")
else:
    print("Error: No images were successfully processed. Check your paths!")