"""
Q3: Model Training & Supervised Evaluation [5 Marks]
- Train a CNN model (ResNet18) for land-use classification
- Evaluate using accuracy and F1-score
- Display a confusion matrix and briefly interpret the results
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "Shapefile of the Delhi-NCR region (EPSG4326)")
RGB_DIR = os.path.join(DATA_DIR, "rgb")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
TRAIN_CSV = os.path.join(OUTPUT_DIR, "q2_train.csv")
TEST_CSV = os.path.join(OUTPUT_DIR, "q2_test.csv")

# ── Config ─────────────────────────────────────────────────────────────────────
BATCH_SIZE = 64
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── Label encoding ─────────────────────────────────────────────────────────────
CLASSES = sorted(["Built-up", "Cropland", "Vegetation", "Water"])
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}
NUM_CLASSES = len(CLASSES)
print(f"Classes ({NUM_CLASSES}): {CLASSES}")

# ── Dataset ────────────────────────────────────────────────────────────────────
class LandCoverDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        # Encode labels
        self.df["label_idx"] = self.df["label"].map(CLASS_TO_IDX)
        # Drop rows with unmapped labels (e.g., 'Others' if not in CLASSES)
        self.df = self.df.dropna(subset=["label_idx"]).reset_index(drop=True)
        self.df["label_idx"] = self.df["label_idx"].astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = row["label_idx"]
        return image, label


# ── Transforms ─────────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── DataLoaders ────────────────────────────────────────────────────────────────
print("\nLoading datasets...")
train_dataset = LandCoverDataset(TRAIN_CSV, RGB_DIR, transform=train_transform)
test_dataset = LandCoverDataset(TEST_CSV, RGB_DIR, transform=test_transform)
print(f"  Train samples: {len(train_dataset)}")
print(f"  Test samples:  {len(test_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── Model: ResNet18 ───────────────────────────────────────────────────────────
print("\nBuilding ResNet18 model...")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# Freeze early layers for faster training
for param in model.parameters():
    param.requires_grad = False
# Unfreeze layer4 and fc
for param in model.layer4.parameters():
    param.requires_grad = True
# Replace final FC layer
model.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, NUM_CLASSES)
)
model = model.to(DEVICE)

# Class weights to handle imbalance
train_labels = train_dataset.df["label_idx"].values
class_counts = np.bincount(train_labels, minlength=NUM_CLASSES).astype(float)
class_counts[class_counts == 0] = 1  # avoid division by zero
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * NUM_CLASSES
class_weights_tensor = torch.FloatTensor(class_weights).to(DEVICE)
print(f"  Class weights: {dict(zip(CLASSES, class_weights.round(3)))}")

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ── Training ───────────────────────────────────────────────────────────────────
print(f"\nTraining for {NUM_EPOCHS} epochs...")
train_losses = []
train_accs = []

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    scheduler.step()
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)
    print(f"  Epoch [{epoch+1:2d}/{NUM_EPOCHS}] Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

# ── Evaluation ─────────────────────────────────────────────────────────────────
print("\nEvaluating on test set...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Metrics
accuracy = accuracy_score(all_labels, all_preds)
f1_macro = f1_score(all_labels, all_preds, average="macro")
f1_weighted = f1_score(all_labels, all_preds, average="weighted")

print(f"\n{'='*50}")
print(f"  TEST RESULTS")
print(f"{'='*50}")
print(f"  Accuracy:         {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"  F1-Score (macro): {f1_macro:.4f}")
print(f"  F1-Score (weighted): {f1_weighted:.4f}")
print(f"{'='*50}")

# Classification report
print("\nClassification Report:")
report = classification_report(all_labels, all_preds, target_names=CLASSES)
print(report)

# ── Confusion Matrix ───────────────────────────────────────────────────────────
print("Generating confusion matrix...")
cm = confusion_matrix(all_labels, all_preds)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Raw counts
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[0])
axes[0].set_title("Confusion Matrix (Counts)", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Predicted", fontsize=11)
axes[0].set_ylabel("Actual", fontsize=11)

# Normalized
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
cm_norm = np.nan_to_num(cm_norm)
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES, ax=axes[1])
axes[1].set_title("Confusion Matrix (Normalized)", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Predicted", fontsize=11)
axes[1].set_ylabel("Actual", fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "q3_confusion_matrix.png"), dpi=150, bbox_inches="tight")
print(f"  Confusion matrix saved to outputs/q3_confusion_matrix.png")

# ── Training curves ────────────────────────────────────────────────────────────
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(range(1, NUM_EPOCHS + 1), train_losses, "b-o", markersize=4)
ax1.set_title("Training Loss", fontsize=13, fontweight="bold")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.grid(True, alpha=0.3)

ax2.plot(range(1, NUM_EPOCHS + 1), train_accs, "g-o", markersize=4)
ax2.set_title("Training Accuracy", fontsize=13, fontweight="bold")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "q3_training_curves.png"), dpi=150, bbox_inches="tight")
print(f"  Training curves saved to outputs/q3_training_curves.png")

# ── Save results to text file ──────────────────────────────────────────────────
results_path = os.path.join(OUTPUT_DIR, "q3_results.txt")
with open(results_path, "w") as f:
    f.write("Q3: Model Training & Supervised Evaluation Results\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Model: ResNet18 (pretrained, fine-tuned layer4 + fc)\n")
    f.write(f"Epochs: {NUM_EPOCHS}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Learning Rate: {LEARNING_RATE}\n")
    f.write(f"Device: {DEVICE}\n\n")
    f.write(f"Train samples: {len(train_dataset)}\n")
    f.write(f"Test samples:  {len(test_dataset)}\n\n")
    f.write(f"Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)\n")
    f.write(f"F1-Score (macro): {f1_macro:.4f}\n")
    f.write(f"F1-Score (weighted): {f1_weighted:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(report + "\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n\n")
    f.write("Interpretation:\n")
    f.write("- The model uses ResNet18 pretrained on ImageNet, fine-tuned on our\n")
    f.write("  Sentinel-2 Delhi-NCR satellite imagery for land-use classification.\n")
    f.write("- Class imbalance (Cropland dominant) is addressed via weighted loss.\n")
    f.write("- Cropland and Built-up classes are expected to have higher accuracy\n")
    f.write("  due to larger sample sizes and distinct spectral signatures.\n")
    f.write("- Vegetation and Water have fewer samples, which may lead to lower\n")
    f.write("  recall for these minority classes.\n")
    f.write("- The confusion matrix reveals any systematic misclassifications\n")
    f.write("  between similar land-cover types.\n")

print(f"  Results saved to outputs/q3_results.txt")

plt.close("all")
print("\n[DONE] Q3 complete!")
