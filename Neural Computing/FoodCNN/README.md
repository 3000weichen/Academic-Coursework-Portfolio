# FoodCNN – Neural Computing Assignment (10/10)

This project implements a **custom CNN (FoodCNN)** for multi-class food image classification under strict computational constraints.  
It was developed as the final assignment for the *Neural Computing* course at Leiden University and received a **10 / 10** grade.

---

## Files

- `FoodCNN.ipynb` – main notebook with data loading, model, training loop and evaluation
- `report_foodcnn.pdf` – full assignment report with detailed analysis
- `requirements.txt` – Python dependencies
- (local only) `best_model.pth` – saved best model weights, **ignored in Git** for size reasons

---

## Task & Dataset

The goal is to classify food images into **91 classes**.

- Data is provided in `train/` and `test/` folders, each containing 91 subdirectories (one per class).
- Images are resized to **64×64** to reduce GPU memory usage while preserving enough detail.
- Pixel values are normalized to `[0, 1]`.

We split the original training set into:

- **Train set**: 90%  
- **Validation set**: 10%

using `train_test_split(..., random_state=1)` for reproducibility.:contentReference[oaicite:2]{index=2}

---

## Data Augmentation

To combat overfitting and improve generalization, we use an extensive augmentation pipeline on the training set:​:contentReference[oaicite:3]{index=3}  

- `RandomHorizontalFlip`
- `RandomRotation(±30°)`
- `RandomResizedCrop(scale=(0.6, 1.0))`
- `ColorJitter` (brightness & contrast)
- `RandomGrayscale(p=0.1)`
- `RandomAffine`
- `RandomErasing(p=0.25)`
- `Normalize(mean=[0.5]*3, std=[0.5]*3)`

Validation & test sets only use normalization to keep evaluation deterministic.

Batch size is set to **32** with `num_workers=0` to ensure reproducibility and low memory usage.

---

## Model Architecture – FoodCNN

FoodCNN is a residual CNN inspired by ResNet:​:contentReference[oaicite:4]{index=4}  

- 4 main blocks (`Block1`–`Block4`)
- Each block contains **two ResidualBlocks** with:
  - Conv → BatchNorm → ReLU
  - Strided convolutions (`stride=2`) to downsample feature maps
- Skip connections to ease gradient flow and mitigate vanishing gradients
- Final head:
  - Adaptive Average Pooling
  - Flatten
  - Fully connected layer → 91 output classes

This design allows the model to gradually reduce spatial resolution while increasing channel depth, capturing higher-level semantic features.

---

## Training Strategy

- Optimizer: **Adam**
- Initial learning rate: `1e-4`
- Scheduler: `ReduceLROnPlateau` (factor = 0.3, patience = 2, `min_lr=1e-7`)
- Loss: Cross-Entropy
- Early stopping: stop if validation accuracy does not improve for **10 epochs**
- Max epochs: 100
- Deterministic behavior:
  - Fixed seeds for Python, NumPy, PyTorch
  - `torch.backends.cudnn.deterministic = True`

The notebook implements:

- `train()` – training loop per epoch
- `validate()` – evaluation loop on validation set
- Checkpoint saving for the best validation accuracy (`best_model.pth`)
- Progress visualization using `tqdm`:contentReference[oaicite:5]{index=5}  

---

## Results

Final test performance:​:contentReference[oaicite:6]{index=6}  

- **Test accuracy**: **61.06%**
- **Test loss**: 1.57

This shows that the model generalizes reasonably well on a **91-class** problem given memory and compute constraints. Validation and test performance are comparable, indicating stable learning.

---

## Observations & Future Work

Key lessons:

- Strong data augmentation + LR scheduling helps reduce overfitting.
- Adam with `ReduceLROnPlateau` provides stable convergence under limited compute.
- Residual connections significantly help training deeper CNNs from scratch.

Potential improvements:​:contentReference[oaicite:7]{index=7}  

- Add dropout with systematic hyperparameter search (e.g., Bayesian optimization)
- Explore deeper ResNet variants (e.g., ResNet-50) with transfer learning
- Semi-supervised learning to leverage unlabeled data

---

## Authors

- **Hao Chen** (s3990788)  
- **Simone de Vos Burchart** (s1746995)

Course: **Neural Computing (2024–2025)** – Leiden University  
Final grade for this assignment: **10 / 10**
