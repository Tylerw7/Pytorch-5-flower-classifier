# 🌸 Flower Image Classifier (ResNet50 Transfer Learning)

![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?logo=pytorch)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![TorchVision](https://img.shields.io/badge/TorchVision-Computer%20Vision-green?logo=pytorch)
![Model](https://img.shields.io/badge/Model-ResNet50-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-90.98%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Project Overview

This project is a **deep learning-based flower classification system** built using **PyTorch** and **Transfer Learning** with a pretrained **ResNet50** model.

The classifier is trained to recognize **five flower species**:  

- 🌼 Daisy  
- 🌻 Sunflower  
- 🌹 Rose  
- 🌷 Tulip  
- 🌱 Dandelion  

Using advanced image augmentation and fine-tuning techniques, the model achieved:

✅ **90.98% validation accuracy** on the test dataset.

---

## 🚀 Key Features

- Transfer Learning using **ResNet50 pretrained on ImageNet**
- Custom fully-connected classification head
- Strong data augmentation pipeline for better generalization
- Dataset splitting automation (train/test)
- Layer freezing + selective fine-tuning
- Model checkpoint saving after every epoch
- Single-image prediction support

---

## 🧠 Model Architecture

The project uses a pretrained **ResNet50 backbone**, replacing the final classification layer with a custom head:

```python
model.fc = nn.Sequential(
    nn.Linear(2048, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 5)
)
