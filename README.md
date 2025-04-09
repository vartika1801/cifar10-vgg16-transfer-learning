# 🧠 CIFAR-10 Image Classification using Transfer Learning (VGG16)

This project implements image classification on the CIFAR-10 dataset using **transfer learning** with a pre-trained **VGG16** model. The goal is to leverage powerful deep features from VGG16 (trained on ImageNet) and adapt it to classify small 32x32 images into 10 categories.

---

## 📚 Dataset: CIFAR-10
CIFAR-10 is a widely used dataset for image classification, containing **60,000 32x32 color images** in **10 classes**, with 6,000 images per class.

**Classes**:
- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

---

## 🔧 Model Architecture

- **Base Model**: VGG16 (pre-trained on ImageNet)
- **Transfer Learning Strategy**:
  - All convolutional layers **frozen**
  - Custom **fully connected classifier head** added on top
- **Classifier Head**:
  - Global Average Pooling
  - Dense (512 units, ReLU)
  - Dropout (0.5)
  - Dense (10 units, Softmax)

---

## 🛠️ Project Setup

### 1. Clone the Repository
git clone https://github.com/your-username/cifar10-vgg16-transfer-learning.git
cd cifar10-vgg16-transfer-learning

### 2. Install Requirements
pip install -r requirements.txt

### 3. Run Training
python train.py

### Future Improvements
1. Fine-tune deeper VGG layers
2. Try alternative models: ResNet50, MobileNet, EfficientNet
3. Deploy with Streamlit or Flask for real-time predictions
4. Add Grad-CAM visualizations to interpret predictions
