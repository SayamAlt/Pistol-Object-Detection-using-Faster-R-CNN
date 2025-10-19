# 🔫 Pistol Object Detection using Faster R-CNN (ResNet50 RPN)

<p align="center">
  <img src="https://img.shields.io/badge/Framework-PyTorch-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Model-Faster%20R--CNN-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Backbone-ResNet50%20FPN-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Deployed%20With-Streamlit%20%7C%20Telegram%20Bot-red?style=for-the-badge"/>
</p>

---

## 🚀 Overview

**Pistol Object Detection using Faster R-CNN with ResNet50 RPN** is an end-to-end deep learning project built to **detect and localize pistols in images** for potential safety and security applications.  
It leverages **state-of-the-art object detection architecture** — *Faster R-CNN with a ResNet50 Feature Pyramid Network (FPN)* — fine-tuned on a custom dataset of pistol images.

The project showcases the **full lifecycle of AI solution development**:

> ⚙️ Model Training → 🔍 Object Detection → 🌐 Streamlit Web App → 🤖 Telegram Bot Deployment

---

## 🎯 Project Highlights

- 🧠 **Deep Learning Model** — Faster R-CNN (ResNet50 backbone) trained for high-accuracy pistol detection  
- 🗃️ **Custom Dataset** — Curated and annotated dataset of pistols in real-world scenarios  
- 🧩 **Transfer Learning** — Efficient adaptation from COCO pre-trained weights  
- 🎨 **Bounding Box Visualization** — Real-time pistol localization with labeled boxes  
- ⚡ **Streamlit Web App** — Interactive image-based detection interface  
- 💬 **Telegram Bot Integration** — Chat-based pistol detection for fast mobile access  
- 📈 **Performance Optimization** — SGD optimizer, StepLR scheduler, data augmentation, and checkpointing  

---

## 🧩 Model Architecture

The model architecture is based on **Faster R-CNN with ResNet50 FPN**, a two-stage object detection pipeline:

1. **Region Proposal Network (RPN)** – Generates potential bounding boxes  
2. **ROI Pooling + Classification Head** – Refines proposals and predicts class labels  

**Architecture Summary:**

- Backbone: `ResNet50 Feature Pyramid Network`
- ROI Head: Custom `FastRCNNPredictor`
- Loss Function: Multi-task loss (Classification + Regression)
- Optimizer: SGD (momentum=0.9)
- Learning Rate Scheduler: StepLR
- Framework: PyTorch (torchvision)

---

## 📦 Dataset & Preprocessing

The dataset contains **images of pistols and backgrounds**, annotated in COCO-like JSON format.

| Component | Details |
|------------|----------|
| Classes | 2 → `['bg', 'Pistol']` |
| Image Size | Variable (resized dynamically) |
| Format | JPG |

**Preprocessing Steps:**
- Image normalization (`ToTensor()`)
- Data augmentation: random flip, scale, color jitter  
- Label encoding for class mapping  
- Batched DataLoader for GPU efficiency  

---

## 🧠 Model Training

```python
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

model = fasterrcnn_resnet50_fpn(weights='COCO_V1')
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
```

---

## 🌐 Streamlit Web App Deployment

A clean, interactive **Streamlit web interface** enables users to upload images and visualize pistol detections in real-time.

### 🧩 Features

- 📤 Upload an image (JPG/PNG)
- ⚙️ Model runs inference with bounding boxes and confidence scores
- 🖼️ Displays uploaded and detected images **side-by-side**
- 🟥 Dynamic **red bounding boxes** and **white labels** for detected pistols
- 🎚️ Adjustable confidence threshold for precision control

### ▶️ Run the App

```bash
streamlit run app.py
```

## 🖥️ App Layout & ⚙️ Tech Stack

### 🧩 App Layout
- **Left:** Uploaded Image  
- **Right:** Pistol Detection Results  

### ⚙️ Tech Stack
- **Frontend:** Streamlit  
- **Backend:** PyTorch  
- **Frameworks:** torchvision, Pillow, numpy  

---

## 🤖 Telegram Bot Deployment

A fully functional **Telegram Bot** allows pistol detection directly from chat — fast, simple, and secure!

### ⚙️ Commands

- `/start` → Welcome and usage info  
- `/help` → Guide on how to use the bot  
- `/about` → Model and project details  

### 🧠 How It Works

1. User sends an image 📸  
2. Bot detects pistol 🔍  
3. Returns the image with bounding boxes 🟥  

### ▶️ Run the Bot

```bash
python telegram_bot.py
```

---

## ⚙️ Required Setup

To get started with the **Pistol Object Detection** system, follow these simple setup steps:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/pistol-object-detection.git
   cd pistol-object-detection
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Telegram Bot (Optional)**

- 🤖 Create a bot using **[@BotFather](https://t.me/BotFather)** on Telegram.  
- 🔑 Retrieve your **API Token** and add it to a `.env` file in your project directory:
  ```bash
  TELEGRAM_BOT_TOKEN=your_token_here
  ```

4. **🚀 Run the Applications**

### 🧠 For Streamlit Web App

Launch the interactive web app to visualize pistol detection in real-time:

```bash
streamlit run app.py
```

### 🤖 For Telegram Bot

Run the bot to enable pistol detection directly via chat:

```bash
python telegram_bot.py
```

Once running, the bot supports:

- /start → Welcome message
- /help → How to use the bot
- /about → Project and model details

---

## ⚙️ Installation

### 🧩 Clone Repository

```bash
git clone https://github.com/yourusername/pistol-object-detection.git
cd pistol-object-detection
```

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages

- torch
- torchvision
- streamlit
- pillow
- opencv-python
- python-telegram-bot
- numpy
- python-dotenv

---

## 🧱 Project Structure

📂 Pistol Object Detection using Faster R-CNN
│
├── 📄 app.py                        # Streamlit web app
├── 📄 telegram_bot.py               # Telegram bot integration
├── 📄 detection.py                  # Core detection logic
├── 📄 pistol_object_detector.pth    # Trained model weights
├── 📄 requirements.txt              # Python dependencies
├── 📄 .env                          # Telegram bot token
├── 📁 data/                         # Dataset directory
├── 📁 outputs/                      # Model outputs & results
└── 📄 README.md                     # Project documentation

---

## 🧩 Key Learnings

- Built a custom Faster R-CNN object detector for domain-specific pistol detection.
- Used transfer learning with fine-tuning for faster convergence.
- Developed real-time, interactive AI apps using Streamlit and Telegram.
- Integrated deep learning with communication platforms for instant inference.

---

## 🧰 Tools & Frameworks

## Category & Tools

### Deep Learning
- PyTorch
- Torchvision

### Data Handling
- OpenCV
- PIL
- NumPy

### Visualization
- Matplotlib
- Seaborn

### Deployment
- Streamlit
- Telegram Bot API

### Utilities
- dotenv
- tqdm

---

## 🌍 Future Enhancements
- 🚀 Expand to detect multiple weapon types (rifles, knives, etc.)
- 🧩 Integrate YOLOv8 for faster, real-time detection
- 🧠 Deploy on edge devices (Jetson Nano, Raspberry Pi)
- 🌐 Create a web-based analytics dashboard for detections
- 🔔 Enable CCTV-based alert system for proactive monitoring

---

## 🧡 Acknowledgements

Special thanks to:
- 🧠 PyTorch & Torchvision teams for robust deep learning frameworks
- 💻 Streamlit for enabling rapid AI app deployment
- 🤖 Telegram Bot API for seamless conversational integration
- 📚 Faster R-CNN (Ren et al.) researchers for their foundational work in object detection

---

## 👨‍💻 Author

**Sayam Kumar**  
📧 sayamk565@gmail.com  
🔗 [LinkedIn](#) | [GitHub](#)  

> “AI isn’t just about automation — it’s about making intelligence visible.”

---

## 🏁 License

This project is licensed under the MIT License.  
You are free to use, modify, and distribute it with proper attribution.