# 🟢 Real-Time Face Mask Detection  
A computer vision project utilizing deep learning with TensorFlow/Keras and OpenCV to detect whether individuals in a video stream are wearing face masks. This system can be deployed for real-time monitoring and compliance checks in public spaces.

Overview
This project implements a robust face mask detection model trained on a custom dataset. The goal is to provide an efficient and accurate method for identifying correct mask usage in real-world scenarios. The core technology stack relies on Python, OpenCV for video processing, and a deep learning model for classification.
A deep learning–based real-time face mask detection system using **TensorFlow/Keras**, **OpenCV**, and **MobileNetV2**.  
The model detects whether a person is wearing a mask **with_mask** or **without_mask** using a webcam feed or video stream.

The process involves:
Data Preprocessing: Preparing images of masked and unmasked faces.
Model Training: Utilizing a Convolutional Neural Network (CNN) architecture via Keras.
Real-time Inference: Applying the trained model to live video feeds to draw bounding boxes and status labels ("Mask On" / "Mask Off")

---

## 📌 Features
- ✔ **Real-time detection** using OpenCV  
- ✔ **Deep Learning model** trained using MobileNetV2  
- ✔ **High accuracy** on test images  
- ✔ Detects multiple faces simultaneously  
- ✔ Easy to run on any system  
- ✔ Lightweight & fast inference  

---

## 📂 Project Structure
Real_time_Face_Mask/

│── mask_detector.h5 # Trained classification model

│── deploy.prototxt AND res10_300x300_ssd_iter_140000.caffemodel # Face detector config

│── res10_300x300_ssd_iter_140000.caffemodel # Face detector weights

│── facemaskdetection.ipynb # Real-time mask detection script/jupyternotebook


│── plot.png # Model training_accuracy/loss curves

│── requirements.txt

│── README.md
-Add your dataset in facemaskdetection.ipynb



---

## 🛠️ Installation

### 1. Clone the Repository

git clone https://github.com/KrishMakwana28/Real_time_Face_Mask.git
cd Real_time_Face_Mask

## Create Virtual Environment (Recommended)
python_env - 3.10
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate
pip install -r requirements.txt
#Run from Start if you want to train model again(Remember to change the path in Notebook_Cell)
#Or directly Run the last cell to implement

program will:

Detect faces using OpenCV DNN

Classify each face as with_mask or without_mask

Display bounding boxes with colors and labels

## 📊 Model Performance
Metric	Score
Accuracy	99%
Precision	0.99
Recall	0.99
F1-Score	0.99

Evaluated on a dataset containing both mask and no-mask samples.

The training plot is shown below:

## 🧠 Dataset Information

This project uses a mask/no-mask image dataset containing:

~7284 training images(I pick it up from kaggle.com -Dataset)

Balanced classes (mask & no mask)

Data augmentation applied

80/20 train-test split

You can replace or extend the dataset to improve accuracy.

## 🧪 Model Architecture

Backbone: MobileNetV2

Final Layers: Fully Connected + Softmax

Optimizer: Adam

Loss Function: Binary Crossentropy

This architecture is optimized for speed + accuracy, making it ideal for real-time applications.

## 📦 Training the Model (Optional)

If you want to retrain the model:

facemaskdetection.ipynb
(#Run from Start if you want to train model again(Remember to change the path in Notebook_Cell)
#Or directly Run the last cell to implement)


(Ensure dataset is in the correct folder structure)

## 📸 Demo Output

Green box → Mask Detected

Red box → No Mask

FPS optimized for smooth performance


## 📜 License

This project is open-source and available.

## 🔗 LinkedIn Project Post

This project has been shared on LinkedIn :

👉 View the LinkedIn post here:
https://www.linkedin.com/posts/krish-makwana-58ab64374_ai-machinelearning-deeplearning-activity-7409257287015391232-qR7q?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFy4SDoB5RKus0IjrjxH2XoHrWA_8rtyLCY

## ✨ Author

Krish Makwana - https://github.com/KrishMakwana28
Feel free to reach out for suggestions or improvements.
