Real-time Face Mask Detection System
A computer vision project utilizing deep learning with TensorFlow/Keras and OpenCV to detect whether individuals in a video stream are wearing face masks. This system can be deployed for real-time monitoring and compliance checks in public spaces.


Overview
This project implements a robust face mask detection model trained on a custom/online dataset. The goal is to provide an efficient and accurate method for identifying correct mask usage in real-world scenarios. The core technology stack relies on Python, OpenCV for video processing, and a deep learning model for classification.

The process involves:
Data Preprocessing: Preparing images of masked and unmasked faces.
Model Training: Utilizing a Convolutional Neural Network (CNN) architecture via Keras.
Real-time Inference: Applying the trained model to live video feeds to draw bounding boxes and status labels ("Mask On" / "Mask Off").

reference:
Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.
Prerequisites:
You need Python 3.10 installed on your system.
Installation
Clone the repository:
bash
git clone github.com
cd Real_time_Face_Mask
Use code with caution.

Set up a virtual environment (recommended):Using conda:
bash
conda create -p venv python==3.10 -y
conda activate venv
Use code with caution.


Install project dependencies:
pip install -r requirements.txt 
bash
pip install tensorflow keras opencv-python numpy imutils
Use code with caution.

Usage
The project can be run directly from the main Jupyter Notebook.
Open the notebook:Ensure you have Jupyter Lab or Jupyter Notebook installed (usually included if you use Anaconda).
bash
jupyter notebook facemaskdetectionsystem.ipynb
Use code with caution.

Run the cells:Follow the instructions within the notebook to load the model, prepare the input data, and start the real-time detection loop using your webcam.
Things to Do/Notebook Guide
The following tasks are handled within the main notebook (facemaskdetectionsystem.ipynb):
Environment Setup: Ensure you are running Python 3.10.
Data Loading: Instructions on loading the pre-processed dataset.
Model Training & Evaluation: Steps for running the CNN training cycle.
Real-time Detection: Code to initialize the webcam stream and apply the trained model for inference.
Path Management: Remember to adjust model paths in the code where specified by comments to point to your local file structure.
Dataset Information
The model in this repository was trained using data sourced from public domain datasets. You can find suitable datasets for training and testing at the following resources:
Kaggle: Excellent source for curated machine learning datasets, often including face mask specific compilations. Explore the Kaggle website using search terms like "face mask detection".
Google Dataset Search: A comprehensive search engine for all public datasets worldwide, useful for finding diverse image collections. Search via Google Dataset Search.
Data.gov: Government, economics, health, and agriculture data, primarily useful for broader research but less focused on image data specifics.

We welcome contributions! If you have suggestions for improving the model, optimizing the code, or enhancing the documentation, please feel free to:
Fork the Project
Create your Feature Branch (git checkout -b feature/AmazingFeature)
Commit your Changes (git commit -m 'Add some AmazingFeature')
Push to the Branch (git push origin feature/AmazingFeature)
Open a Pull Request


Contact
Krish Makwana -https://github.com/KrishMakwana28
Project Link: github.com

Acknowledgments
OpenCV for image processing utilities.
TensorFlow/Keras team for the deep learning framework.
The open-source community for providing accessible datasets and tools.
