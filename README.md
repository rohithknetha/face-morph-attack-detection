# face-morph-attack-detection
Face Morph Attack Detection using Deep Learning detects morphed facial images that attempt to bypass biometric systems. The project uses an LSTM-CNN hybrid model for spatial and temporal feature learning to classify genuine and morphed faces, achieving 74.9% test accuracy on the dataset.
#  Face Morph Attack Detection Using Deep Learning

This project detects **face morphing attacks** using a deep learning-based approach. It combines **EfficientNetB2** for extracting spatial features and **LSTM/CNN** layers to analyze temporal relationships. A clean and user-friendly **Flask web application** allows users to upload a video and receive a prediction: `REAL` or `MORPHED`.

---

##  Project Structure

Face-Morph-Attack-Detection/
├── Code/ # Python scripts for the entire pipeline
│ ├── extract_frames.py
│ ├── feature_extraction.py
│ ├── model_training.py
│ ├── evaluate_model.py
│ └── app.py # Flask backend
├── WebApp/ # Frontend files for web interface
│ ├── templates/
│ │ ├── index.html
│ │ └── result.html
│ └── static/
│ ├── style.css
│ └── scripts.js
├── saved_models/ # Trained .h5 model files
├── outputs/ # Evaluation results, graphs
├── requirements.txt # Required Python packages
└── README.md # Project documentation

##  Key Features

- Extracts frames from video input
- Uses EfficientNetB2 for spatial feature extraction
- Employs LSTM + CNN to learn temporal attack patterns
- Evaluates model using accuracy, ROC-AUC, confusion matrix, etc.
- Flask-based web app for easy real-time predictions

---

##  How to Run the Project

### 1.  Clone the Repository
```bash
git clone https://github.com/yourusername/face-morph-attack-detection.git
cd face-morph-attack-detection
2. Create Virtual Environment & Install Dependencies
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install -r requirements.txt
3.  Run the Flask Web App
python app.py
Then open your browser and go to: http://127.0.0.1:5000

 Model Performance
Test Accuracy	: 74.90%
Test Loss :	0.5449
ROC-AUC Score :	~0.87



 Technologies Used
Python

TensorFlow / Keras

OpenCV

Flask

HTML/CSS/JS

EfficientNetB2

LSTM / CNN

scikit-learn

Matplotlib / Seaborn

👨‍💻 Author
Rohit
Final Year B.Tech. – Electronics and Communication Engineering
Project Title: Face Morph Attack Detection Using Deep Learning

📄 License
This project is licensed under the MIT License. Feel free to use, modify, and share!
