import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import time
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

# ✅ Configuration
sequence_length = 25
image_size = (224, 224)
test_dir = "F:/Major Project/Datasets/Final_Data/test/"
MODEL_PATH = "F:/Major Project/Code/Saved_Models/lstm_cnn_best.h5"

# ✅ Load test sequences
def load_sequences(data_dir):
    sequences, labels = [], []
    for label, category in enumerate(["REAL", "FAKE"]):
        category_path = os.path.join(data_dir, category)
        for folder in sorted(os.listdir(category_path)):
            folder_path = os.path.join(category_path, folder)
            frames = sorted(os.listdir(folder_path))
            frame_paths = [os.path.join(folder_path, f) for f in frames]
            for i in range(0, len(frame_paths) - sequence_length + 1, sequence_length):
                sequences.append(frame_paths[i:i+sequence_length])
                labels.append(label)
    return sequences, labels

print("📂 Loading test sequences...")
test_sequences, test_labels = load_sequences(test_dir)
y_test = np.array(test_labels)
print(f"✅ Loaded {len(test_sequences)} test sequences.")

# ✅ Feature extractor (EfficientNetB2)
base_model = keras.applications.EfficientNetB2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
gap = keras.layers.GlobalAveragePooling2D()
cnn_model = keras.Model(inputs=base_model.input, outputs=gap(base_model.output))

# ✅ Extract features
def extract_features(sequences, extractor):
    features = []
    start_time = time.time()
    for idx, seq in enumerate(tqdm(sequences, desc="🖼️ Extracting Features", unit="seq")):
        imgs = []
        for path in seq:
            img = keras.preprocessing.image.load_img(path, target_size=image_size)
            img = keras.preprocessing.image.img_to_array(img) / 255.0
            imgs.append(img)
        imgs = np.array(imgs)
        feats = extractor.predict(imgs, verbose=0)
        features.append(feats)

        elapsed = time.time() - start_time
        avg_time = elapsed / (idx + 1)
        remaining = avg_time * (len(sequences) - idx - 1)
        print(f"\r⏱️ Progress: {idx+1}/{len(sequences)} | Elapsed: {elapsed:.2f}s | ETA: {remaining:.2f}s", end='')
    print()
    return np.array(features)

print("🔍 Extracting features from test data...")
x_test = extract_features(test_sequences, cnn_model)

# ✅ Load trained model
print("📦 Loading trained LSTM+CNN model...")
model = keras.models.load_model(MODEL_PATH)

# ✅ Evaluate model
print("🧪 Evaluating model on test data...")
start_eval = time.time()
y_test_cat = to_categorical(y_test, num_classes=2)
test_loss, test_accuracy = model.evaluate(x_test, y_test_cat, verbose=1)
end_eval = time.time()

# ✅ Predict
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_fake_probs = y_pred_probs[:, 1]  # Probability of being FAKE class

# ✅ Evaluation results
print("\n🎯 Final Evaluation Results:")
print(f"✅ Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"✅ Test Loss: {test_loss:.4f}")
print(f"⏱️ Evaluation Time: {end_eval - start_eval:.2f} seconds")

print("\n📌 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["REAL", "FAKE"]))

print("📌 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ✅ Plot histogram of predicted FAKE probabilities
plt.hist(y_fake_probs[y_test==0], bins=20, alpha=0.7, label='REAL')
plt.hist(y_fake_probs[y_test==1], bins=20, alpha=0.7, label='FAKE')
plt.title("Prediction Probability Distribution (FAKE Class)")
plt.xlabel("Predicted Probability of Being FAKE")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ✅ Try different thresholds on FAKE probability
for threshold in [0.5, 0.4, 0.3, 0.2]:
    print(f"\n🔍 Threshold = {threshold}")
    y_pred_thresh = (y_fake_probs > threshold).astype(int)
    print("📌 Classification Report:")
    print(classification_report(y_test, y_pred_thresh, target_names=["REAL", "FAKE"]))
    print("📌 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_thresh))

# ✅ ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_fake_probs)
print(f"\n🔥 ROC-AUC Score (FAKE Class): {roc_auc:.4f}")
