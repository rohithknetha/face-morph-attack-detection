import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import time
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

# ✅ Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory growth enabled.")
    except RuntimeError as e:
        print(f"❌ Error setting memory growth: {e}")

# ✅ Configurations
sequence_length = 25
image_size = (224, 224)
batch_size = 8
EPOCHS = 15

# ✅ Dataset Paths
train_dir = "F:/Major Project/Datasets/Final_Data/train/"
val_dir = "F:/Major Project/Datasets/Final_Data/val/"
model_dir = "F:/Major Project/Code/Saved_Models"
os.makedirs(model_dir, exist_ok=True)

# ✅ Load Sequences and Labels
def load_sequences(data_dir):
    sequences, labels = [], []
    for label, category in enumerate(["REAL", "FAKE"]):
        category_path = os.path.join(data_dir, category)
        for folder in sorted(os.listdir(category_path)):
            frame_paths = sorted([
                os.path.join(category_path, folder, f)
                for f in os.listdir(os.path.join(category_path, folder))
            ])
            for i in range(0, len(frame_paths) - sequence_length + 1, sequence_length):
                sequences.append(frame_paths[i:i+sequence_length])
                labels.append(label)
    return sequences, labels

train_sequences, train_labels = load_sequences(train_dir)
val_sequences, val_labels = load_sequences(val_dir)
print(f"✅ Loaded {len(train_sequences)} train and {len(val_sequences)} val sequences")

# ✅ CNN Feature Extractor
base_model = keras.applications.EfficientNetB2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
gap_layer = keras.layers.GlobalAveragePooling2D()
cnn_model = keras.Model(inputs=base_model.input, outputs=gap_layer(base_model.output))

def extract_features(sequences, model):
    all_features = []
    for sequence in tqdm(sequences, desc="🔄 Extracting Features", unit="seq"):
        batch = [keras.preprocessing.image.img_to_array(
                    keras.preprocessing.image.load_img(path, target_size=image_size)) / 255.0
                 for path in sequence]
        features = model.predict(np.array(batch), verbose=0)
        all_features.append(features)
    return np.array(all_features)

start_time = time.time()
x_train = extract_features(train_sequences, cnn_model)
x_val = extract_features(val_sequences, cnn_model)
y_train = np.array(train_labels)
y_val = np.array(val_labels)
print(f"✅ Feature extraction complete in {(time.time() - start_time)/60:.2f} mins.")

# ✅ One-hot Encode Labels
y_train = to_categorical(y_train, num_classes=2)
y_val = to_categorical(y_val, num_classes=2)

# ✅ Compute Class Weights
train_labels_flat = np.argmax(y_train, axis=1)
class_weights = dict(enumerate(class_weight.compute_class_weight(
    class_weight='balanced', classes=np.unique(train_labels_flat), y=train_labels_flat)))
print(f"✅ Class weights: {class_weights}")

# ✅ Shuffle Training Data
indices = np.arange(len(x_train))
np.random.shuffle(indices)
x_train = x_train[indices]
y_train = y_train[indices]

# ✅ Model Architecture
model = keras.models.Sequential([
    keras.layers.Input(shape=(sequence_length, 1408)),
    keras.layers.LSTM(128, return_sequences=False, dropout=0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ✅ Custom Callback to Show ETA and Save Accuracy/Loss Plots
class TimeAndPlotCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        self.train_acc = []
        self.val_acc = []
        self.train_loss = []
        self.val_loss = []

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.start_time
        eta = (elapsed / (epoch + 1)) * (EPOCHS - epoch - 1)
        print(f"⏱️ Time Elapsed: {elapsed:.2f}s || ⏳ ETA: {eta:.2f}s")

        self.train_acc.append(logs['accuracy'])
        self.val_acc.append(logs['val_accuracy'])
        self.train_loss.append(logs['loss'])
        self.val_loss.append(logs['val_loss'])

        # Plot & Save Progress
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.train_acc, label='Train Acc')
        plt.plot(self.val_acc, label='Val Acc')
        plt.title('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.train_loss, label='Train Loss')
        plt.plot(self.val_loss, label='Val Loss')
        plt.title('Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{model_dir}/lstm_cnn_progress_epoch_{epoch+1}.png")
        plt.close()

# ✅ Callbacks
callbacks = [
    keras.callbacks.ModelCheckpoint(f"{model_dir}/lstm_cnn_best.h5", save_best_only=True, monitor='val_accuracy', mode='max'),
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6),
    TimeAndPlotCallback()
]

# ✅ Train the Model
print("🚀 Starting training...\n")
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    batch_size=batch_size,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# ✅ Save Training History
with open(f"{model_dir}/lstm_cnn_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

# ✅ Evaluate Model
y_val_pred = model.predict(x_val)
y_val_true = np.argmax(y_val, axis=1)
y_val_pred_label = np.argmax(y_val_pred, axis=1)

print("📊 Classification Report:\n", classification_report(y_val_true, y_val_pred_label))
print("📉 Confusion Matrix:\n", confusion_matrix(y_val_true, y_val_pred_label))

# ✅ Find and Save Best Threshold
precisions = []
thresholds = np.linspace(0.3, 0.9, 61)
for thresh in thresholds:
    y_pred_thresh = (y_val_pred[:, 1] >= thresh).astype(int)
    TP = np.sum((y_val_true == 1) & (y_pred_thresh == 1))
    FP = np.sum((y_val_true == 0) & (y_pred_thresh == 1))
    FN = np.sum((y_val_true == 1) & (y_pred_thresh == 0))
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    precisions.append((thresh, precision, recall, f1))

best_thresh = max(precisions, key=lambda x: x[1] * x[2])  # maximize precision * recall
print(f"✅ Best threshold found: {best_thresh[0]:.2f} | Precision: {best_thresh[1]:.2f} | Recall: {best_thresh[2]:.2f} | F1: {best_thresh[3]:.2f}")

with open(f"{model_dir}/best_threshold.pkl", "wb") as f:
    pickle.dump(best_thresh[0], f)

print("✅ Training complete and best threshold saved.")
