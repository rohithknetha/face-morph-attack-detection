import pickle
import matplotlib.pyplot as plt

# ✅ Load the history object
history_path = "F:/Major Project/Code/Saved_Models/lstm_cnn_history.pkl"
with open(history_path, "rb") as f:
    history = pickle.load(f)

print("📂 Keys in history:", history.keys())
for key, value in history.items():
    print(f"{key}: {value[:5]}")  # Print first 5 values for each metric

# ✅ Extract values
acc = history['accuracy']
val_acc = history['val_accuracy']
loss = history['loss']
val_loss = history['val_loss']
epochs = range(1, len(acc) + 1)

# ✅ Plot accuracy
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'b-', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# ✅ Plot loss
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'b-', label='Training Loss')
plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# ✅ Save and show
plt.tight_layout()
plt.savefig("F:/Major Project/Code/Saved_Models/training_metrics_plot.png")
plt.show()
