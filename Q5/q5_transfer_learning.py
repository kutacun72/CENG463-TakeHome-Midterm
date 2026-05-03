# Question 5 - Transfer Learning
# MobileNetV2 on CIFAR-10

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import tarfile
import pickle
import warnings

warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# -----------------------------
# Load CIFAR-10 from local archive
# -----------------------------

def load_cifar10_from_tar(tar_path):
    extract_dir = "cifar10_data"

    if not os.path.exists(extract_dir):
        print("Extracting CIFAR-10 archive...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)

    batch_dir = os.path.join(extract_dir, "cifar-10-batches-py")

    X_train_batches = []
    y_train_batches = []

    for i in range(1, 6):
        batch_path = os.path.join(batch_dir, f"data_batch_{i}")
        with open(batch_path, "rb") as f:
            batch = pickle.load(f, encoding="latin1")
            X_train_batches.append(batch["data"])
            y_train_batches.extend(batch["labels"])

    test_batch_path = os.path.join(batch_dir, "test_batch")
    with open(test_batch_path, "rb") as f:
        test_batch = pickle.load(f, encoding="latin1")
        X_test = test_batch["data"]
        y_test = test_batch["labels"]

    X_train = np.concatenate(X_train_batches, axis=0)
    y_train = np.array(y_train_batches)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    X_train = X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    X_test = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    return (X_train, y_train), (X_test, y_test)


print("Loading CIFAR-10 from local archive...")

tar_path = r"C:\Users\LENOVO\.keras\datasets\cifar-10-python.tar.gz"
(X_train_full, y_train_full), (X_test, y_test) = load_cifar10_from_tar(tar_path)

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Smaller subset for practical CPU training
subset_size = 10000
indices = np.random.choice(X_train_full.shape[0], subset_size, replace=False)

X_train = X_train_full[indices].astype("float32")
y_train = y_train_full[indices]

X_test = X_test.astype("float32")

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

print("Transfer learning data ready.")
print("Train subset:", X_train.shape)
print("Test:", X_test.shape)

# -----------------------------
# Build transfer learning model
# -----------------------------

print("\nBuilding MobileNetV2 transfer learning model...")

data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.1),
], name="transfer_data_augmentation")

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(96, 96, 3)
)

base_model.trainable = False

inputs = layers.Input(shape=(32, 32, 3))

x = data_augmentation(inputs)

# MobileNetV2 expects larger images, so resize CIFAR-10 images
x = layers.Resizing(96, 96)(x)

# MobileNetV2 preprocessing expects image values in [0,255]
x = preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)

outputs = layers.Dense(10, activation="softmax")(x)

transfer_model = models.Model(inputs, outputs)

transfer_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

transfer_model.summary()

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# -----------------------------
# Stage 1: Train classifier head
# -----------------------------

print("\nStage 1: Training classifier head...")

start_time = time.time()

history_head = transfer_model.fit(
    X_train,
    y_train_cat,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# -----------------------------
# Stage 2: Fine-tune last layers
# -----------------------------

print("\nStage 2: Fine-tuning last MobileNetV2 layers...")

base_model.trainable = True

# Freeze most layers, fine-tune only the last part
for layer in base_model.layers[:-20]:
    layer.trainable = False

transfer_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history_finetune = transfer_model.fit(
    X_train,
    y_train_cat,
    epochs=5,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

transfer_runtime = time.time() - start_time

print("Transfer learning completed.")
print("Transfer runtime:", round(transfer_runtime, 4), "seconds")
print("Final transfer training accuracy:", round(history_finetune.history["accuracy"][-1], 4))
print("Final transfer validation accuracy:", round(history_finetune.history["val_accuracy"][-1], 4))
print("Final transfer training loss:", round(history_finetune.history["loss"][-1], 4))
print("Final transfer validation loss:", round(history_finetune.history["val_loss"][-1], 4))

# -----------------------------
# Save training curves
# -----------------------------

acc = history_head.history["accuracy"] + history_finetune.history["accuracy"]
val_acc = history_head.history["val_accuracy"] + history_finetune.history["val_accuracy"]
loss = history_head.history["loss"] + history_finetune.history["loss"]
val_loss = history_head.history["val_loss"] + history_finetune.history["val_loss"]

plt.figure(figsize=(8, 5))
plt.plot(acc, label="Training Accuracy")
plt.plot(val_acc, label="Validation Accuracy")
plt.title("Transfer Learning Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig("q5_transfer_accuracy_curve.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")
plt.title("Transfer Learning Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("q5_transfer_loss_curve.png", dpi=300)
plt.close()

# -----------------------------
# Test-set evaluation
# -----------------------------

print("\nEvaluating transfer model on test set...")

y_prob = transfer_model.predict(X_test, batch_size=64)
y_pred = np.argmax(y_prob, axis=1)

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")

top5_preds = np.argsort(y_prob, axis=1)[:, -5:]
top5_correct = np.array([y_test[i] in top5_preds[i] for i in range(len(y_test))])
top5_error = 1 - np.mean(top5_correct)

per_class_recall = recall_score(y_test, y_pred, average=None)

print("Transfer Test Accuracy:", round(accuracy, 4))
print("Transfer Macro-F1:", round(macro_f1, 4))
print("Transfer Top-5 Error Rate:", round(top5_error, 4))
print("Transfer Per-class Recall:")
for i, r in enumerate(per_class_recall):
    print(f"  {class_names[i]}: {round(r, 4)}")

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation="nearest")
plt.title("Transfer Learning Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45, ha="right")
plt.yticks(tick_marks, class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("q5_transfer_confusion_matrix.png", dpi=300)
plt.close()

transfer_results_df = pd.DataFrame([{
    "Model": "Transfer Learning MobileNetV2",
    "Accuracy": accuracy,
    "Macro-F1": macro_f1,
    "Top-5 Error Rate": top5_error,
    "Runtime": transfer_runtime
}])

transfer_results_df.to_csv("q5_transfer_results_summary.csv", index=False)

recall_df = pd.DataFrame({
    "Class": class_names,
    "Recall": per_class_recall
})

recall_df.to_csv("q5_transfer_per_class_recall.csv", index=False)

transfer_model.save("q5_transfer_model.keras")

print("\nTransfer files saved:")
print("q5_transfer_accuracy_curve.png")
print("q5_transfer_loss_curve.png")
print("q5_transfer_confusion_matrix.png")
print("q5_transfer_results_summary.csv")
print("q5_transfer_per_class_recall.csv")
print("q5_transfer_model.keras")

# -----------------------------
# Final comparison of MLP, CNN, and Transfer Learning
# -----------------------------

print("\nCreating final model comparison table...")

final_comparison_df = pd.DataFrame({
    "Model": ["MLP", "CNN", "Transfer Learning MobileNetV2"],
    "Accuracy": [0.3241, 0.5775, accuracy],
    "Macro-F1": [0.3018, 0.5747, macro_f1],
    "Top-5 Error Rate": [0.1689, 0.0420, top5_error]
})

print(final_comparison_df)

final_comparison_df.to_csv("q5_final_model_comparison.csv", index=False)

# Accuracy comparison
plt.figure(figsize=(8, 5))
plt.bar(final_comparison_df["Model"], final_comparison_df["Accuracy"])
plt.title("Q5 Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Test Accuracy")
plt.xticks(rotation=20, ha="right")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("q5_accuracy_comparison.png", dpi=300)
plt.close()

# Macro-F1 comparison
plt.figure(figsize=(8, 5))
plt.bar(final_comparison_df["Model"], final_comparison_df["Macro-F1"])
plt.title("Q5 Model Macro-F1 Comparison")
plt.xlabel("Model")
plt.ylabel("Macro-F1")
plt.xticks(rotation=20, ha="right")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("q5_macro_f1_comparison.png", dpi=300)
plt.close()

# Top-5 error comparison
plt.figure(figsize=(8, 5))
plt.bar(final_comparison_df["Model"], final_comparison_df["Top-5 Error Rate"])
plt.title("Q5 Model Top-5 Error Rate Comparison")
plt.xlabel("Model")
plt.ylabel("Top-5 Error Rate")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("q5_top5_error_comparison.png", dpi=300)
plt.close()

print("\nFinal comparison files saved:")
print("q5_final_model_comparison.csv")
print("q5_accuracy_comparison.png")
print("q5_macro_f1_comparison.png")
print("q5_top5_error_comparison.png")