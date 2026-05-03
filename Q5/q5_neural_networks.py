# Question 5 - Neural Networks
# CENG 463 Machine Learning Take-Home Midterm

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import warnings
import tarfile
import pickle

warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    recall_score,
    classification_report
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# -----------------------------
# Step 1: Manual CIFAR-10 loader
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

    # CIFAR-10 files are stored as flat vectors:
    # 3072 = 3 channels x 32 x 32
    # Convert to image format: (samples, 32, 32, 3)
    X_train = X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    X_test = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    return (X_train, y_train), (X_test, y_test)


print("Loading CIFAR-10 dataset from local archive...")

tar_path = r"C:\Users\LENOVO\.keras\datasets\cifar-10-python.tar.gz"

(X_train_full, y_train_full), (X_test, y_test) = load_cifar10_from_tar(tar_path)

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

print("CIFAR-10 loaded successfully.")
print("Full train shape:", X_train_full.shape)
print("Test shape:", X_test.shape)
print("Train labels shape:", y_train_full.shape)
print("Test labels shape:", y_test.shape)
print("Classes:", class_names)
print("Pixel value range before scaling:", X_train_full.min(), X_train_full.max())

# -----------------------------
# Step 2: Preprocessing
# -----------------------------

subset_size = 15000

indices = np.random.choice(
    X_train_full.shape[0],
    subset_size,
    replace=False
)

X_train = X_train_full[indices]
y_train = y_train_full[indices]

# Normalize images to [0, 1]
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# One-hot labels for neural network training
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

print("\nPreprocessing completed.")
print("Selected train subset shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Pixel value range after scaling:", X_train.min(), X_train.max())
print("Class distribution in selected train subset:")
print(pd.Series(y_train).value_counts().sort_index())

# -----------------------------
# Helper: Plot training curves
# -----------------------------

def save_training_curves(history, model_name, accuracy_filename, loss_filename):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title(f"{model_name} Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(accuracy_filename, dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"{model_name} Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_filename, dpi=300)
    plt.close()

# -----------------------------
# Step 3: Deep MLP model
# -----------------------------

print("\nTraining Deep MLP model...")

def build_mlp_model(dropout_rate=0.4, learning_rate=0.001, weight_decay=1e-4):
    model = models.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Flatten(),

        layers.Dense(
            512,
            activation="relu",
            kernel_regularizer=regularizers.l2(weight_decay)
        ),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),

        layers.Dense(
            512,
            activation="relu",
            kernel_regularizer=regularizers.l2(weight_decay)
        ),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),

        layers.Dense(
            512,
            activation="relu",
            kernel_regularizer=regularizers.l2(weight_decay)
        ),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),

        layers.Dense(
            512,
            activation="relu",
            kernel_regularizer=regularizers.l2(weight_decay)
        ),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),

        layers.Dense(10, activation="softmax")
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


mlp_model = build_mlp_model(
    dropout_rate=0.4,
    learning_rate=0.001,
    weight_decay=1e-4
)

mlp_model.summary()

early_stop_mlp = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

start_time = time.time()

mlp_history = mlp_model.fit(
    X_train,
    y_train_cat,
    epochs=40,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stop_mlp],
    verbose=1
)

mlp_runtime = time.time() - start_time

print("Deep MLP training completed.")
print("MLP runtime:", round(mlp_runtime, 4), "seconds")
print("Final MLP training accuracy:", round(mlp_history.history["accuracy"][-1], 4))
print("Final MLP validation accuracy:", round(mlp_history.history["val_accuracy"][-1], 4))
print("Final MLP training loss:", round(mlp_history.history["loss"][-1], 4))
print("Final MLP validation loss:", round(mlp_history.history["val_loss"][-1], 4))

save_training_curves(
    mlp_history,
    "MLP",
    "q5_mlp_accuracy_curve.png",
    "q5_mlp_loss_curve.png"
)

print("MLP curves saved:")
print("q5_mlp_accuracy_curve.png")
print("q5_mlp_loss_curve.png")

# -----------------------------
# Step 4: CNN model with data augmentation
# -----------------------------

print("\nTraining CNN model...")

data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.1),
], name="data_augmentation")


def build_cnn_model(dropout_rate=0.3, learning_rate=0.001, weight_decay=1e-4):
    inputs = layers.Input(shape=(32, 32, 3))

    x = data_augmentation(inputs)

    x = layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(weight_decay)
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(weight_decay)
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(weight_decay)
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(weight_decay)
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(weight_decay)
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(weight_decay)
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)

    # Correct Flatten usage
    x = layers.Flatten()(x)

    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(weight_decay)
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(10, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


cnn_model = build_cnn_model(
    dropout_rate=0.3,
    learning_rate=0.001,
    weight_decay=1e-4
)

cnn_model.summary()

early_stop_cnn = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

start_time = time.time()

cnn_history = cnn_model.fit(
    X_train,
    y_train_cat,
    epochs=40,
    batch_size=128,
    validation_split=0.2,
    callbacks=[early_stop_cnn],
    verbose=1
)

cnn_runtime = time.time() - start_time

print("CNN training completed.")
print("CNN runtime:", round(cnn_runtime, 4), "seconds")
print("Final CNN training accuracy:", round(cnn_history.history["accuracy"][-1], 4))
print("Final CNN validation accuracy:", round(cnn_history.history["val_accuracy"][-1], 4))
print("Final CNN training loss:", round(cnn_history.history["loss"][-1], 4))
print("Final CNN validation loss:", round(cnn_history.history["val_loss"][-1], 4))

save_training_curves(
    cnn_history,
    "CNN",
    "q5_cnn_accuracy_curve.png",
    "q5_cnn_loss_curve.png"
)

print("CNN curves saved:")
print("q5_cnn_accuracy_curve.png")
print("q5_cnn_loss_curve.png")

# -----------------------------
# Step 5: Test-set evaluation for MLP and CNN
# -----------------------------

print("\nSaving trained models...")

mlp_model.save("q5_mlp_model.keras")
cnn_model.save("q5_cnn_model.keras")

print("Models saved:")
print("q5_mlp_model.keras")
print("q5_cnn_model.keras")


def top5_error_rate(y_true, y_prob):
    top5_preds = np.argsort(y_prob, axis=1)[:, -5:]
    correct_top5 = np.array([
        y_true[i] in top5_preds[i]
        for i in range(len(y_true))
    ])
    top5_accuracy = np.mean(correct_top5)
    return 1 - top5_accuracy


def evaluate_model(model, model_name, X_eval, y_eval):
    print(f"\nEvaluating {model_name} on test set...")

    y_prob = model.predict(X_eval, batch_size=128)
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_eval, y_pred)
    macro_f1 = f1_score(y_eval, y_pred, average="macro")
    top5_error = top5_error_rate(y_eval, y_prob)
    per_class_recall = recall_score(y_eval, y_pred, average=None)

    print(f"{model_name} Test Accuracy:", round(acc, 4))
    print(f"{model_name} Macro-F1:", round(macro_f1, 4))
    print(f"{model_name} Top-5 Error Rate:", round(top5_error, 4))
    print(f"{model_name} Per-class Recall:")
    for i, recall in enumerate(per_class_recall):
        print(f"  {class_names[i]}: {round(recall, 4)}")

    cm = confusion_matrix(y_eval, y_pred)

    # Save confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"{model_name} Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(f"q5_{model_name.lower()}_confusion_matrix.png", dpi=300)
    plt.close()

    recall_df = pd.DataFrame({
        "Class": class_names,
        "Recall": per_class_recall
    })

    recall_df.to_csv(f"q5_{model_name.lower()}_per_class_recall.csv", index=False)

    return {
        "Model": model_name,
        "Accuracy": acc,
        "Macro-F1": macro_f1,
        "Top-5 Error Rate": top5_error
    }


mlp_test_results = evaluate_model(
    mlp_model,
    "MLP",
    X_test,
    y_test
)

cnn_test_results = evaluate_model(
    cnn_model,
    "CNN",
    X_test,
    y_test
)

test_results_df = pd.DataFrame([
    mlp_test_results,
    cnn_test_results
])

print("\nFinal test-set comparison:")
print(test_results_df)

test_results_df.to_csv("q5_test_results_summary.csv", index=False)

print("\nEvaluation files saved:")
print("q5_mlp_confusion_matrix.png")
print("q5_cnn_confusion_matrix.png")
print("q5_mlp_per_class_recall.csv")
print("q5_cnn_per_class_recall.csv")
print("q5_test_results_summary.csv")