# Question 5 - Interpretability and Robustness
# CENG 463 Machine Learning Take-Home Midterm

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tarfile
import pickle
import warnings

warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from sklearn.metrics import accuracy_score


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


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

    test_batch_path = os.path.join(batch_dir, "test_batch")

    with open(test_batch_path, "rb") as f:
        test_batch = pickle.load(f, encoding="latin1")
        X_test = test_batch["data"]
        y_test = test_batch["labels"]

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    X_test = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    return X_test, y_test


print("Loading CIFAR-10 test set...")

tar_path = r"C:\Users\LENOVO\.keras\datasets\cifar-10-python.tar.gz"

X_test_raw, y_test = load_cifar10_from_tar(tar_path)

X_test = X_test_raw.astype("float32") / 255.0
y_test_cat = to_categorical(y_test, 10)

print("Test set loaded.")
print("X_test shape:", X_test.shape)


# -----------------------------
# Load trained models
# -----------------------------

print("\nLoading trained models...")

mlp_model = models.load_model("q5_mlp_model.keras")
cnn_model = models.load_model("q5_cnn_model.keras")
transfer_model = models.load_model("q5_transfer_model.keras")

print("Models loaded successfully.")


# -----------------------------
# Misclassified examples for Transfer Learning model
# -----------------------------

print("\nFinding misclassified examples for transfer model...")

transfer_probs = transfer_model.predict(X_test_raw.astype("float32"), batch_size=64)
transfer_preds = np.argmax(transfer_probs, axis=1)

wrong_indices = np.where(transfer_preds != y_test)[0]
selected_wrong = wrong_indices[:10]

plt.figure(figsize=(12, 6))

for i, idx in enumerate(selected_wrong):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test_raw[idx])
    plt.axis("off")
    true_label = class_names[y_test[idx]]
    pred_label = class_names[transfer_preds[idx]]
    plt.title(f"T: {true_label}\nP: {pred_label}", fontsize=9)

plt.tight_layout()
plt.savefig("q5_transfer_10_misclassified_examples.png", dpi=300)
plt.close()

print("Saved q5_transfer_10_misclassified_examples.png")


# -----------------------------
# Simple Grad-CAM for CNN model
# -----------------------------

print("\nGenerating Grad-CAM examples for CNN...")

# Pick last Conv2D layer automatically
last_conv_layer_name = None
for layer in reversed(cnn_model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer_name = layer.name
        break

print("Last CNN conv layer:", last_conv_layer_name)

grad_model = tf.keras.models.Model(
    inputs=cnn_model.inputs,
    outputs=[
        cnn_model.get_layer(last_conv_layer_name).output,
        cnn_model.output
    ]
)

gradcam_indices = selected_wrong[:10]

plt.figure(figsize=(12, 6))

for i, idx in enumerate(gradcam_indices):
    img = X_test[idx:idx + 1]

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        pred_class = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_class]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap = heatmap / np.max(heatmap)

    heatmap = tf.image.resize(
        heatmap[..., np.newaxis],
        (32, 32)
    ).numpy().squeeze()

    plt.subplot(2, 5, i + 1)
    plt.imshow(X_test_raw[idx])
    plt.imshow(heatmap, alpha=0.45)
    plt.axis("off")
    plt.title(
        f"T:{class_names[y_test[idx]]}\nP:{class_names[transfer_preds[idx]]}",
        fontsize=9
    )

plt.tight_layout()
plt.savefig("q5_cnn_gradcam_examples.png", dpi=300)
plt.close()

print("Saved q5_cnn_gradcam_examples.png")


# -----------------------------
# FGSM adversarial robustness
# -----------------------------

print("\nRunning FGSM adversarial robustness test...")

def fgsm_attack(model, images, labels_cat, epsilon):
    images_tensor = tf.convert_to_tensor(images)

    with tf.GradientTape() as tape:
        tape.watch(images_tensor)
        predictions = model(images_tensor)
        loss = tf.keras.losses.categorical_crossentropy(labels_cat, predictions)

    gradient = tape.gradient(loss, images_tensor)
    signed_grad = tf.sign(gradient)

    adversarial_images = images_tensor + epsilon * signed_grad
    adversarial_images = tf.clip_by_value(adversarial_images, 0, 1)

    return adversarial_images.numpy()


def evaluate_fgsm(model, model_name, X_clean, y_true, y_cat, epsilon):
    clean_probs = model.predict(X_clean, batch_size=64, verbose=0)
    clean_preds = np.argmax(clean_probs, axis=1)
    clean_acc = accuracy_score(y_true, clean_preds)

    X_adv = fgsm_attack(model, X_clean, y_cat, epsilon)

    adv_probs = model.predict(X_adv, batch_size=64, verbose=0)
    adv_preds = np.argmax(adv_probs, axis=1)
    adv_acc = accuracy_score(y_true, adv_preds)

    print(f"{model_name} clean accuracy:", round(clean_acc, 4))
    print(f"{model_name} FGSM accuracy:", round(adv_acc, 4))

    return clean_acc, adv_acc


# Use small subset for speed
robustness_size = 1000
robust_indices = np.random.choice(X_test.shape[0], robustness_size, replace=False)

X_robust = X_test[robust_indices]
y_robust = y_test[robust_indices]
y_robust_cat = to_categorical(y_robust, 10)

epsilon = 0.03

mlp_clean, mlp_adv = evaluate_fgsm(
    mlp_model,
    "MLP",
    X_robust,
    y_robust,
    y_robust_cat,
    epsilon
)

cnn_clean, cnn_adv = evaluate_fgsm(
    cnn_model,
    "CNN",
    X_robust,
    y_robust,
    y_robust_cat,
    epsilon
)

# Transfer model expects raw 0-255 input due to preprocessing layer usage
X_robust_transfer = X_test_raw[robust_indices].astype("float32")
y_robust_transfer_cat = to_categorical(y_robust, 10)

# For FGSM with transfer model, clip to valid image range [0,255]
def fgsm_attack_transfer(model, images, labels_cat, epsilon_pixel):
    images_tensor = tf.convert_to_tensor(images)

    with tf.GradientTape() as tape:
        tape.watch(images_tensor)
        predictions = model(images_tensor)
        loss = tf.keras.losses.categorical_crossentropy(labels_cat, predictions)

    gradient = tape.gradient(loss, images_tensor)
    signed_grad = tf.sign(gradient)

    adversarial_images = images_tensor + epsilon_pixel * signed_grad
    adversarial_images = tf.clip_by_value(adversarial_images, 0, 255)

    return adversarial_images.numpy()


def evaluate_fgsm_transfer(model, X_clean, y_true, y_cat, epsilon_pixel):
    clean_probs = model.predict(X_clean, batch_size=64, verbose=0)
    clean_preds = np.argmax(clean_probs, axis=1)
    clean_acc = accuracy_score(y_true, clean_preds)

    X_adv = fgsm_attack_transfer(model, X_clean, y_cat, epsilon_pixel)

    adv_probs = model.predict(X_adv, batch_size=64, verbose=0)
    adv_preds = np.argmax(adv_probs, axis=1)
    adv_acc = accuracy_score(y_true, adv_preds)

    print("Transfer clean accuracy:", round(clean_acc, 4))
    print("Transfer FGSM accuracy:", round(adv_acc, 4))

    return clean_acc, adv_acc


transfer_clean, transfer_adv = evaluate_fgsm_transfer(
    transfer_model,
    X_robust_transfer,
    y_robust,
    y_robust_transfer_cat,
    epsilon_pixel=8.0
)

robustness_df = pd.DataFrame({
    "Model": ["MLP", "CNN", "Transfer Learning MobileNetV2"],
    "Clean Accuracy": [mlp_clean, cnn_clean, transfer_clean],
    "FGSM Accuracy": [mlp_adv, cnn_adv, transfer_adv],
    "Accuracy Drop": [
        mlp_clean - mlp_adv,
        cnn_clean - cnn_adv,
        transfer_clean - transfer_adv
    ]
})

print("\nFGSM robustness results:")
print(robustness_df)

robustness_df.to_csv("q5_fgsm_robustness_results.csv", index=False)

plt.figure(figsize=(8, 5))
x = np.arange(len(robustness_df["Model"]))
width = 0.35

plt.bar(x - width / 2, robustness_df["Clean Accuracy"], width, label="Clean")
plt.bar(x + width / 2, robustness_df["FGSM Accuracy"], width, label="FGSM")

plt.xticks(x, robustness_df["Model"], rotation=20, ha="right")
plt.ylabel("Accuracy")
plt.title("FGSM Robustness Comparison")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig("q5_fgsm_robustness_comparison.png", dpi=300)
plt.close()

print("\nRobustness files saved:")
print("q5_fgsm_robustness_results.csv")
print("q5_fgsm_robustness_comparison.png")