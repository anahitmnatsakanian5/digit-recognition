"""Train a small CNN on MNIST, evaluate it, and save the model and plots."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras

# 1. Load and preprocess
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = (x_train / 255.0).astype("float32")[..., np.newaxis]
x_test = (x_test / 255.0).astype("float32")[..., np.newaxis]

# 2. Model
model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    keras.layers.Conv2D(32, 3, activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation="relu"),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# 3. Train
model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# 4. Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# 5. Save model
model.save("model.keras")

# 6. Plot sample predictions
probs = model.predict(x_test[:25])
preds = probs.argmax(axis=1)
fig, axes = plt.subplots(5, 5, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_test[i].squeeze(), cmap="gray")
    ax.set_title(f"pred {preds[i]} (true {y_test[i]})",
                 color="green" if preds[i] == y_test[i] else "red", fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.savefig("predictions.png", dpi=120)
plt.close()

# 7. Confusion matrix
all_preds = model.predict(x_test).argmax(axis=1)
cm = np.zeros((10, 10), dtype=int)
for t, p in zip(y_test, all_preds):
    cm[t, p] += 1
plt.figure(figsize=(7, 6))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.xticks(range(10)); plt.yticks(range(10))
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion matrix")
for i in range(10):
    for j in range(10):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=120)
print("Saved model.keras, predictions.png, confusion_matrix.png")
