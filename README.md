# 🚗💤 Drowsiness Detection using CNN

## 📌 Project Overview
This project is designed to detect driver drowsiness using **Convolutional Neural Networks (CNNs)**. It classifies images into **Drowsy** and **Non-Drowsy** categories to help prevent accidents caused by fatigue.

## 🏗️ Model Architecture
The model consists of **Convolutional Layers**, **MaxPooling**, **Dropout**, and **Data Augmentation** to improve accuracy and generalization.

### 📜 Model Summary:
- **Input Layer:** Image size **256x256x3**
- **Data Preprocessing:** Rescaling, Random Flipping, Rotation, and Zoom
- **Convolutional Layers:**
  - **Conv2D (32 filters, 3x3, ReLU, Same Padding) + MaxPooling (2x2)**
  - **Conv2D (64 filters, 3x3, ReLU, Same Padding) + MaxPooling (2x2)**
  - **Conv2D (128 filters, 3x3, ReLU, Same Padding) + MaxPooling (2x2)**
  - **Conv2D (128 filters, 3x3, ReLU, Same Padding) + MaxPooling (2x2)**
- **Flatten Layer**
- **Fully Connected Layer (Dense 256, ReLU, Dropout 0.5)**
- **Output Layer (Dense 1, Sigmoid Activation)**

### 🏗️ Model Code:
```python
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(256, 256, 3)),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),
    
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary Classification
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
model.summary()
```

## 🚀 Training & Evaluation
- **Training:** The model is trained using **Binary Crossentropy** loss and **Adam optimizer**.
- **Learning Rate Scheduler:** Reduces learning rate if validation loss stops improving.
- **Evaluation:** The model is tested on unseen data to measure accuracy.

```python
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    verbose=1
)

history = model.fit(
    train_ds,
    epochs=50,
    validation_data=val_ds,
    callbacks=[lr_scheduler]
)
```

## 💾 Model Saving & Loading
```python
model.save("models/drowsiness_model.keras")
```
To load the model:
```python
from tensorflow.keras.models import load_model
model = load_model("models/drowsiness_model.keras")
```

## 📸 Making Predictions
```python
import cv2
import numpy as np

img = cv2.imread("test_image.jpg")
img = cv2.resize(img, (256, 256)) / 255.0
img = np.expand_dims(img, axis=0)
prediction = model.predict(img)
print("Drowsy" if prediction > 0.5 else "Non-Drowsy")
```

---

## 📌 Future Enhancements
- Use **Transfer Learning** (MobileNetV2, ResNet)
- Deploy as a **Web App using FastAPI**
- Implement **real-time detection** using OpenCV

### 🎯 Contributions are Welcome!
Feel free to fork, improve, and submit PRs. 🚀

