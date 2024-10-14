# 🛰️ Satellite Image Classification using CNNs 🌍

This project uses **Convolutional Neural Networks (CNNs)** to classify satellite images into different categories. The dataset consists of satellite imagery, and the model is trained to differentiate between various classes using data augmentation, multiple layers of CNNs, and modern deep learning techniques.

## 📁 Dataset

We are using a **Satellite Image Classification** dataset, which can be downloaded from Kaggle using the following link: 

[Kaggle Dataset: Satellite Image Classification](https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification) 🌐

The dataset contains satellite images stored in directories, representing different categories (for example, agricultural, forest, etc.).

## ⚙️ Steps to Run the Code

### 1. Environment Setup 🛠️

Before running the code, you need to have the following packages installed:
- `keras`
- `tensorflow`
- `numpy`
- `matplotlib`

You can install the necessary libraries using the following command:

```bash
pip install keras tensorflow numpy matplotlib
```

### 2. Downloading the Dataset 📦

To download the dataset from Kaggle, follow these steps:
- Upload your `kaggle.json` API key.
- Use the following command to download the dataset:

```bash
!kaggle datasets download -d mahmoudreda55/satellite-image-classification
```

After downloading, unzip the dataset to access the images for training and validation.

### 3. Model Architecture 🧠

The model used in this project is a sequential CNN with multiple layers:

- **Input Layer**: The model accepts images of size 224x224x3 (height, width, color channels).
- **Convolution Layers**: We use 4 convolutional layers with filters of increasing depth (32, 64, 128, 256), and ReLU activation functions.
- **Pooling Layers**: MaxPooling layers are added to downsample the feature maps.
- **Dropout**: To prevent overfitting, a 40% dropout is applied before the output layer.
- **Global Average Pooling**: Replaces flattening for dimensionality reduction.
- **Output Layer**: A softmax layer with 4 units for multi-class classification.

### 4. Data Augmentation 📈

The training images undergo various transformations to enhance model generalization:
- Horizontal and vertical flips
- Shearing
- Zooming
- Shifting

These transformations help in creating a more robust model that can handle a variety of satellite imagery variations.

### 5. Training the Model 🚀

The model is compiled using the **Adam optimizer** with a learning rate of 0.0001 and categorical cross-entropy as the loss function.

```python
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=metrics)
```

We use **early stopping** to monitor validation loss and stop training once the performance starts to degrade.

```python
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
```

The training process is set to run for 30 epochs with the option to stop early.

### 6. Evaluation 📊

After training, the model is evaluated on the validation data. Key metrics used are:
- **Accuracy**: Measures the percentage of correct predictions.
- **AUC (Area Under the Curve)**: Evaluates the model’s ability to differentiate between classes.

```python
val_loss, val_accuracy, val_auc = model.evaluate(validation_generator, verbose=0)
```

### 7. Results Visualization 📈

We plot the training and validation loss and accuracy to understand the model's performance over epochs.

```python
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
```

### 8. Fine-Tuning with Pretrained NASNetMobile 📊

In addition to training a CNN from scratch, we fine-tune a **NASNetMobile** pretrained model. The top layers are retrained on the satellite image dataset, with the lower layers frozen to retain learned features from the ImageNet dataset.

---

## 📊 Model Performance

After training, the model is evaluated on accuracy, loss, and AUC. The model can be further fine-tuned to improve performance using techniques like data augmentation, learning rate scheduling, and transfer learning.

---

## 🛠️ How to Use the Code

1. **Prepare your environment**: Make sure you have all necessary libraries installed.
2. **Download the dataset**: Use the provided Kaggle API link to download the dataset.
3. **Run the training code**: Execute the training process and monitor the output for model performance.
4. **Evaluate and plot results**: Use the plotting functions to visualize the training performance.

---

## 🎯 Future Work

- Experiment with different architectures like ResNet, VGG16, and EfficientNet.
- Try out hyperparameter tuning for improved performance.
- Implement Grad-CAM to visualize which parts of the image contribute to the classification decision.

---

## 📄 License

This project is licensed under the MIT License.

---
