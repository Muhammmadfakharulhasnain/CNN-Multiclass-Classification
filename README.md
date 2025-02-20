# Paddy Rice Disease Classification

This repository contains a deep learning model for classifying paddy rice diseases using TensorFlow and Keras. The model is trained on the **Paddy Doctor** dataset, which consists of images of rice plants affected by various diseases.

## Dataset
The dataset used for this project can be found on Kaggle:
[Paddy Doctor Dataset](https://www.kaggle.com/datasets/imbikramsaha/paddy-doctor)

It consists of 10 different classes of rice diseases:
- Bacterial Leaf Blight
- Bacterial Leaf Streak
- Bacterial Panicle Blight
- Blast
- Brown Spot
- Dead Heart
- Downy Mildew
- Hispa
- Normal (Healthy)
- Tungro

## Model Architecture
The model is a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The architecture includes:
- Image rescaling
- Four convolutional layers with ReLU activation
- Max pooling layers
- Flatten layer
- Dropout layer (to prevent overfitting)
- Dense layers with softmax activation for classification

## Training
The model was trained with:
- **Optimizer:** Adam
- **Loss Function:** Sparse Categorical Crossentropy
- **Metrics:** Accuracy
- **Epochs:** 100 (with early stopping to prevent overfitting)

## Results
During training, the model's performance was monitored using accuracy and loss. The validation accuracy fluctuated, and further improvements are needed to enhance performance.

## Usage
To use this repository:
1. Clone the repo:
   ```sh
   git clone https://github.com/yourusername/paddy-rice-disease-classification.git
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the training script:
   ```sh
   python train.py
   ```

## Future Improvements
- Fine-tuning the model with data augmentation
- Experimenting with different architectures (e.g., Transfer Learning)
- Hyperparameter tuning

## Contributors
Muhammad Fakhar ul Hasnain

---
Feel free to contribute and improve this project!

