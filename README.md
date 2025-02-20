# Paddy Disease Classification

A deep learning-based classification model for detecting and identifying paddy rice leaf diseases using CNNs and TensorFlow.

## Dataset

The dataset consists of **10 classes** of paddy leaf diseases:
- Bacterial Leaf Blight
- Bacterial Leaf Streak
- Bacterial Panicle Blight
- Blast
- Brown Spot
- Dead Heart
- Downy Mildew
- Hispa
- Normal (Healthy Leaves)
- Tungro

Total Images: **10,407**  
Training Set: **8,326 images**  
Validation Set: **2,081 images**

## Model Architecture

The model is built using a **Convolutional Neural Network (CNN)** with the following layers:
- Rescaling layer (Normalization)
- Convolutional layers with ReLU activation
- MaxPooling layers
- Flatten layer
- Dropout layer
- Fully connected Dense layers with Softmax activation

## Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/paddy-disease-classification.git
   cd paddy-disease-classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the model training script:
   ```bash
   python train.py
   ```

## Training & Evaluation

- **Optimizer:** Adam
- **Loss Function:** Sparse Categorical Crossentropy
- **Metrics:** Accuracy
- **Early Stopping:** Patience of 30 epochs

### Performance
- The model was trained for **100 epochs** with early stopping.
- Training & validation accuracy and loss were plotted to visualize performance.

## Results

| Metric | Training | Validation |
|--------|----------|------------|
| Accuracy | ~42% | ~30% |
| Loss | High | Very High |

**Note:** The model is experiencing high loss values and needs hyperparameter tuning.

## Future Improvements

- Data augmentation to improve generalization
- Hyperparameter tuning (learning rate, dropout, number of filters, etc.)
- Using a pre-trained model (Transfer Learning)
- Increasing dataset size for better learning

## Contributing

Contributions are welcome! Feel free to **fork** this repository and submit a **pull request**.



**Author:** Muhammad Fakhar ul Hasnain

