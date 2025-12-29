# Brain Tumor Classification using CNN

This project presents a Convolutional Neural Network (CNN)-based approach for brain tumor classification from MRI scans, implemented using Tensorflow/Keras. The model classifies grayscale MRI images into four clinically relevant categories: Glioma, Meningioma, Pituitary (tumors), and Nontumor. The primary objective is to evaluate how effectively a deep-learning model can distinguish between visually similar brain tumor types using structural MRI data.  

## Repository Structure

├── Brain_tumor_SamayAsubadin.py     # Model training script  
├── model_analysis.ipynb             # Evaluation and visualization  
├── images/                          # Result figures  
├── requirements.txt                 # Dependencies  
├── .gitignore                       # Ignored files  
└── README.md

The repository is structured to keep training logic in a clean, reproducible .py script, and an exploratory analysis and visualization in a separate Jupyter notebook.  

## Model Architecture
- Framework: TensorFlow / Keras
- Input: 224×224 grayscale MRI images
- Architecture:
    -3 convolutional blocks (Conv2D + MaxPooling).
    - Fully connected layers with dropout generalization.
- Optimizer: Adam.
- Output: Softmax layer with 4 classes.
- Loss Function: Categorical cross-entropy.
- Evaluation: Accuracy, Precision, Recall, F1-score, Confusion Matrix.

## Dataset
The dataset is NOT included in this repository, but similar datasets are publicly available in Kaggle, containing labeled images for four classes: glioma, meningioma, pituitary tumor, and non-tumor. The dataset came preorganized into train and test directories, with the following class-wise distribution:
- Training Set:
    Glioma: 1,321 images
    Meningioma: 1,339 images
    Non-tumor: 1,596 images
    Pituitary: 1,457 images

-Testing Set:
    Glioma: 299 images
    Meningioma: 305 images
    Non-tumor: 404 images
    Pituitary: 299 images

In total, the dataset used contains 7,020 training images and 1,307 testing images. During training, 20% of the training set is further used as a validation subset via Keras’ _ImageDataGenerator_ to monitor generalization performance and apply early stopping.

Expected directory structure:
data/
├── Train/
    ├── glioma/
    ├── meningioma/
    ├── nontumor/
    └── pituitary/
└── Test/
    ├── glioma/
    ├── meningioma/
    ├── nontumor/
    └── pituitary/

## Results
### Training Set Class Distribution
The training dataset exhibits a moderately imbalanced class distribution. Non-tumor images constitute the largest class, while glioma and meningioma samples are slightly underrepresented. Pituitary tumor images lie between these extremes. The imbalance is moderate enough to be handled without explicit class reweighting, allowing the model’s performance to be evaluated under practical conditions.
![Training Set Class Distribution](images/class_distribution.png)

Additionally, random samples are visualized to qualitatively inspect image quality, contrast, and inter-class variability. This can be found in the separate model_analysis.ipynb notebook. 

### Classification Performance
The trained CNN achieves an overall test accuracy of **88%**.
Class-wise performance is summarized below:

| Class        | Precision | Recall | F1-score |
|--------------|-----------|--------|----------|
| Glioma       | 0.94      | 0.85   | 0.89     |
| Meningioma   | 0.81      | 0.68   | 0.74     |
| Nontumor     | 0.88      | 0.98   | 0.92     |
| Pituitary    | 0.90      | 0.99   | 0.94     |
| **Overall**  |           |        | **0.88** |


### Confusion Matrix
The confusion matrix shows strong overall classification performance, with most predictions concentrated along the diagonal. Non-tumor and pituitary cases are classified with particularly high accuracy, while most misclassifications occur between glioma and meningioma tumors.
![Confusion Matrix](images/confusion_matrix.png)

### Predicted vs Actual Labels
Qualitative assessment of the model’s performance can be visualized in the model_analysis.ipynb notebook, and was done by comparing predicted labels against ground-truth annotations for randomly selected MRI scans. Correct predictions are highlighted in green, while misclassifications are shown in red, allowing for immediate visual identification of success and failure cases. This visualization complements quantitative metrics by exposing individual prediction behavior, supporting transparent evaluation of model reliability in clinical-like scenarios.
![Predicted vs Actual Labels](images/predictions.png)

### Training Curves
The training curves show a steady increase in training accuracy accompanied by a continuous decrease in training loss, indicating that the model is effectively learning patterns from the training data. In contrast, validation accuracy improves initially but then plateaus, while validation loss begins to increase after a few epochs. This divergence suggests the onset of overfitting, where the model continues to fit the training data more closely without achieving corresponding improvements on unseen data.
![Training and Validation Curves](images/training_curves.png)

## How to Run
1. Install dependencies
```bash
pip install -r requirements.txt
```
2. Train and evaluate the model
```bash
python Brain_tumor_SamayAsubadin.py
```
3. Open model_analysis.ipynb to explore dataset distribution, check random samples, visualize predicted vs. actual sample classification, confusion matrix visualization, and training curves.

## Reproducibility
Due to randomness in weight initialization and data shuffling, exact results may vary slightly across runs.
