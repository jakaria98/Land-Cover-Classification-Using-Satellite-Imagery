### README: Land Use Classification using Transfer Learning (VGG16 & InceptionV3)

#### Project Overview:
This project focuses on classifying land use types from satellite imagery using deep learning techniques. The dataset used for this task is the UCMerced LandUse dataset, consisting of 21 different land use classes. The project implements transfer learning with two pre-trained models, **VGG16** and **InceptionV3**, and evaluates their performance using various metrics.

#### Key Features:
- **Transfer Learning** with pre-trained VGG16 and InceptionV3 models.
- **Data Augmentation** to improve the model's generalization ability.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, True Positive Rate (TPR), and False Positive Rate (FPR).
- **Visualizations** for training history, precision-recall, confusion matrix, and classification report.
- **Model Saving**: Both models (VGG16 and InceptionV3) are saved in JSON and weights format, as well as serialized using pickle.

#### Dataset:
- **UCMerced LandUse Dataset**: The dataset contains 21 classes such as agricultural, airplane, beach, buildings, forest, etc. Each class contains a set of satellite images.
  
  **Download the dataset** from the official repository: [UCMerced LandUse Dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html)
  
  **Dataset Organization**: After downloading, organize the dataset into `Train` and `Test` folders with subfolders for each class:

  ```
  ├── Train
  │   ├── agricultural
  │   ├── airplane
  │   ├── beach
  │   └── ...
  └── Test
      ├── agricultural
      ├── airplane
      ├── beach
      └── ...
  ```

  Each subfolder contains images of the respective class.

#### Dependencies:
The following Python libraries are required:
- `tensorflow`
- `numpy`
- `matplotlib`
- `pandas`
- `scikit-learn`
- `pickle`

You can install the dependencies by running:

```bash
pip install tensorflow numpy matplotlib pandas scikit-learn
```

#### Training and Testing Workflow:
1. **Data Loading**: The `load_data()` function is responsible for loading images from the dataset directory and organizing them into training and testing sets. Images are resized to (256x256) and normalized.

2. **Data Augmentation**: The project applies data augmentation (rotation, zoom, flip, etc.) using `ImageDataGenerator` to improve the model's generalization.

3. **Model Architecture**:
   - **VGG16 Model**: A pre-trained VGG16 model is used as the base, followed by additional layers (Flatten, Dense, Dropout, BatchNormalization).
   - **InceptionV3 Model**: A similar approach is used with the InceptionV3 model as the base.

4. **Training**: 
   - Models are trained for 10 epochs using `Adam` optimizer and categorical cross-entropy loss.
   - Training and validation accuracy and loss are monitored, and results are stored in the history object.

5. **Evaluation**: Both models are evaluated on the test dataset. The project computes and visualizes metrics like accuracy, precision, recall, F1-score, TPR, FPR, and confusion matrix.

