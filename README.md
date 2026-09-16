# Fashion-MNIST Image Classification with Feedforward Neural Networks

An end-to-end deep learning project exploring image classification on the Fashion-MNIST dataset using TensorFlow and Keras. This project compares a baseline feedforward multilayer perceptron (MLP) against an enhanced, regularized architecture with data augmentation, evaluating where simple dense networks succeed and where spatial limitations emerge.

---

## Project Overview

- **Dataset**: Fashion-MNIST (70,000 28x28 grayscale images across 10 apparel categories)
- **Framework**: TensorFlow 2.x / Keras
- **Core Architecture**: Dense Feedforward Neural Network (Multilayer Perceptron)
- **Baseline Test Accuracy**: ~88.2%
- **Key Focus**: Evaluating non-linear feature representation on raw pixels vs. classical ML, error analysis across silhouette-overlapping apparel categories, and operational deployment in automated e-commerce cataloging.

---

## Repository Structure

```text
├── Fashion_MNIST_Neural_Network.ipynb   # Main Google Colab notebook (runnable)
├── README.md                            # Project documentation and setup instructions
└── requirements.txt                     # Python dependencies
```

---

## Pipeline Workflow

The end-to-end pipeline mirrors an automated e-commerce catalog tagging service:

1. **Upload & Ingestion**: Ingests raw pixel tensors (28x28 grayscale) directly via Keras datasets API.
2. **Preprocessing**: Rescales pixel values from [0, 255] down to [0.0, 1.0] and converts labels to 10-class one-hot vectors.
3. **Partitioning**: Allocates 50,000 samples for training, 10,000 for validation monitoring, and 10,000 isolated test samples.
4. **Baseline Architecture**:
   - `Flatten`: Transforms 28x28x1 inputs into a 784-dimensional vector.
   - `Dense(128, activation='relu')`: Learns non-linear representations across pixel intensities.
   - `Dense(10, activation='softmax')`: Outputs normalized class probabilities.
5. **Optimization**: Minimizes Categorical Cross-Entropy using the Adam optimizer with Early Stopping.
6. **Architectural Improvements**: Adds a secondary dense layer (256 -> 128), Batch Normalization, Dropout (p=0.3), and real-time data augmentation (horizontal flips, slight rotations, translations).

---

## Key Results & Takeaways

| Model Architecture | Hidden Layers | Regularization | Test Accuracy |
| :--- | :--- | :--- | :--- |
| **Baseline MLP** | 128 (ReLU) | Early Stopping | ~88.2% |
| **Enhanced MLP + Aug** | 256 -> 128 (ReLU) | Dropout (0.3) + BatchNorm + Aug | ~87.8% – 88.5% |

### Key Diagnostic Observations:
- **High Separation**: Structurally distinct classes (*Trouser*, *Bag*, *Ankle boot*) consistently achieve >95% recall.
- **Silhouette Ambiguity**: The network frequently confuses *Shirts*, *T-shirts*, *Pullovers*, and *Coats* due to overlapping contours in low-resolution (28x28) grayscale.
- **Spatial Inductive Bias**: Because a standard MLP flattens the input into a 1D array (784), it discards 2D spatial relationships. Distinguishing subtle localized details (such as buttons vs. crew necks) requires transitioning to Convolutional Neural Networks (CNNs).

---

## Installation & Local Setup

### Prerequisites
- Python 3.9+
- pip

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Set Up a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

*Or install core packages directly:*
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

### 4. Run the Notebook
Launch Jupyter Lab, Jupyter Notebook, or upload directly to [Google Colab](https://colab.research.google.com/):
```bash
jupyter notebook Fashion_MNIST_Neural_Network.ipynb
```

---

## Dependencies

- `tensorflow`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

---

## Author
Cesar Juarez-Vargas 

Developed as part of the academic curriculum for the Data Analytics and AI Program.
