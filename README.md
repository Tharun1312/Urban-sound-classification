# Urban Sound Classification using CNN

This project builds a deep learning pipeline to classify real‑world urban sounds such as dog barks, sirens, drilling, and other city noises into 10 classes. It uses the UrbanSound8K dataset and a 2D Convolutional Neural Network (CNN) trained on MFCC‑based spectrograms for robust audio event recognition.[file:16]

## Project Overview

- Uses the UrbanSound8K dataset with 8,732 labeled audio clips across 10 urban sound categories.  
- Performs end‑to‑end processing: loading audio files, extracting MFCC features, preparing train/validation/test splits, and normalising data.  
- Trains a 2D CNN on MFCC feature maps to learn discriminative temporal–spectral patterns.  
- Evaluates performance with accuracy, loss curves, confusion matrix, and classification metrics to analyse class‑wise behaviour.[file:16]

## Model Performance

- Achieved around **90% test accuracy** on the UrbanSound8K dataset with the final CNN configuration.  
- Confusion matrix and evaluation reports show strong performance on most classes, with misclassifications primarily between acoustically similar sounds (for example, certain mechanical or background noise classes).[file:16]  
- The model and code are organised so that hyperparameters, architecture depth, and feature settings can be tuned easily for further improvements.

## Tech Stack

- **Language:** Python  
- **Libraries:** TensorFlow / Keras, Librosa, NumPy, Pandas, Matplotlib, Scikit‑learn  
- **Environment:** Jupyter Notebook  

This combination supports audio preprocessing, model design, training, visualisation, and detailed evaluation in a single workflow.[file:16]

## Repository Structure

- `Urban-Sound-Classification-Using-Convolutional-Neural-Networks/`  
  - Main Jupyter Notebook containing:
    - Dataset loading and exploration  
    - Feature extraction (MFCCs and related parameters)  
    - CNN model definition  
    - Training loop and callbacks  
    - Evaluation, plots, and confusion matrix  
- `.gitignore`  
  - Excludes large dataset files, temporary artifacts, and notebook checkpoints to keep the repository clean.[web:1]

Update the folder/notebook name above if it differs in your repo.

## Getting Started

### Prerequisites

- Python 3.x  
- Jupyter Notebook or JupyterLab  
- Installed Python packages listed in the Tech Stack  

### Installation

1. Clone the repository:
git clone https://github.com/Tharun1312/Urban-sound-classification.git
cd Urban-sound-classification

text
2. (Optional) Create and activate a virtual environment.  
3. Install dependencies:
pip install numpy pandas librosa matplotlib scikit-learn tensorflow



### Dataset Setup

1. Download the **UrbanSound8K** dataset from its official source.  
2. Extract it to a folder on your system (for example, `UrbanSound8K/`).  
3. Open the notebook and set the dataset path variable so it points to your `UrbanSound8K` directory.[file:16]

## Running the Notebook

1. Start Jupyter:
jupyter notebook


2. Open the main notebook in the `Urban-Sound-Classification-Using-Convolutional-Neural-Networks` directory.  
3. Run the cells in order to:
- Load and preprocess the audio data  
- Extract MFCC features and build feature tensors  
- Define and compile the CNN model  
- Train the model and monitor accuracy/loss  
- Evaluate the model, generate metrics, and plot the confusion matrix  

## Future Enhancements

- Experiment with deeper or alternative CNN architectures, or add residual connections to capture more complex patterns.  
- Use additional audio representations such as log‑mel spectrograms or chroma features and compare performance.  
- Add audio data augmentation (noise, time‑shifting, pitch/tempo changes) for better generalisation.  
- Deploy the trained model as a small web app or API to perform real‑time urban sound classification from the microphone.  

## Author

**Tharun Chinthala**  
B.Tech Computer Science, SR University  
Focus areas: machine learning, audio signal processing, and full‑stack development.[file:16]
