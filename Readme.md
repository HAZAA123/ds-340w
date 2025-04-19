# Malaria Detection Using Hybrid Deep Learning Models

## Project Overview
This repository contains the implementation of hybrid deep learning models for automated detection of malaria parasites in microscopic blood smear images. The focus is on combining Convolutional Neural Networks (CNNs) with different types of Recurrent Neural Networks (RNNs) to enhance classification accuracy and efficiency.

## Key Features
- Implementation of three hybrid architectures:
  - CNN-LSTM-LSTM
  - CNN-GRU-GRU
  - CNN-BiLSTM-BiLSTM
- Enhanced image resolution (64×64 pixels)
- Data augmentation techniques
- Comparative analysis of model performance

## Dataset
The project uses the NIH Malaria Dataset containing 27558 cell images with equal instances of parasitized and uninfected cells.

## File Description
- `Mid_Malaria_Detection_Paper.ipynb`: Jupyter notebook containing the complete implementation, including data preprocessing, model building, training, and evaluation.

## Results
The CNN-GRU-GRU model demonstrated superior performance with:
- 95.65% accuracy on the test dataset
- 99% precision for parasitized cells
- 93% recall for uninfected cells

## Requirements
- Python 3.8+
- TensorFlow 2.8+
- Keras
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn

## How to Run
1. Clone this repository
2. Install the required packages
3. Run the Jupyter notebook

## Future Work
- Implementation of attention mechanisms
- Exploration of transfer learning with modern architectures
- Investigation of multimodal approaches combining image data with clinical metadata

## Author
Hazaa Albreiki  
College of Information Sciences/Technology  
The Pennsylvania State University  
Email: Hra5155@psu.edu