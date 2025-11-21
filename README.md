# Cats-vs-Dogs-Image-Classification-with-Data-Augmentation
This repository contains a notebook that trains a convolutional neural network (CNN) to classify images of cats vs dogs, with a strong focus on data augmentation to improve generalization and robustness.  The project is ideal as a deep learning starter project and a portfolio piece for computer vision and Keras/TensorFlow skills.

The core of the project is the notebook `Cats_v_Dogs_Augmentation.ipynb`, which shows:
- How to **load and preprocess image data**  
- How to apply **data augmentation** to reduce overfitting  
- How to **build, train, and evaluate** a CNN for binary classification  
- How augmentation impacts model performance compared to a non-augmented baseline  

It is meant as a **practical introduction** to deep learning for computer vision.

## What the Notebook Does
# 1. Loads image data from train/val/test
# 2. Applies data augmentation:
#      - random flip
#      - rotation
#      - zoom
#      - shift/shear
# 3. Builds a CNN model for binary classification
# 4. Trains the model on augmented images
# 5. Evaluates accuracy and loss on validation set
# 6. Plots training curves (accuracy + loss)
# 7. Tests predictions on unseen images

## Project Structure
Cats_v_Dogs_Augmentation.ipynb   # Main notebook: data, model, training, evaluation
README.md                        # Project documentation

## Optional Improvements
# Try transfer learning:
#   - VGG16
#   - MobileNetV2
#   - ResNet50
#
# Add regularization:
#   - Dropout
#   - L2 weight decay
#
# Add TensorBoard logging:
#   tensorboard --logdir logs/
#
# Build a prediction demo:
#   - Streamlit
#   - Gradio

## Dataset
Downloaded from Kaggle: https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset 

## Install Dependencies
pip install tensorflow keras numpy matplotlib pandas scikit-learn jupyter
