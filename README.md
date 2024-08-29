# Face-Detection-and-Recognition

## Project Overview

This project implements a face recognition system, which involves two key steps: face detection and face recognition.

1. **Face Detection**: A YOLOv5-based model is used to detect faces within images. The model has been fine-tuned specifically for face identification.

2. **Face Recognition**: The recognition process uses a model that leverages 128-dimensional embeddings for identifying faces. Built on a fine-tuned ResNet50, this model does not require retraining when adding new faces; instead, it calculates embeddings, which are then compared during inference. If the distance between the new face and known embeddings exceeds a certain threshold, the face is classified as "Unknown." The model was trained on a diverse dataset with applied data augmentation techniques like rotation and lighting adjustments to enhance robustness.


**GUI**: Also a user-friendly GUI has been developed to integrate all components, allowing for easy interaction with the face detection and recognition processes.

