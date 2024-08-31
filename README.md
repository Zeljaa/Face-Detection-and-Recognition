# Face-Detection-and-Recognition

## Project Overview

This project implements a face recognition system, which involves two key steps: face detection and face recognition.

1. **Face Detection**: A YOLOv5-based model is used to detect faces within images. The model has been fine-tuned specifically for face identification.

2. **Face Recognition**: The recognition process uses a model that leverages 128-dimensional embeddings for identifying faces. Built on a fine-tuned ResNet50, this model does not require retraining when adding new faces; instead, it calculates embeddings, which are then compared during inference. If the distance between the new face and known embeddings exceeds a certain threshold, the face is classified as "Unknown." The model was trained on a diverse dataset with applied data augmentation techniques like rotation and lighting adjustments to enhance robustness.


**GUI**: Also a user-friendly GUI has been developed to integrate all components, allowing for easy interaction with the face detection and recognition processes.


## Usage

### Face Recognition (folder_name: FaceRecognition)
This folder contains the following components:

1. **FaceEmbeddings File**
   - This file includes the model used for calculating embeddings from faces. Below is an example of how to use the `FaceEmbeddings` class:

   ```python
   from FaceEmbeddings import FaceEmbeddings
   from torchvision import transforms
   from PIL import Image
   import torch

   # Initialize the FaceEmbeddings model
   model = FaceEmbeddings()

   # Path to the image
   image_path = 'face.jpg'

   # Define the image transformations
   transform = transforms.Compose([
       transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
       transforms.ToTensor(),          # Convert the image to a tensor
   ])

   # Open and process the image
   image = Image.open(image_path).convert('RGB')
   input_tensor = transform(image)

   # Add a batch dimension (1, 3, 224, 224)
   input_tensor = input_tensor.unsqueeze(0)

   # Compute the embedding
   with torch.no_grad():
       embedding = model(input_tensor).numpy()

   print(embedding)
    
2. **StoreNewEmbeddings File**
   - This file is used for creating embeddings from face images and storing them in a database in pickle format (`.pkl`). These stored embeddings are later used during face recognition.

   The structure of the Pickle (`.pkl`) file is as follows:
   - A list of tuples: `[(embedding, class_name), ...]`
     - `embedding`: A NumPy array representing the 128-dimensional face embedding.
     - `class_name`: The label or name associated with the embedding.

   Additionally, there is a `class_names` list, which contains all the unique names present in the database. This setup facilitates efficient face recognition by enabling quick comparisons between stored embeddings and new face inputs.


### Face Detection (folder_name: yoloface)
This folder contains a downloaded repository from [elyha7](https://github.com/elyha7/yoloface), whose YOLO-based model and pre-trained weights have been utilized for face detection in this project.


### GUI
The `gui.py` file integrates the face detection model from `model.py`, where the face detection is performed, and the embeddings for face recognition are calculated. These embeddings are then compared to the stored database to find the closest match.

The GUI also includes an option to save detected faces. You can enable this feature at the beginning of the script by modifying the relevant setting.

**How to Run the GUI:**
  ```bash
  python gui.py
  ```



Make sure you have all the required dependencies installed and the models properly set up before running the scripts.
