# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 23:02:44 2024

@author: Vladimir Zeljkovic
"""

import sys
sys.path.append('yoloface/')
sys.path.append('FaceRecognizer/')

import torch
import pickle

import numpy as np
from PIL import Image
from torchvision import transforms

from face_detector import YoloDetector
from FaceEmbeddings import FaceEmbeddings


def face_detection(img):    
    global detector
    return detector.predict(np.array(img))    




def face_recognition(image):
    with open('known_embeddings1.pkl', 'rb') as f:
        known_embeddings, class_names = pickle.load(f)
        
    global model
    threshold = 0.9
    #image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
        transforms.ToTensor(),           # Convert the image to a tensor
    ])
    
    # Apply transformations to the image
    input_tensor = transform(image)
    
    # Add a batch dimension (1, 3, 224, 224)
    input_tensor = input_tensor.unsqueeze(0).to(device)
    model.eval()
    # Compute the embedding
    with torch.no_grad():
        embedding = model(input_tensor).cpu().numpy()
    
    embedding = embedding.flatten()  # Flatten the embedding to ensure it's 1D

    # Convert known embeddings to 1D arrays
    known_embeddings = [(np.squeeze(np.array(embedding)),class_name) for embedding, class_name in known_embeddings]

    # Compute distances
    distances = [np.linalg.norm(embedding - known_embedding[0]) for known_embedding in known_embeddings]
    
    # Debugging statements
    print(f"Distances: {distances}")
    print(f"Class names: {class_names}")
    
    if not distances:
        return "Unknown"  # or some other fallback

    min_distance = min(distances)
    min_index = distances.index(min_distance)
    
    print(min_distance)
    
    if min_distance > threshold:
        return "Unknown"
    else:
        return known_embeddings[min_index][1]
        


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize the face detector model
detector = YoloDetector(target_size=720, device=device, min_face=90)
model = FaceEmbeddings()

