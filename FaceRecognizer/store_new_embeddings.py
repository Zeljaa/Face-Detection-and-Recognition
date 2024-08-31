# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 13:37:20 2024

@author: Vladimir Zeljkovic
"""

from FaceEmbeddings import FaceEmbeddings
import pickle
from PIL import Image
from torchvision import transforms
import torch

def add_new_embeddings(image_path, class_name, base='try.pkl'):
    
    try:
        image = Image.open(image_path).convert('RGB')
        print("Image opened successfully")
    except:
        print("Error opening Image");
        return
    try:
        with open(base, 'rb') as f:
            known_embeddings, class_names = pickle.load(f)
            print("File opened and data loaded successfully.")
    except FileNotFoundError:
        class_names = []
        known_embeddings = []
        print("The file was not found, it will be created new pickle file")
        
    
    try:
        model = FaceEmbeddings()
        model.eval()
        print("Model loaded successfully")
    except:
        print("Error with loading Model")
        return  
        
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
        transforms.ToTensor(),           # Convert the image to a tensor# Normalize as done for ImageNet
    ])
    
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        embedding = model(input_tensor).detach().numpy()
        
    if class_name not in class_names:
        class_names.append(class_name)
    
    known_embeddings.append((embedding,class_name))
    
    with open(base, 'wb') as f:
        pickle.dump((known_embeddings, class_names), f)
        
    


if __name__=='__main__':
    
    add_new_embeddings('aki_0.jpg', 'Aki')

    