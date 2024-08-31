# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 23:10:18 2024

@author: Vladimir Zeljkovic
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import os

class FaceEmbeddings(nn.Module):
    def __init__(self, device='cpu', weights='weights/face_recognition_model_embeddings1.pth', embedding_size = 128):
        super(FaceEmbeddings, self).__init__()
        self.device = torch.device(device)
        self.resnet = resnet50(weights=None)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.fc = nn.Linear(2048, embedding_size)
        self._initialize_weights(weights)

    def _initialize_weights(self, weights):
        """Load pre-trained weights if available."""
        weights = os.path.join(os.path.dirname(os.path.abspath(__file__)), weights)
        if os.path.exists(weights):
            self.load_state_dict(torch.load(weights, map_location=self.device))
            self.eval()
        else:
            print("Pretrained weights not found, model initialized with random weights.")
        
    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)  # Normalize embeddings
        return x


if __name__=='__main__':
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FaceEmbeddings()