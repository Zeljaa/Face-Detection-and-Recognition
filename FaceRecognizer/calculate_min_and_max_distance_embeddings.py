# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 00:02:25 2024

@author: Vladimir Zeljkovic
"""



import pickle
#import torch
import numpy as np

from scipy.spatial import distance

def separate_embeddings(known_embeddings):
    """Separate embeddings by class."""
    class_embeddings = {}
    for emb, class_name in known_embeddings:
        emb = np.squeeze(emb)  # Remove single-dimensional entries
        if class_name not in class_embeddings:
            class_embeddings[class_name] = []
        class_embeddings[class_name].append(emb)
    return class_embeddings

def find_min_distance_between_classes(class_embeddings):
    """Find the minimum distance between embeddings from different classes."""
    min_distance = float('inf')
    classes = list(class_embeddings.keys())
    
    # Compare embeddings from each pair of different classes
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            class1 = classes[i]
            class2 = classes[j]
            
            embeddings_class1 = np.array(class_embeddings[class1])
            embeddings_class2 = np.array(class_embeddings[class2])
            
            # Debugging output
            print(f"Class1: {class1}, Number of embeddings: {embeddings_class1.shape[0]}")
            print(f"Class2: {class2}, Number of embeddings: {embeddings_class2.shape[0]}")
            
            if embeddings_class1.size == 0 or embeddings_class2.size == 0:
                print(f"Warning: One of the classes has no embeddings.")
                continue

            # Ensure embeddings are 2D with shape (num_embeddings, num_features)
            if embeddings_class1.ndim == 1:
                embeddings_class1 = embeddings_class1[None, :]
            if embeddings_class2.ndim == 1:
                embeddings_class2 = embeddings_class2[None, :]
            
            # Compute pairwise distances between embeddings of different classes
            dist_matrix = distance.cdist(embeddings_class1, embeddings_class2, metric='euclidean')
            min_dist = np.min(dist_matrix)
            
            if min_dist < min_distance:
                min_distance = min_dist
    
    return min_distance

def find_max_distance_within_classes(class_embeddings):
    """Find the maximum distance between embeddings of the same class."""
    max_distance = float('-inf')
    classes = list(class_embeddings.keys())
    
    # Compare embeddings within each class
    for class_name in classes:
        embeddings = np.array(class_embeddings[class_name])
        
        # Debugging output
        print(f"Class: {class_name}, Number of embeddings: {embeddings.shape[0]}")
        
        if embeddings.shape[0] < 2:
            # Skip classes with less than 2 embeddings
            print(f"Skipping class {class_name} due to insufficient embeddings.")
            continue
        
        # Ensure embeddings are 2D with shape (num_embeddings, num_features)
        if embeddings.ndim == 1:
            embeddings = embeddings[None, :]
        
        # Compute pairwise distances between embeddings of the same class
        dist_matrix = distance.pdist(embeddings, metric='euclidean')
        max_dist = np.max(dist_matrix)
        
        if max_dist > max_distance:
            max_distance = max_dist
    
    return max_distance

def calculate_min_and_max_distance_within_embeddings(base='try.pkl'):
    
    with open(base, 'rb') as f:
        known_embeddings, class_names = pickle.load(f)
        
    class_embeddings = separate_embeddings(known_embeddings)
    min_distance = find_min_distance_between_classes(class_embeddings)
    print(f"Minimum distance between embeddings from different classes: {min_distance}")
    max_distance = find_max_distance_within_classes(class_embeddings)
    print(f"Maximum distance between embeddings of the same class: {max_distance}")
    
    
if __name__=='__main__':
    
    calculate_min_and_max_distance_within_embeddings()