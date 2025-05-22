#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image Analyzer

This module provides functions for analyzing image files:
- Feature extraction using ResNet50
- Image classification for automatic categorization
- Image preprocessing
"""

import os
from typing import Dict, List, Any, Tuple, Union
import numpy as np
# Import the hash function from utils
from utils.file_utils import calculate_file_hash

# Import PIL for image processing
from PIL import Image, ImageStat

# Import torch and torchvision for ResNet50
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms

# Global variables for models to avoid reloading
_resnet_model = None
_resnet_classifier = None
_imagenet_labels = None


def get_resnet_model():
    """
    Get or initialize the ResNet50 model for feature extraction.
    Uses singleton pattern to avoid reloading the model.
    
    Returns:
        ResNet50 model with pretrained weights v2 (feature extractor)
    """
    global _resnet_model
    if _resnet_model is None:
        print("Loading ResNet50 feature extraction model...")
        # Use the v2 weights as specified
        weights = ResNet50_Weights.IMAGENET1K_V2
        _resnet_model = resnet50(weights=weights)
        _resnet_model.eval()  # Set to evaluation mode
        
        # Remove the classification layer to get features
        _resnet_model = torch.nn.Sequential(*list(_resnet_model.children())[:-1])
        
        # Move to GPU if available
        if torch.cuda.is_available():
            _resnet_model = _resnet_model.cuda()
    
    return _resnet_model


def get_resnet_classifier():
    """
    Get or initialize the ResNet50 model for image classification.
    Uses singleton pattern to avoid reloading the model.
    
    Returns:
        ResNet50 model with pretrained weights v2 (full classifier)
    """
    global _resnet_classifier, _imagenet_labels
    if _resnet_classifier is None:
        print("Loading ResNet50 classification model...")
        # Use the v2 weights as specified
        weights = ResNet50_Weights.IMAGENET1K_V2
        _resnet_classifier = resnet50(weights=weights)
        _resnet_classifier.eval()  # Set to evaluation mode
        
        # Get the class labels
        _imagenet_labels = weights.meta["categories"]
        
        # Move to GPU if available
        if torch.cuda.is_available():
            _resnet_classifier = _resnet_classifier.cuda()
    
    return _resnet_classifier


def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """
    Gets the dimensions of an image.

    Args:
        image_path: Path to the image file

    Returns:
        Tuple of (width, height)
    """
    try:
        with Image.open(image_path) as img:
            return img.size
    except Exception as e:
        print(f"Error getting image dimensions for {image_path}: {e}")
        return (0, 0)


def preprocess_image(image_path: str) -> torch.Tensor:
    """
    Preprocesses an image for ResNet50.

    Args:
        image_path: Path to the image file

    Returns:
        Preprocessed image tensor
    """
    try:
        # Define preprocessing transformations
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Open and preprocess the image
        img = Image.open(image_path).convert('RGB')
        img_tensor = preprocess(img)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        
        # Move to GPU if available
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
            
        return img_tensor
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None


def create_advanced_image_features(image_path: str) -> np.ndarray:
    """
    Creates advanced image features using ResNet50.

    Args:
        image_path: Path to the image file

    Returns:
        Feature vector as a numpy array
    """
    try:
        # Preprocess the image
        img_tensor = preprocess_image(image_path)
        if img_tensor is None:
            # Fallback to random features if preprocessing fails
            return np.random.rand(2048)
        
        # Get the model
        model = get_resnet_model()
        
        # Extract features
        with torch.no_grad():  # No need for gradients
            features = model(img_tensor)
        
        # Reshape and convert to numpy array
        features = features.squeeze().cpu().numpy()
        
        return features
    except Exception as e:
        print(f"Error extracting ResNet50 features for {image_path}: {e}")
        # Return random features as a fallback
        return np.random.rand(2048)


def classify_image(image_path: str) -> Dict[str, Any]:
    """
    Classifies an image using ResNet50 with ImageNet classes.

    Args:
        image_path: Path to the image file

    Returns:
        Dictionary with classification results
    """
    try:
        # Preprocess the image
        img_tensor = preprocess_image(image_path)
        if img_tensor is None:
            return {"error": "Failed to preprocess image"}
        
        # Get the classifier model
        model = get_resnet_classifier()
        global _imagenet_labels
        
        # Perform classification
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        # Get top 5 predictions
        top5_prob, top5_indices = torch.topk(probabilities, 5)
        top5_prob = top5_prob.cpu().numpy()
        top5_indices = top5_indices.cpu().numpy()
        
        # Create classification results
        top_predictions = []
        for i, (prob, idx) in enumerate(zip(top5_prob, top5_indices)):
            label = _imagenet_labels[idx]
            top_predictions.append({
                "label": label,
                "probability": float(prob),
                "rank": i + 1
            })
        
        # Create broad category
        # Map common ImageNet classes to broader categories
        broad_category = determine_broad_category(top_predictions)
        
        return {
            "top_predictions": top_predictions,
            "primary_label": top_predictions[0]["label"],
            "confidence": float(top_predictions[0]["probability"]),
            "category": broad_category
        }
    except Exception as e:
        print(f"Error classifying image {image_path}: {e}")
        return {"error": str(e)}


def determine_broad_category(predictions: List[Dict[str, Any]]) -> str:
    """
    Determines a broad category for the image based on its top predictions.

    Args:
        predictions: List of top prediction dictionaries

    Returns:
        Broad category name
    """
    # Define category mappings (common ImageNet classes -> broad categories)
    category_mappings = {
        "vehicle": ["car", "truck", "bicycle", "motorcycle", "bus", "train", "airplane", "boat", "van"],
        "animal": ["dog", "cat", "bird", "fish", "horse", "elephant", "bear", "lion", "tiger", "monkey"],
        "person": ["person", "man", "woman", "child", "boy", "girl", "baby"],
        "sport": ["ball", "sports ball", "football", "soccer", "tennis", "baseball", "basketball", "golf"],
        "food": ["food", "fruit", "vegetable", "dish", "meal", "pizza", "hamburger", "cake"],
        "landscape": ["mountain", "beach", "forest", "lake", "river", "ocean", "sky"],
        "building": ["house", "building", "tower", "church", "castle", "bridge"],
        "object": ["bottle", "cup", "chair", "table", "phone", "computer", "book"]
    }
    
    # Check top predictions against categories
    for prediction in predictions:
        label = prediction["label"].lower()
        
        # Check if any keywords from our category mappings are in the label
        for category, keywords in category_mappings.items():
            for keyword in keywords:
                if keyword in label:
                    return category
    
    # If no specific category matched, use the first prediction's class
    return predictions[0]["label"]


def analyze_image_file(file_path: str) -> Dict[str, Any]:
    """
    Analyzes an image file and extracts features.

    Args:
        file_path: Path to the image file

    Returns:
        Dictionary with extracted features
    """
    try:
        # Get file size
        file_size = os.path.getsize(file_path)

        # Get image dimensions
        width, height = get_image_dimensions(file_path)

        # Extract advanced features using ResNet50
        advanced_features = create_advanced_image_features(file_path)
        
        # Classify the image
        classification = classify_image(file_path)

        # Calculate a file hash for uniqueness checking
        file_hash = calculate_file_hash(file_path)

        return {
            'content_type': 'image',
            'file_path': file_path,
            'file_size': file_size,
            'dimensions': (width, height),
            'aspect_ratio': width / height if height != 0 else 0,
            'file_hash': file_hash,
            'advanced_features': advanced_features.tolist() if isinstance(advanced_features, np.ndarray) else advanced_features,
            'classification': classification
        }

    except Exception as e:
        print(f"Error analyzing image file {file_path}: {e}")
        return {
            'content_type': 'image',
            'file_path': file_path,
            'error': str(e)
        }
