"""
Image Preprocessing Module
Handles image alignment, centering, and normalization for thyroid ultrasound images
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


class ImagePreprocessor:
    """
    Preprocesses thyroid ultrasound images by:
    1. Aligning nodule center
    2. Normalizing image size to 224x224
    3. Rescaling pixel values to [0, 1]
    """
    
    def __init__(self, target_size=(224, 224)):
        """
        Initialize preprocessor
        
        Args:
            target_size (tuple): Target image dimensions (default: 224x224)
        """
        self.target_size = target_size
        self.stretch_ratios = {}
    
    def find_nodule_center(self, image):
        """
        Find the center of the thyroid nodule using boundary detection
        
        Args:
            image (ndarray): Grayscale ultrasound image
            
        Returns:
            tuple: (center_x, center_y) coordinates
        """
        # Apply binary threshold to find nodule boundary
        _, binary = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            # If no clear contour, use image center
            return image.shape[1] // 2, image.shape[0] // 2
        
        # Get the largest contour (nodule)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate center using moments
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return image.shape[1] // 2, image.shape[0] // 2
        
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        
        return center_x, center_y
    
    def align_to_center(self, image):
        """
        Align nodule center to image center
        
        Args:
            image (ndarray): Input image
            
        Returns:
            ndarray: Centered image
        """
        # Find nodule center
        nodule_center_x, nodule_center_y = self.find_nodule_center(image)
        
        # Image dimensions
        height, width = image.shape[:2]
        image_center_x = width // 2
        image_center_y = height // 2
        
        # Calculate translation
        tx = image_center_x - nodule_center_x
        ty = image_center_y - nodule_center_y
        
        # Apply translation
        M_translate = cv2.getRotationMatrix2D((width // 2, height // 2), 0, 1)
        M_translate[0, 2] += tx
        M_translate[1, 2] += ty
        
        aligned = cv2.warpAffine(image, M_translate, (width, height))
        
        return aligned
    
    def normalize_size(self, image):
        """
        Normalize image size to target_size (224x224)
        
        Args:
            image (ndarray): Input image
            
        Returns:
            tuple: (normalized_image, stretch_ratio_x, stretch_ratio_y)
        """
        height, width = image.shape[:2]
        target_height, target_width = self.target_size
        
        # Calculate stretch ratios
        stretch_x = target_width / width
        stretch_y = target_height / height
        
        # Resize using bilinear interpolation
        normalized = cv2.resize(image, self.target_size, interpolation=cv2.INTER_LINEAR)
        
        return normalized, stretch_x, stretch_y
    
    def rescale_pixel_values(self, image):
        """
        Rescale pixel values to [0, 1] range
        
        Args:
            image (ndarray): Input image
            
        Returns:
            ndarray: Rescaled image with values in [0, 1]
        """
        # Convert to float and normalize
        rescaled = image.astype(np.float32) / 255.0
        return rescaled
    
    def preprocess(self, image_path):
        """
        Complete preprocessing pipeline
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            tuple: (preprocessed_image, metadata_dict)
        """
        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
        else:
            image = image_path
        
        metadata = {
            'original_shape': image.shape,
            'original_image': image.copy()
        }
        
        # Step 1: Align to center
        centered = self.align_to_center(image)
        metadata['centered_image'] = centered.copy()
        
        # Step 2: Normalize size
        normalized, stretch_x, stretch_y = self.normalize_size(centered)
        metadata['stretch_ratio_x'] = stretch_x
        metadata['stretch_ratio_y'] = stretch_y
        
        # Step 3: Rescale pixel values
        rescaled = self.rescale_pixel_values(normalized)
        
        return rescaled, metadata
    
    def batch_preprocess(self, image_paths, verbose=True):
        """
        Preprocess multiple images
        
        Args:
            image_paths (list): List of image file paths
            verbose (bool): Print progress
            
        Returns:
            tuple: (preprocessed_images_array, metadata_list)
        """
        preprocessed_images = []
        metadata_list = []
        
        for idx, path in enumerate(image_paths):
            if verbose and idx % 10 == 0:
                print(f"Processing image {idx}/{len(image_paths)}")
            
            try:
                preprocessed, metadata = self.preprocess(path)
                preprocessed_images.append(preprocessed)
                metadata_list.append(metadata)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
        
        return np.array(preprocessed_images), metadata_list
    
    def visualize_preprocessing(self, image_path, save_path=None):
        """
        Visualize preprocessing steps
        
        Args:
            image_path (str): Path to image
            save_path (str): Optional path to save visualization
        """
        preprocessed, metadata = self.preprocess(image_path)
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        # Original
        axes[0].imshow(metadata['original_image'], cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Centered
        axes[1].imshow(metadata['centered_image'], cmap='gray')
        axes[1].set_title('Centered')
        axes[1].axis('off')
        
        # Normalized size
        axes[2].imshow(preprocessed, cmap='gray')
        axes[2].set_title(f'Normalized to {self.target_size}')
        axes[2].axis('off')
        
        # Rescaled values
        axes[3].imshow(preprocessed, cmap='gray')
        axes[3].set_title('Rescaled [0, 1]')
        axes[3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()


if __name__ == "__main__":
    # Example usage
    preprocessor = ImagePreprocessor(target_size=(224, 224))
    
    # Preprocess a single image
    # preprocessed_image, metadata = preprocessor.preprocess('path/to/image.png')
    
    # Visualize preprocessing
    # preprocessor.visualize_preprocessing('path/to/image.png')
    
    print("ImagePreprocessor module loaded successfully")
