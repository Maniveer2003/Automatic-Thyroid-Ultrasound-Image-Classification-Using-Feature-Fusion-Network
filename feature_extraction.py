"""
Feature Extraction Module
Implements LBP, HOG, and Convolutional Auto-Encoder for thyroid nodule feature extraction
"""

import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from skimage import feature
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
import tensorflow as tf


class LBPExtractor:
    """
    Local Binary Pattern (LBP) feature extractor
    Captures texture information of thyroid nodules
    """
    
    def __init__(self, radius=1, n_points=8):
        """
        Initialize LBP extractor
        
        Args:
            radius (int): Radius of LBP neighborhood
            n_points (int): Number of points in neighborhood
        """
        self.radius = radius
        self.n_points = n_points
    
    def extract(self, image):
        """
        Extract LBP features from image
        
        Args:
            image (ndarray): Input image (should be 224x224, normalized to [0,1])
            
        Returns:
            ndarray: LBP feature vector (histogram)
        """
        # Convert to uint8 if normalized
        if image.dtype == np.float32 or image.dtype == np.float64:
            img = (image * 255).astype(np.uint8)
        else:
            img = image
        
        # Calculate LBP
        lbp = local_binary_pattern(img, self.n_points, self.radius, method='uniform')
        
        # Create histogram
        n_bins = self.n_points + 2  # For uniform LBP
        hist, _ = np.histogram(lbp.ravel(), bins=range(n_bins + 1), range=(0, n_bins))
        
        # Normalize histogram
        hist = hist.astype(np.float32) / hist.sum()
        
        return hist
    
    def batch_extract(self, images):
        """
        Extract LBP features from batch of images
        
        Args:
            images (ndarray): Array of images (N, 224, 224)
            
        Returns:
            ndarray: Array of LBP feature vectors (N, n_bins)
        """
        features = []
        for img in images:
            feat = self.extract(img)
            features.append(feat)
        return np.array(features)


class HOGExtractor:
    """
    Histogram of Oriented Gradients (HOG) feature extractor
    Captures shape and directional patterns of thyroid nodules
    """
    
    def __init__(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        """
        Initialize HOG extractor
        
        Args:
            orientations (int): Number of orientation bins
            pixels_per_cell (tuple): Pixels per cell
            cells_per_block (tuple): Cells per block (for normalization)
        """
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
    
    def extract(self, image):
        """
        Extract HOG features from image
        
        Args:
            image (ndarray): Input image (224x224, normalized to [0,1])
            
        Returns:
            ndarray: HOG feature vector
        """
        # Convert to uint8 if normalized
        if image.dtype == np.float32 or image.dtype == np.float64:
            img = (image * 255).astype(np.uint8)
        else:
            img = image
        
        # Calculate HOG features
        features, _ = feature.hog(
            img,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            visualize=False,
            channel_axis=None
        )
        
        return features.astype(np.float32)
    
    def batch_extract(self, images):
        """
        Extract HOG features from batch of images
        
        Args:
            images (ndarray): Array of images (N, 224, 224)
            
        Returns:
            ndarray: Array of HOG feature vectors
        """
        features = []
        for img in images:
            feat = self.extract(img)
            features.append(feat)
        return np.array(features)


class ConvolutionalAutoencoder:
    """
    Convolutional Auto-Encoder (CAE) for feature extraction
    Learns compressed representations of thyroid nodules
    """
    
    def __init__(self, input_shape=(224, 224, 1), latent_dim=128):
        """
        Initialize CAE
        
        Args:
            input_shape (tuple): Input image shape
            latent_dim (int): Dimension of latent (bottleneck) representation
        """
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.model = None
        self.encoder = None
    
    def build_encoder(self):
        """
        Build encoder part of CAE
        
        Returns:
            keras.Model: Encoder model
        """
        inputs = layers.Input(shape=self.input_shape)
        
        # Encoder: Convolution + MaxPooling to compress
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)  # 112x112
        
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)  # 56x56
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)  # 28x28
        
        # Flatten to latent vector
        x = layers.Flatten()(x)
        latent = layers.Dense(self.latent_dim, activation='relu', name='latent')(x)
        
        encoder = models.Model(inputs, latent, name='encoder')
        return encoder
    
    def build_decoder(self):
        """
        Build decoder part of CAE
        
        Returns:
            keras.Model: Decoder model
        """
        latent_inputs = layers.Input(shape=(self.latent_dim,))
        
        # Reshape to spatial dimensions
        x = layers.Dense(28 * 28 * 128, activation='relu')(latent_inputs)
        x = layers.Reshape((28, 28, 128))(x)
        
        # Decoder: Upsampling + Convolution to reconstruct
        x = layers.UpSampling2D((2, 2))(x)  # 56x56
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        
        x = layers.UpSampling2D((2, 2))(x)  # 112x112
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        
        x = layers.UpSampling2D((2, 2))(x)  # 224x224
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        
        # Output layer
        outputs = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        
        decoder = models.Model(latent_inputs, outputs, name='decoder')
        return decoder
    
    def build_autoencoder(self):
        """
        Build complete autoencoder model
        
        Returns:
            keras.Model: Full autoencoder model
        """
        inputs = layers.Input(shape=self.input_shape)
        
        # Encoder
        self.encoder = self.build_encoder()
        
        # Decoder
        decoder = self.build_decoder()
        
        # Connect encoder to decoder
        latent = self.encoder(inputs)
        outputs = decoder(latent)
        
        # Full autoencoder
        self.model = models.Model(inputs, outputs, name='autoencoder')
        
        return self.model
    
    def compile(self, optimizer='adam', loss='mse'):
        """
        Compile the autoencoder
        
        Args:
            optimizer (str): Optimizer name
            loss (str): Loss function
        """
        if self.model is None:
            self.build_autoencoder()
        
        self.model.compile(optimizer=optimizer, loss=loss)
    
    def train(self, train_images, epochs=50, batch_size=32, validation_split=0.2):
        """
        Train the autoencoder
        
        Args:
            train_images (ndarray): Training images (N, 224, 224, 1)
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            validation_split (float): Validation split ratio
        """
        if self.model is None:
            self.build_autoencoder()
        
        self.compile()
        
        history = self.model.fit(
            train_images, train_images,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        return history
    
    def extract_features(self, images):
        """
        Extract latent features from images using trained encoder
        
        Args:
            images (ndarray): Input images (N, 224, 224, 1)
            
        Returns:
            ndarray: Latent feature vectors (N, latent_dim)
        """
        if self.encoder is None:
            raise ValueError("Encoder not built. Train the autoencoder first.")
        
        features = self.encoder.predict(images, verbose=0)
        return features
    
    def reconstruct(self, images):
        """
        Reconstruct images using trained autoencoder
        
        Args:
            images (ndarray): Input images (N, 224, 224, 1)
            
        Returns:
            ndarray: Reconstructed images
        """
        if self.model is None:
            raise ValueError("Autoencoder not built.")
        
        reconstructed = self.model.predict(images, verbose=0)
        return reconstructed


class FeatureFusion:
    """
    Fuses features from LBP, HOG, and CAE into a single feature vector
    """
    
    def __init__(self):
        """Initialize feature fusion"""
        self.lbp_extractor = LBPExtractor()
        self.hog_extractor = HOGExtractor()
        self.cae = ConvolutionalAutoencoder()
        
        self.lbp_dim = None
        self.hog_dim = None
        self.cae_dim = None
        self.total_dim = None
    
    def train_cae(self, images, epochs=50, batch_size=32):
        """
        Train the CAE on the image dataset
        
        Args:
            images (ndarray): Training images (N, 224, 224)
            epochs (int): Training epochs
            batch_size (int): Batch size
        """
        print("Training Convolutional Auto-Encoder...")
        
        # Reshape for CAE (add channel dimension)
        images_expanded = np.expand_dims(images, axis=-1)
        
        self.cae.build_autoencoder()
        self.cae.train(images_expanded, epochs=epochs, batch_size=batch_size)
        
        print("CAE training complete!")
    
    def extract_fused_features(self, images, use_cae=True):
        """
        Extract and fuse features from LBP, HOG, and CAE
        
        Args:
            images (ndarray): Input images (N, 224, 224)
            use_cae (bool): Whether to use CAE features
            
        Returns:
            ndarray: Fused feature vectors (N, total_features)
        """
        print("Extracting LBP features...")
        lbp_features = self.lbp_extractor.batch_extract(images)
        self.lbp_dim = lbp_features.shape[1]
        
        print("Extracting HOG features...")
        hog_features = self.hog_extractor.batch_extract(images)
        self.hog_dim = hog_features.shape[1]
        
        features_list = [lbp_features, hog_features]
        
        if use_cae:
            print("Extracting CAE features...")
            images_expanded = np.expand_dims(images, axis=-1)
            cae_features = self.cae.extract_features(images_expanded)
            self.cae_dim = cae_features.shape[1]
            features_list.append(cae_features)
        
        # Concatenate all features
        fused_features = np.concatenate(features_list, axis=1)
        self.total_dim = fused_features.shape[1]
        
        print(f"\nFeature Extraction Summary:")
        print(f"  LBP features: {self.lbp_dim}")
        print(f"  HOG features: {self.hog_dim}")
        if use_cae:
            print(f"  CAE features: {self.cae_dim}")
        print(f"  Total fused features: {self.total_dim}")
        
        return fused_features
    
    def visualize_features(self, image, save_path=None):
        """
        Visualize extracted features
        
        Args:
            image (ndarray): Single image (224, 224)
            save_path (str): Optional path to save visualization
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # LBP
        lbp_features = self.lbp_extractor.extract(image)
        axes[0, 1].bar(range(len(lbp_features)), lbp_features)
        axes[0, 1].set_title('LBP Histogram')
        axes[0, 1].set_xlabel('Bins')
        axes[0, 1].set_ylabel('Frequency')
        
        # HOG
        hog_features = self.hog_extractor.extract(image)
        axes[0, 2].plot(hog_features)
        axes[0, 2].set_title('HOG Features')
        axes[0, 2].set_ylabel('Feature Value')
        
        # CAE features
        image_expanded = np.expand_dims(image, axis=(0, -1))
        cae_features = self.cae.extract_features(image_expanded)[0]
        axes[1, 0].plot(cae_features)
        axes[1, 0].set_title('CAE Features (Latent Vector)')
        axes[1, 0].set_ylabel('Feature Value')
        
        # CAE reconstruction
        reconstructed = self.cae.reconstruct(image_expanded)[0, :, :, 0]
        axes[1, 1].imshow(reconstructed, cmap='gray')
        axes[1, 1].set_title('CAE Reconstruction')
        axes[1, 1].axis('off')
        
        # Reconstruction error
        reconstruction_error = np.abs(image - reconstructed)
        axes[1, 2].imshow(reconstruction_error, cmap='hot')
        axes[1, 2].set_title('Reconstruction Error')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()


if __name__ == "__main__":
    print("Feature extraction modules loaded successfully")
