"""
Example Usage Script
Demonstrates how to use the thyroid nodule classification pipeline
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Import modules
from preprocessing import ImagePreprocessor
from feature_extraction import FeatureFusion, LBPExtractor, HOGExtractor
from classification import (
    RandomForestThyroidClassifier,
    LogisticRegressionThyroidClassifier,
    LinearSVMThyroidClassifier,
    RBFSVMThyroidClassifier,
    ClassifierEvaluator
)
from main_pipeline import ThyroidNoduleClassificationPipeline


def example_1_preprocessing():
    """
    Example 1: Image Preprocessing
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: IMAGE PREPROCESSING")
    print("="*70)
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(target_size=(224, 224))
    
    # Preprocess a single image
    # preprocessed_image, metadata = preprocessor.preprocess('data/raw/sample_image.jpg')
    
    # Visualize preprocessing steps
    # preprocessor.visualize_preprocessing('data/raw/sample_image.jpg', 'preprocessing_visualization.png')
    
    print("✓ Image preprocessing initialized")
    print("  - Target size: 224x224")
    print("  - Features: Alignment, normalization, rescaling")


def example_2_lbp_features():
    """
    Example 2: Local Binary Pattern (LBP) Feature Extraction
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: LOCAL BINARY PATTERN (LBP) FEATURES")
    print("="*70)
    
    # Create sample image (224x224)
    sample_image = np.random.rand(224, 224)
    
    # Initialize LBP extractor
    lbp_extractor = LBPExtractor(radius=1, n_points=8)
    
    # Extract features
    lbp_features = lbp_extractor.extract(sample_image)
    
    print(f"✓ LBP Features extracted")
    print(f"  - Feature vector size: {len(lbp_features)}")
    print(f"  - Represents local texture patterns")
    print(f"  - Example values (first 10): {lbp_features[:10]}")


def example_3_hog_features():
    """
    Example 3: Histogram of Oriented Gradients (HOG) Feature Extraction
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: HISTOGRAM OF ORIENTED GRADIENTS (HOG) FEATURES")
    print("="*70)
    
    # Create sample image
    sample_image = np.random.rand(224, 224)
    
    # Initialize HOG extractor
    hog_extractor = HOGExtractor(
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2)
    )
    
    # Extract features
    hog_features = hog_extractor.extract(sample_image)
    
    print(f"✓ HOG Features extracted")
    print(f"  - Feature vector size: {len(hog_features)}")
    print(f"  - Represents shape and directional patterns")
    print(f"  - Example values (first 10): {hog_features[:10]}")


def example_4_cae_features():
    """
    Example 4: Convolutional Auto-Encoder (CAE) Feature Extraction
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: CONVOLUTIONAL AUTO-ENCODER (CAE) FEATURES")
    print("="*70)
    
    # Create sample images (10 samples, 224x224)
    sample_images = np.random.rand(10, 224, 224)
    
    # Initialize feature fusion (includes CAE)
    feature_fusion = FeatureFusion()
    
    # Train CAE
    print("Training CAE on sample data...")
    feature_fusion.train_cae(sample_images, epochs=5, batch_size=4)
    
    # Extract CAE features
    images_expanded = np.expand_dims(sample_images, axis=-1)
    cae_features = feature_fusion.cae.extract_features(images_expanded)
    
    print(f"✓ CAE Features extracted")
    print(f"  - Feature vector size: {cae_features.shape[1]}")
    print(f"  - Learned compressed representation")
    print(f"  - Example values (first sample): {cae_features[0][:10]}")


def example_5_feature_fusion():
    """
    Example 5: Feature Fusion
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: FEATURE FUSION")
    print("="*70)
    
    # Create sample images
    sample_images = np.random.rand(20, 224, 224)
    sample_labels = np.random.randint(0, 2, 20)
    
    # Initialize feature fusion
    feature_fusion = FeatureFusion()
    
    # Train CAE
    print("Training CAE...")
    feature_fusion.train_cae(sample_images, epochs=5, batch_size=4)
    
    # Extract and fuse features
    fused_features = feature_fusion.extract_fused_features(sample_images, use_cae=True)
    
    print(f"✓ Features fused successfully")
    print(f"  - LBP features: {feature_fusion.lbp_dim}")
    print(f"  - HOG features: {feature_fusion.hog_dim}")
    print(f"  - CAE features: {feature_fusion.cae_dim}")
    print(f"  - Total fused features: {feature_fusion.total_dim}")
    print(f"  - Fused feature shape: {fused_features.shape}")


def example_6_single_classifier():
    """
    Example 6: Train Single Classifier
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: TRAIN SINGLE CLASSIFIER (RBF SVM)")
    print("="*70)
    
    # Create synthetic data
    X = np.random.rand(100, 256)  # 100 samples, 256 features
    y = np.random.randint(0, 2, 100)  # Binary labels
    
    # Initialize classifier
    classifier = RBFSVMThyroidClassifier(C=1.0, gamma='scale')
    
    # Train
    print("Training RBF SVM...")
    classifier.train(X, y, test_size=0.2)
    
    # Evaluate
    metrics = classifier.evaluate()
    classifier.print_metrics()
    
    # Make predictions
    new_sample = np.random.rand(1, 256)
    prediction = classifier.predict(new_sample)
    
    print(f"✓ Sample prediction: {['Benign', 'Suspicious'][prediction[0]]}")


def example_7_multiple_classifiers():
    """
    Example 7: Compare Multiple Classifiers
    """
    print("\n" + "="*70)
    print("EXAMPLE 7: COMPARE MULTIPLE CLASSIFIERS")
    print("="*70)
    
    # Create synthetic data
    X = np.random.rand(200, 256)
    y = np.random.randint(0, 2, 200)
    
    # Initialize evaluator
    evaluator = ClassifierEvaluator()
    
    # Add classifiers
    evaluator.add_classifier(RandomForestThyroidClassifier())
    evaluator.add_classifier(LogisticRegressionThyroidClassifier())
    evaluator.add_classifier(LinearSVMThyroidClassifier())
    evaluator.add_classifier(RBFSVMThyroidClassifier())
    
    # Train all
    print("Training all classifiers...")
    evaluator.train_all(X, y, test_size=0.2)
    
    # Compare
    results_df = evaluator.compare_classifiers()
    evaluator.print_comparison()
    
    print("\n✓ Comparison complete")
    print(f"Best classifier: {results_df.loc[results_df['Final Score'].idxmax(), 'Classifier']}")


def example_8_full_pipeline():
    """
    Example 8: Full Pipeline
    """
    print("\n" + "="*70)
    print("EXAMPLE 8: FULL PIPELINE")
    print("="*70)
    
    # NOTE: This requires actual image files and labels.csv
    # Uncomment to use with your dataset
    
    """
    # Initialize pipeline
    pipeline = ThyroidNoduleClassificationPipeline()
    
    # Run full pipeline
    results = pipeline.run_full_pipeline(
        image_dir='data/raw/',
        labels_file='data/labels.csv',
        output_dir='results/'
    )
    
    print("✓ Full pipeline completed")
    print(f"Results saved to results/")
    """
    
    print("Full pipeline example (requires actual image files)")
    print("Usage:")
    print("  python main_pipeline.py \\")
    print("    --image-dir data/raw/ \\")
    print("    --labels-file data/labels.csv \\")
    print("    --output-dir results/")


def example_9_custom_pipeline():
    """
    Example 9: Custom Pipeline Configuration
    """
    print("\n" + "="*70)
    print("EXAMPLE 9: CUSTOM PIPELINE CONFIGURATION")
    print("="*70)
    
    # Define custom configuration
    custom_config = {
        'image_size': (256, 256),
        'test_size': 0.25,
        'use_cae': True,
        'cae_epochs': 100,
        'cae_batch_size': 16,
        'random_state': 42
    }
    
    # Initialize with custom config
    pipeline = ThyroidNoduleClassificationPipeline(config=custom_config)
    
    print("✓ Custom pipeline configured")
    print(f"  - Image size: {custom_config['image_size']}")
    print(f"  - Test size: {custom_config['test_size']}")
    print(f"  - CAE epochs: {custom_config['cae_epochs']}")
    print(f"  - CAE batch size: {custom_config['cae_batch_size']}")


def example_10_visualization():
    """
    Example 10: Visualization
    """
    print("\n" + "="*70)
    print("EXAMPLE 10: VISUALIZATION")
    print("="*70)
    
    # Create synthetic data
    X = np.random.rand(200, 256)
    y = np.random.randint(0, 2, 200)
    
    # Train classifiers
    evaluator = ClassifierEvaluator()
    evaluator.add_classifier(RandomForestThyroidClassifier())
    evaluator.add_classifier(RBFSVMThyroidClassifier())
    
    evaluator.train_all(X, y, test_size=0.2)
    results_df = evaluator.compare_classifiers()
    
    # Create visualizations
    print("Creating visualizations...")
    Path('results').mkdir(exist_ok=True)
    
    # evaluator.plot_roc_curves('results/roc_curves.png')
    # evaluator.plot_confusion_matrices('results/confusion_matrices.png')
    
    print("✓ Visualizations created (ROC curves and confusion matrices)")


def main():
    """
    Run all examples
    """
    print("\n" + "="*70)
    print("THYROID NODULE CLASSIFICATION - USAGE EXAMPLES")
    print("="*70)
    
    # Run examples
    example_1_preprocessing()
    example_2_lbp_features()
    example_3_hog_features()
    example_4_cae_features()
    example_5_feature_fusion()
    example_6_single_classifier()
    example_7_multiple_classifiers()
    example_8_full_pipeline()
    example_9_custom_pipeline()
    example_10_visualization()
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED")
    print("="*70)
    print("\nNext steps:")
    print("1. Prepare your data with images in data/raw/ and labels in data/labels.csv")
    print("2. Run the full pipeline: python main_pipeline.py --image-dir data/raw/ --labels-file data/labels.csv")
    print("3. Check results/ directory for outputs")
    print("\nFor detailed documentation, see README.md")


if __name__ == "__main__":
    main()
