"""
Main Pipeline Script
Orchestrates the complete thyroid nodule classification workflow from preprocessing to evaluation
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from preprocessing import ImagePreprocessor
from feature_extraction import FeatureFusion
from classification import (
    RandomForestThyroidClassifier,
    LogisticRegressionThyroidClassifier,
    LinearSVMThyroidClassifier,
    RBFSVMThyroidClassifier,
    ClassifierEvaluator
)


class ThyroidNoduleClassificationPipeline:
    """
    Complete pipeline for thyroid nodule classification
    """
    
    def __init__(self, config=None):
        """
        Initialize pipeline
        
        Args:
            config (dict): Configuration dictionary
        """
        if config is None:
            config = {
                'image_size': (224, 224),
                'test_size': 0.2,
                'use_cae': True,
                'cae_epochs': 50,
                'cae_batch_size': 32,
                'random_state': 42
            }
        
        self.config = config
        self.preprocessor = ImagePreprocessor(target_size=config['image_size'])
        self.feature_fusion = FeatureFusion()
        self.evaluator = ClassifierEvaluator()
        
        self.preprocessed_images = None
        self.labels = None
        self.fused_features = None
        self.best_classifier = None
    
    def load_images_and_labels(self, image_dir, labels_file):
        """
        Load images and corresponding labels
        
        Args:
            image_dir (str): Directory containing ultrasound images
            labels_file (str): CSV file with labels (filename, label)
        """
        print("\n" + "="*70)
        print("STEP 1: LOADING IMAGES AND LABELS")
        print("="*70)
        
        # Load labels
        df_labels = pd.read_csv(labels_file)
        
        # Load and preprocess images
        images = []
        valid_labels = []
        image_paths = []
        
        for _, row in tqdm(df_labels.iterrows(), total=len(df_labels)):
            img_path = Path(image_dir) / row['filename']
            label = row['label']  # 0=benign, 1=suspicious
            
            if img_path.exists():
                try:
                    preprocessed, _ = self.preprocessor.preprocess(str(img_path))
                    images.append(preprocessed)
                    valid_labels.append(label)
                    image_paths.append(str(img_path))
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
        
        self.preprocessed_images = np.array(images)
        self.labels = np.array(valid_labels)
        
        print(f"\nLoaded {len(self.preprocessed_images)} images")
        print(f"Class distribution:")
        print(f"  Benign (0): {np.sum(self.labels == 0)} ({np.mean(self.labels == 0)*100:.1f}%)")
        print(f"  Suspicious (1): {np.sum(self.labels == 1)} ({np.mean(self.labels == 1)*100:.1f}%)")
    
    def extract_features(self):
        """
        Extract features from preprocessed images
        """
        print("\n" + "="*70)
        print("STEP 2: FEATURE EXTRACTION")
        print("="*70)
        
        if self.preprocessed_images is None:
            raise ValueError("Images not loaded. Call load_images_and_labels() first.")
        
        # Train CAE if using it
        if self.config['use_cae']:
            self.feature_fusion.train_cae(
                self.preprocessed_images,
                epochs=self.config['cae_epochs'],
                batch_size=self.config['cae_batch_size']
            )
        
        # Extract and fuse features
        self.fused_features = self.feature_fusion.extract_fused_features(
            self.preprocessed_images,
            use_cae=self.config['use_cae']
        )
    
    def train_classifiers(self):
        """
        Train and evaluate multiple classifiers
        """
        print("\n" + "="*70)
        print("STEP 3: CLASSIFIER TRAINING AND EVALUATION")
        print("="*70)
        
        if self.fused_features is None:
            raise ValueError("Features not extracted. Call extract_features() first.")
        
        # Initialize classifiers
        classifiers = [
            RandomForestThyroidClassifier(n_estimators=100, max_depth=10),
            LogisticRegressionThyroidClassifier(max_iter=1000),
            LinearSVMThyroidClassifier(C=1.0),
            RBFSVMThyroidClassifier(C=1.0, gamma='scale')
        ]
        
        for clf in classifiers:
            self.evaluator.add_classifier(clf)
        
        # Train all classifiers
        self.evaluator.train_all(
            self.fused_features,
            self.labels,
            test_size=self.config['test_size']
        )
    
    def evaluate_and_compare(self):
        """
        Compare all classifiers and identify best one
        """
        print("\n" + "="*70)
        print("STEP 4: CLASSIFIER COMPARISON")
        print("="*70)
        
        # Compare classifiers
        results_df = self.evaluator.compare_classifiers()
        self.evaluator.print_comparison()
        
        # Identify best classifier by final score
        best_idx = results_df['Final Score'].idxmax()
        best_clf_name = results_df.iloc[best_idx]['Classifier']
        self.best_classifier = self.evaluator.classifiers[best_clf_name]
        
        return results_df
    
    def visualize_results(self, output_dir='results'):
        """
        Visualize classification results
        
        Args:
            output_dir (str): Directory to save visualizations
        """
        print("\n" + "="*70)
        print("STEP 5: VISUALIZATION")
        print("="*70)
        
        Path(output_dir).mkdir(exist_ok=True)
        
        # Plot ROC curves
        print("Plotting ROC curves...")
        self.evaluator.plot_roc_curves(
            save_path=f'{output_dir}/roc_curves.png'
        )
        
        # Plot confusion matrices
        print("Plotting confusion matrices...")
        self.evaluator.plot_confusion_matrices(
            save_path=f'{output_dir}/confusion_matrices.png'
        )
        
        # Plot comparison metrics
        print("Plotting comparison metrics...")
        self._plot_metrics_comparison(f'{output_dir}/metrics_comparison.png')
    
    def _plot_metrics_comparison(self, save_path):
        """Plot metrics comparison bar chart"""
        results_df = self.evaluator.results_df
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        metrics = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1-Score', 'Final Score']
        positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        
        for metric, pos in zip(metrics, positions):
            ax = axes[pos]
            ax.bar(results_df['Classifier'], results_df[metric], color='steelblue')
            ax.set_title(metric)
            ax.set_ylabel('Score')
            ax.set_ylim([0, 1])
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def generate_report(self, output_file='classification_report.txt'):
        """
        Generate comprehensive classification report
        
        Args:
            output_file (str): Output file path
        """
        print("\n" + "="*70)
        print("GENERATING REPORT")
        print("="*70)
        
        report = []
        report.append("="*80)
        report.append("THYROID NODULE CLASSIFICATION - COMPREHENSIVE REPORT")
        report.append("="*80)
        
        # Configuration
        report.append("\n## CONFIGURATION ##")
        for key, value in self.config.items():
            report.append(f"{key}: {value}")
        
        # Dataset summary
        report.append("\n## DATASET SUMMARY ##")
        report.append(f"Total samples: {len(self.labels)}")
        report.append(f"Benign samples: {np.sum(self.labels == 0)} ({np.mean(self.labels == 0)*100:.1f}%)")
        report.append(f"Suspicious samples: {np.sum(self.labels == 1)} ({np.mean(self.labels == 1)*100:.1f}%)")
        report.append(f"Image size: {self.config['image_size']}")
        
        # Feature summary
        report.append("\n## FEATURE EXTRACTION ##")
        report.append(f"LBP features: {self.feature_fusion.lbp_dim}")
        report.append(f"HOG features: {self.feature_fusion.hog_dim}")
        if self.config['use_cae']:
            report.append(f"CAE features: {self.feature_fusion.cae_dim}")
        report.append(f"Total features: {self.feature_fusion.total_dim}")
        
        # Classifier comparison
        report.append("\n## CLASSIFIER COMPARISON ##")
        report.append(self.evaluator.results_df.to_string(index=False))
        
        # Best classifier details
        if self.best_classifier:
            report.append("\n## BEST CLASSIFIER DETAILS ##")
            report.append(f"Name: {self.best_classifier.name}")
            report.append(f"Accuracy: {self.best_classifier.metrics['accuracy']:.4f}")
            report.append(f"Precision: {self.best_classifier.metrics['precision']:.4f}")
            report.append(f"Sensitivity: {self.best_classifier.metrics['sensitivity']:.4f}")
            report.append(f"Specificity: {self.best_classifier.metrics['specificity']:.4f}")
            report.append(f"F1-Score: {self.best_classifier.metrics['f1']:.4f}")
            report.append(f"Final Score: {self.best_classifier.metrics['final_score']:.4f}")
            
            # Confusion matrix
            cm = self.best_classifier.get_confusion_matrix()
            report.append("\nConfusion Matrix:")
            report.append(f"  True Negatives (TN): {cm[0, 0]}")
            report.append(f"  False Positives (FP): {cm[0, 1]}")
            report.append(f"  False Negatives (FN): {cm[1, 0]}")
            report.append(f"  True Positives (TP): {cm[1, 1]}")
        
        # Save report
        report_text = '\n'.join(report)
        with open(output_file, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\nReport saved to {output_file}")
    
    def run_full_pipeline(self, image_dir, labels_file, output_dir='results'):
        """
        Run the complete pipeline
        
        Args:
            image_dir (str): Directory containing images
            labels_file (str): CSV file with labels
            output_dir (str): Directory for outputs
        """
        print("\n" + "="*70)
        print("STARTING THYROID NODULE CLASSIFICATION PIPELINE")
        print("="*70)
        
        # Step 1: Load images
        self.load_images_and_labels(image_dir, labels_file)
        
        # Step 2: Extract features
        self.extract_features()
        
        # Step 3: Train classifiers
        self.train_classifiers()
        
        # Step 4: Evaluate and compare
        results_df = self.evaluate_and_compare()
        
        # Step 5: Visualize results
        self.visualize_results(output_dir)
        
        # Step 6: Generate report
        self.generate_report(f'{output_dir}/classification_report.txt')
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETE!")
        print("="*70)
        
        return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Thyroid Nodule Classification Pipeline')
    parser.add_argument('--image-dir', type=str, required=True, help='Directory containing ultrasound images')
    parser.add_argument('--labels-file', type=str, required=True, help='CSV file with labels')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--cae-epochs', type=int, default=50, help='CAE training epochs')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = ThyroidNoduleClassificationPipeline()
    
    # Run pipeline
    results = pipeline.run_full_pipeline(
        image_dir=args.image_dir,
        labels_file=args.labels_file,
        output_dir=args.output_dir
    )
