"""
Classification Module
Trains and evaluates classifiers for thyroid nodule diagnosis
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_curve, auc, 
                             classification_report, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
import pandas as pd


class ThyroidClassifier:
    """
    Base class for thyroid nodule classifiers
    """
    
    def __init__(self, name):
        """
        Initialize classifier
        
        Args:
            name (str): Name of the classifier
        """
        self.name = name
        self.model = None
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_proba = None
        self.metrics = {}
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into train and test sets
        
        Args:
            X (ndarray): Features
            y (ndarray): Labels (0=benign, 1=suspicious)
            test_size (float): Test split ratio
            random_state (int): Random seed
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Standardize features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"Data split:")
        print(f"  Training samples: {len(self.X_train)}")
        print(f"  Testing samples: {len(self.X_test)}")
        print(f"  Class distribution (train): {np.bincount(self.y_train)}")
        print(f"  Class distribution (test): {np.bincount(self.y_test)}")
    
    def train(self, X, y, test_size=0.2):
        """
        Train the classifier
        
        Args:
            X (ndarray): Features
            y (ndarray): Labels
            test_size (float): Test split ratio
        """
        # Split data
        self.split_data(X, y, test_size=test_size)
        
        # Train model
        print(f"\nTraining {self.name}...")
        self.model.fit(self.X_train, self.y_train)
        
        # Predict
        self.y_pred = self.model.predict(self.X_test)
        
        # Probability predictions (for ROC curve)
        if hasattr(self.model, 'predict_proba'):
            self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        elif hasattr(self.model, 'decision_function'):
            self.y_pred_proba = self.model.decision_function(self.X_test)
        
        print(f"{self.name} training complete!")
    
    def evaluate(self):
        """
        Evaluate classifier performance
        
        Returns:
            dict: Performance metrics
        """
        self.metrics = {
            'accuracy': accuracy_score(self.y_test, self.y_pred),
            'precision': precision_score(self.y_test, self.y_pred),
            'sensitivity': recall_score(self.y_test, self.y_pred),  # True Positive Rate
            'specificity': recall_score(self.y_test, self.y_pred, pos_label=0),  # True Negative Rate
            'f1': f1_score(self.y_test, self.y_pred),
        }
        
        # Medical scoring: prioritize sensitivity (minimize false negatives)
        # Custom score: (Sensitivity * 0.7) + (Specificity * 0.3) - (Precision * 0.2)
        self.metrics['final_score'] = (
            self.metrics['sensitivity'] * 0.7 +
            self.metrics['specificity'] * 0.3 -
            (1 - self.metrics['precision']) * 0.2
        )
        
        return self.metrics
    
    def print_metrics(self):
        """Print performance metrics"""
        print(f"\n{'='*60}")
        print(f"Performance Metrics for {self.name}")
        print(f"{'='*60}")
        print(f"Accuracy:     {self.metrics['accuracy']:.4f}")
        print(f"Precision:    {self.metrics['precision']:.4f}")
        print(f"Sensitivity:  {self.metrics['sensitivity']:.4f} (catch malignancies)")
        print(f"Specificity:  {self.metrics['specificity']:.4f} (identify benign)")
        print(f"F1-Score:     {self.metrics['f1']:.4f}")
        print(f"Final Score:  {self.metrics['final_score']:.4f}")
        print(f"{'='*60}\n")
    
    def get_confusion_matrix(self):
        """Get confusion matrix"""
        return confusion_matrix(self.y_test, self.y_pred)
    
    def save_model(self, filepath):
        """Save trained model"""
        dump((self.model, self.scaler), filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        self.model, self.scaler = load(filepath)
        print(f"Model loaded from {filepath}")
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Args:
            X (ndarray): Features
            
        Returns:
            ndarray: Predictions
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class RandomForestThyroidClassifier(ThyroidClassifier):
    """Random Forest classifier for thyroid nodules"""
    
    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        """
        Initialize Random Forest classifier
        
        Args:
            n_estimators (int): Number of trees
            max_depth (int): Maximum tree depth
            random_state (int): Random seed
        """
        super().__init__("Random Forest")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )


class LogisticRegressionThyroidClassifier(ThyroidClassifier):
    """Logistic Regression classifier for thyroid nodules"""
    
    def __init__(self, max_iter=1000, random_state=42):
        """
        Initialize Logistic Regression classifier
        
        Args:
            max_iter (int): Maximum iterations
            random_state (int): Random seed
        """
        super().__init__("Logistic Regression")
        self.model = LogisticRegression(max_iter=max_iter, random_state=random_state)


class LinearSVMThyroidClassifier(ThyroidClassifier):
    """Linear SVM classifier for thyroid nodules"""
    
    def __init__(self, C=1.0, random_state=42):
        """
        Initialize Linear SVM classifier
        
        Args:
            C (float): Regularization parameter
            random_state (int): Random seed
        """
        super().__init__("Linear SVM")
        self.model = SVC(kernel='linear', C=C, random_state=random_state, probability=True)


class RBFSVMThyroidClassifier(ThyroidClassifier):
    """RBF SVM classifier for thyroid nodules (Best performer)"""
    
    def __init__(self, C=1.0, gamma='scale', random_state=42):
        """
        Initialize RBF SVM classifier
        
        Args:
            C (float): Regularization parameter
            gamma (str): Kernel coefficient
            random_state (int): Random seed
        """
        super().__init__("RBF SVM")
        self.model = SVC(kernel='rbf', C=C, gamma=gamma, 
                        random_state=random_state, probability=True)


class ClassifierEvaluator:
    """
    Evaluates and compares multiple classifiers
    """
    
    def __init__(self):
        """Initialize evaluator"""
        self.classifiers = {}
        self.results_df = None
    
    def add_classifier(self, classifier):
        """
        Add classifier to evaluation
        
        Args:
            classifier (ThyroidClassifier): Classifier instance
        """
        self.classifiers[classifier.name] = classifier
    
    def train_all(self, X, y, test_size=0.2):
        """
        Train all classifiers
        
        Args:
            X (ndarray): Features
            y (ndarray): Labels
            test_size (float): Test split ratio
        """
        for name, classifier in self.classifiers.items():
            print(f"\n{'='*60}")
            print(f"Training: {name}")
            print(f"{'='*60}")
            classifier.train(X, y, test_size=test_size)
            classifier.evaluate()
            classifier.print_metrics()
    
    def compare_classifiers(self):
        """
        Compare all classifiers
        
        Returns:
            pd.DataFrame: Comparison results
        """
        results = []
        for name, classifier in self.classifiers.items():
            results.append({
                'Classifier': name,
                'Accuracy': classifier.metrics['accuracy'],
                'Precision': classifier.metrics['precision'],
                'Sensitivity': classifier.metrics['sensitivity'],
                'Specificity': classifier.metrics['specificity'],
                'F1-Score': classifier.metrics['f1'],
                'Final Score': classifier.metrics['final_score']
            })
        
        self.results_df = pd.DataFrame(results)
        return self.results_df
    
    def print_comparison(self):
        """Print classifier comparison"""
        print("\n" + "="*100)
        print("CLASSIFIER COMPARISON")
        print("="*100)
        print(self.results_df.to_string(index=False))
        print("="*100 + "\n")
        
        # Find best classifier by final score
        best_idx = self.results_df['Final Score'].idxmax()
        best_clf = self.results_df.iloc[best_idx]
        print(f"🏆 Best Classifier: {best_clf['Classifier']} (Final Score: {best_clf['Final Score']:.4f})")
        print(f"   Sensitivity: {best_clf['Sensitivity']:.4f} (Catches {best_clf['Sensitivity']*100:.2f}% of malignancies)")
        print(f"   Specificity: {best_clf['Specificity']:.4f} (Identifies {best_clf['Specificity']*100:.2f}% as benign)\n")
    
    def plot_roc_curves(self, save_path=None):
        """
        Plot ROC curves for all classifiers
        
        Args:
            save_path (str): Optional path to save figure
        """
        plt.figure(figsize=(10, 8))
        
        for name, classifier in self.classifiers.items():
            if classifier.y_pred_proba is not None:
                fpr, tpr, _ = roc_curve(classifier.y_test, classifier.y_pred_proba)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Thyroid Nodule Classification')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_confusion_matrices(self, save_path=None):
        """
        Plot confusion matrices for all classifiers
        
        Args:
            save_path (str): Optional path to save figure
        """
        n_classifiers = len(self.classifiers)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for idx, (name, classifier) in enumerate(self.classifiers.items()):
            cm = classifier.get_confusion_matrix()
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Benign', 'Suspicious'],
                       yticklabels=['Benign', 'Suspicious'])
            
            axes[idx].set_title(f'{name}\nAccuracy: {classifier.metrics["accuracy"]:.4f}')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        # Hide unused subplots
        for idx in range(n_classifiers, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()


if __name__ == "__main__":
    print("Classification modules loaded successfully")
