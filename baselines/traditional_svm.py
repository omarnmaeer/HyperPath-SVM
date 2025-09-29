# File: hyperpath_svm/baselines/traditional_svm.py

"""
Traditional SVM Baseline Models for HyperPath-SVM Comparison

This module implements traditional SVM variants that serve as baselines:
- Static SVM: Standard SVM with fixed parameters
- Weighted SVM: SVM with class weight balancing for imbalanced data
- Quantum SVM: SVM with quantum-inspired kernel enhancements
- Ensemble SVM: Multiple SVM ensemble with voting
- Online SVM: Incremental learning SVM for streaming data

"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier, BaggingClassifier
from scipy.spatial.distance import pdist, squareform
from scipy.special import gamma
import joblib
from concurrent.futures import ThreadPoolExecutor

from ..utils.logging_utils import get_logger


@dataclass
class SVMConfig:
    
    C: float = 1.0
    kernel: str = 'rbf'
    gamma: Union[str, float] = 'scale'
    degree: int = 3
    coef0: float = 0.0
    shrinking: bool = True
    probability: bool = True
    tol: float = 1e-3
    cache_size: float = 200
    class_weight: Optional[Union[str, dict]] = None
    verbose: bool = False
    max_iter: int = -1
    decision_function_shape: str = 'ovr'
    break_ties: bool = False
    random_state: int = 42


class BaseSVM(BaseEstimator, ClassifierMixin, ABC):
    
    
    def __init__(self, config: Optional[SVMConfig] = None, **kwargs):
        self.config = config or SVMConfig()
        self.logger = get_logger(__name__)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.is_fitted = False
        self.training_time = 0.0
        self.feature_importances_ = None
        
        # Override config with kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    @abstractmethod
    def _build_model(self) -> BaseEstimator:
        """Build the SVM model."""
        pass
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseSVM':
        """Train the SVM model."""
        try:
            self.logger.info(f"Training {self.__class__.__name__}")
            start_time = time.time()
            
            # Prepare data
            X_scaled = self.scaler.fit_transform(X)
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Build and train model
            self.model = self._build_model()
            self.model.fit(X_scaled, y_encoded)
            
            # Calculate training time
            self.training_time = time.time() - start_time
            self.is_fitted = True
            
            # Compute feature importances if possible
            self._compute_feature_importance(X_scaled, y_encoded)
            
            self.logger.info(f"{self.__class__.__name__} training completed in {self.training_time:.2f}s")
            return self
            
        except Exception as e:
            self.logger.error(f"Training failed for {self.__class__.__name__}: {str(e)}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
      
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            X_scaled = self.scaler.transform(X)
            y_pred_encoded = self.model.predict(X_scaled)
            y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
            return y_pred
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
       
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            X_scaled = self.scaler.transform(X)
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X_scaled)
            else:
                # For models without probability, use decision function
                if hasattr(self.model, 'decision_function'):
                    scores = self.model.decision_function(X_scaled)
                    # Convert to probabilities using softmax
                    if scores.ndim == 1:  # Binary classification
                        proba = np.exp(scores) / (1 + np.exp(scores))
                        return np.column_stack([1 - proba, proba])
                    else:  # Multi-class
                        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
                else:
                    raise ValueError("Model does not support probability prediction")
                    
        except Exception as e:
            self.logger.error(f"Probability prediction failed: {str(e)}")
            raise
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
      
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            X_scaled = self.scaler.transform(X)
            return self.model.decision_function(X_scaled)
            
        except Exception as e:
            self.logger.error(f"Decision function computation failed: {str(e)}")
            raise
    
    def _compute_feature_importance(self, X: np.ndarray, y: np.ndarray):
       
        try:
            if hasattr(self.model, 'coef_'):
                # Linear kernel - use coefficients
                self.feature_importances_ = np.abs(self.model.coef_).mean(axis=0)
            else:
                # Non-linear kernel - use permutation importance
                baseline_score = accuracy_score(y, self.model.predict(X))
                importances = []
                
                for i in range(X.shape[1]):
                    X_permuted = X.copy()
                    np.random.shuffle(X_permuted[:, i])
                    permuted_score = accuracy_score(y, self.model.predict(X_permuted))
                    importance = baseline_score - permuted_score
                    importances.append(max(0, importance))  # Ensure non-negative
                
                self.feature_importances_ = np.array(importances)
                
        except Exception as e:
            self.logger.warning(f"Feature importance computation failed: {str(e)}")
            self.feature_importances_ = np.zeros(X.shape[1])
    
    def get_support_vectors(self) -> Dict[str, Any]:
       
        if not self.is_fitted:
            raise ValueError("Model must be fitted")
        
        try:
            return {
                'support_vectors': self.model.support_vectors_,
                'support_indices': self.model.support_,
                'n_support': self.model.n_support_,
                'dual_coef': self.model.dual_coef_
            }
        except AttributeError:
            return {}
    
    def save_model(self, filepath: str):
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        try:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'config': self.config,
                'feature_importances': self.feature_importances_
            }, filepath)
            
            self.logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Model saving failed: {str(e)}")
            raise
    
    def load_model(self, filepath: str):
       
        try:
            saved_data = joblib.load(filepath)
            self.model = saved_data['model']
            self.scaler = saved_data['scaler']
            self.label_encoder = saved_data['label_encoder']
            self.config = saved_data.get('config', self.config)
            self.feature_importances_ = saved_data.get('feature_importances')
            self.is_fitted = True
            
            self.logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Model loading failed: {str(e)}")
            raise


class StaticSVM(BaseSVM):
   
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _build_model(self) -> SVC:
        
        return SVC(
            C=self.config.C,
            kernel=self.config.kernel,
            gamma=self.config.gamma,
            degree=self.config.degree,
            coef0=self.config.coef0,
            shrinking=self.config.shrinking,
            probability=self.config.probability,
            tol=self.config.tol,
            cache_size=self.config.cache_size,
            class_weight=self.config.class_weight,
            verbose=self.config.verbose,
            max_iter=self.config.max_iter,
            decision_function_shape=self.config.decision_function_shape,
            break_ties=self.config.break_ties,
            random_state=self.config.random_state
        )


class WeightedSVM(BaseSVM):
    
    
    def __init__(self, class_weight: str = 'balanced', **kwargs):
        config = kwargs.get('config', SVMConfig())
        config.class_weight = class_weight
        kwargs['config'] = config
        super().__init__(**kwargs)
    
    def _build_model(self) -> SVC:
      
        return SVC(
            C=self.config.C,
            kernel=self.config.kernel,
            gamma=self.config.gamma,
            degree=self.config.degree,
            coef0=self.config.coef0,
            shrinking=self.config.shrinking,
            probability=self.config.probability,
            tol=self.config.tol,
            cache_size=self.config.cache_size,
            class_weight=self.config.class_weight,
            verbose=self.config.verbose,
            max_iter=self.config.max_iter,
            decision_function_shape=self.config.decision_function_shape,
            break_ties=self.config.break_ties,
            random_state=self.config.random_state
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'WeightedSVM':
       
        try:
            # Calculate class weights if 'balanced'
            if self.config.class_weight == 'balanced':
                unique_classes = np.unique(y)
                class_weights = compute_class_weight(
                    'balanced', classes=unique_classes, y=y
                )
                self.config.class_weight = dict(zip(unique_classes, class_weights))
                
                self.logger.info(f"Computed class weights: {self.config.class_weight}")
            
            return super().fit(X, y)
            
        except Exception as e:
            self.logger.error(f"Weighted SVM training failed: {str(e)}")
            raise


class QuantumSVM(BaseSVM):
    
    
    def __init__(self, quantum_kernel: bool = True, quantum_feature_map: str = 'z_pauli', **kwargs):
        super().__init__(**kwargs)
        self.quantum_kernel = quantum_kernel
        self.quantum_feature_map = quantum_feature_map
    
    def _build_model(self) -> SVC:
        
        if self.quantum_kernel:
            kernel_func = self._create_quantum_kernel()
            return SVC(
                C=self.config.C,
                kernel=kernel_func,
                shrinking=self.config.shrinking,
                probability=self.config.probability,
                tol=self.config.tol,
                cache_size=self.config.cache_size,
                class_weight=self.config.class_weight,
                verbose=self.config.verbose,
                max_iter=self.config.max_iter,
                decision_function_shape=self.config.decision_function_shape,
                break_ties=self.config.break_ties,
                random_state=self.config.random_state
            )
        else:
            return super()._build_model()
    
    def _create_quantum_kernel(self) -> Callable:
       
        def quantum_kernel(X, Y=None):
            
            if Y is None:
                Y = X
            
            # Apply quantum feature map
            X_mapped = self._quantum_feature_map(X)
            Y_mapped = self._quantum_feature_map(Y)
            
            # Compute inner products (quantum kernel value)
            kernel_matrix = np.dot(X_mapped, Y_mapped.T)
            
            # Apply quantum interference effect
            kernel_matrix = np.abs(kernel_matrix) ** 2
            
            return kernel_matrix
        
        return quantum_kernel
    
    def _quantum_feature_map(self, X: np.ndarray) -> np.ndarray:
        
        if self.quantum_feature_map == 'z_pauli':
            # Z-Pauli feature map: exp(i * sum(x_i * Z_i))
            # Classical approximation using trigonometric functions
            mapped = np.column_stack([
                np.cos(np.sum(X, axis=1)),
                np.sin(np.sum(X, axis=1)),
                np.cos(np.sum(X**2, axis=1)),
                np.sin(np.sum(X**2, axis=1))
            ])
        elif self.quantum_feature_map == 'zz_pauli':
            # ZZ-Pauli feature map with interaction terms
            n_features = X.shape[1]
            mapped_features = []
            
            # Single qubit rotations
            for i in range(n_features):
                mapped_features.extend([
                    np.cos(X[:, i]),
                    np.sin(X[:, i])
                ])
            
            # Two-qubit interactions (limited to avoid explosion)
            for i in range(min(5, n_features)):
                for j in range(i+1, min(5, n_features)):
                    mapped_features.extend([
                        np.cos(X[:, i] * X[:, j]),
                        np.sin(X[:, i] * X[:, j])
                    ])
            
            mapped = np.column_stack(mapped_features)
        else:
            # Default: simple trigonometric expansion
            mapped = np.column_stack([
                X,
                np.cos(X),
                np.sin(X)
            ])
        
        return mapped


class EnsembleSVM(BaseSVM):
  
    
    def __init__(self, n_estimators: int = 5, ensemble_method: str = 'voting', **kwargs):
        super().__init__(**kwargs)
        self.n_estimators = n_estimators
        self.ensemble_method = ensemble_method
        self.base_estimators = []
    
    def _build_model(self) -> BaseEstimator:
      
        # Create base estimators with different configurations
        base_models = []
        
        for i in range(self.n_estimators):
            # Vary parameters for diversity
            C_values = [0.1, 0.5, 1.0, 2.0, 5.0]
            gamma_values = ['scale', 'auto', 0.001, 0.01, 0.1]
            kernel_values = ['rbf', 'poly', 'sigmoid']
            
            estimator_config = SVMConfig(
                C=C_values[i % len(C_values)],
                gamma=gamma_values[i % len(gamma_values)],
                kernel=kernel_values[i % len(kernel_values)] if i < len(kernel_values) else 'rbf',
                probability=True,
                random_state=self.config.random_state + i
            )
            
            base_model = SVC(
                C=estimator_config.C,
                kernel=estimator_config.kernel,
                gamma=estimator_config.gamma,
                probability=estimator_config.probability,
                random_state=estimator_config.random_state
            )
            
            base_models.append((f'svm_{i}', base_model))
            self.base_estimators.append(estimator_config)
        
        if self.ensemble_method == 'voting':
            return VotingClassifier(
                estimators=base_models,
                voting='soft',  # Use probabilities
                n_jobs=-1
            )
        elif self.ensemble_method == 'bagging':
            return BaggingClassifier(
                base_estimator=SVC(
                    C=self.config.C,
                    kernel=self.config.kernel,
                    gamma=self.config.gamma,
                    probability=True,
                    random_state=self.config.random_state
                ),
                n_estimators=self.n_estimators,
                random_state=self.config.random_state,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")


class OnlineSVM(BaseSVM):
    
    
    def __init__(self, batch_size: int = 100, forgetting_factor: float = 0.9, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.forgetting_factor = forgetting_factor
        self.seen_samples = 0
        self.model_history = []
    
    def _build_model(self) -> SVC:
      
        return SVC(
            C=self.config.C,
            kernel=self.config.kernel,
            gamma=self.config.gamma,
            probability=self.config.probability,
            random_state=self.config.random_state
        )
    
    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> 'OnlineSVM':
        
        try:
            if not self.is_fitted:
                # First batch - regular fit
                return self.fit(X, y)
            
            # Scale new data
            X_scaled = self.scaler.transform(X)
            y_encoded = self.label_encoder.transform(y)
            
            # For SVM, we need to retrain with combined data
            # This is a simplified approach - true online SVM would be more complex
            
            # Get previous support vectors if available
            if hasattr(self.model, 'support_vectors_'):
                # Combine with weighted previous support vectors
                prev_sv = self.model.support_vectors_
                prev_sv_labels = self.label_encoder.transform(
                    self.label_encoder.inverse_transform(
                        self.model.predict(prev_sv)
                    )
                )
                
                # Apply forgetting factor by sampling previous data
                n_keep = int(len(prev_sv) * self.forgetting_factor)
                if n_keep > 0:
                    keep_indices = np.random.choice(len(prev_sv), n_keep, replace=False)
                    X_combined = np.vstack([prev_sv[keep_indices], X_scaled])
                    y_combined = np.hstack([prev_sv_labels[keep_indices], y_encoded])
                else:
                    X_combined, y_combined = X_scaled, y_encoded
            else:
                X_combined, y_combined = X_scaled, y_encoded
            
            # Retrain model
            self.model.fit(X_combined, y_combined)
            self.seen_samples += len(X)
            
            self.logger.info(f"Online SVM updated with {len(X)} samples "
                           f"(total seen: {self.seen_samples})")
            
            return self
            
        except Exception as e:
            self.logger.error(f"Online SVM partial fit failed: {str(e)}")
            raise
    
    def get_model_evolution(self) -> List[Dict[str, Any]]:
        
        return self.model_history


class AdaptiveSVM(BaseSVM):
  
    
    def __init__(self, param_grid: Optional[Dict] = None, cv: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.param_grid = param_grid or {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.1, 1],
            'kernel': ['rbf', 'poly']
        }
        self.cv = cv
        self.best_params_ = None
        self.grid_search_ = None
    
    def _build_model(self) -> GridSearchCV:
        
        base_svm = SVC(
            probability=self.config.probability,
            random_state=self.config.random_state
        )
        
        return GridSearchCV(
            base_svm,
            self.param_grid,
            cv=self.cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1 if self.config.verbose else 0
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AdaptiveSVM':
       
        try:
            self.logger.info(f"Training {self.__class__.__name__} with grid search")
            start_time = time.time()
            
            # Prepare data
            X_scaled = self.scaler.fit_transform(X)
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Build and train model with grid search
            self.model = self._build_model()
            self.model.fit(X_scaled, y_encoded)
            
            # Store results
            self.best_params_ = self.model.best_params_
            self.grid_search_ = self.model
            
            # Update model to best estimator
            self.model = self.model.best_estimator_
            
            # Calculate training time
            self.training_time = time.time() - start_time
            self.is_fitted = True
            
            self.logger.info(f"Best parameters: {self.best_params_}")
            self.logger.info(f"Best cross-validation score: {self.grid_search_.best_score_:.4f}")
            self.logger.info(f"Training completed in {self.training_time:.2f}s")
            
            return self
            
        except Exception as e:
            self.logger.error(f"Adaptive SVM training failed: {str(e)}")
            raise


# Factory functions for easy model creation
def create_static_svm(**kwargs) -> StaticSVM:
  
    return StaticSVM(**kwargs)


def create_weighted_svm(**kwargs) -> WeightedSVM:
   
    return WeightedSVM(**kwargs)


def create_quantum_svm(**kwargs) -> QuantumSVM:
   
    return QuantumSVM(**kwargs)


def create_ensemble_svm(**kwargs) -> EnsembleSVM:
    
    return EnsembleSVM(**kwargs)


def create_online_svm(**kwargs) -> OnlineSVM:
   
    return OnlineSVM(**kwargs)


def create_adaptive_svm(**kwargs) -> AdaptiveSVM:
    
    return AdaptiveSVM(**kwargs)


if __name__ == "__main__":
    # Test traditional SVM models
    logger = get_logger(__name__)
    logger.info("Testing traditional SVM baseline models...")
    
    # Generate test data
    np.random.seed(42)
    X_train = np.random.random((500, 10))
    y_train = np.random.randint(0, 3, 500)
    X_test = np.random.random((100, 10))
    y_test = np.random.randint(0, 3, 100)
    
    # Test each model
    models = {
        'Static SVM': create_static_svm(C=1.0),
        'Weighted SVM': create_weighted_svm(class_weight='balanced'),
        'Quantum SVM': create_quantum_svm(quantum_kernel=True),
        'Ensemble SVM': create_ensemble_svm(n_estimators=3),
        'Online SVM': create_online_svm(batch_size=50),
        'Adaptive SVM': create_adaptive_svm(cv=2)  # Reduced CV for testing
    }
    
    for name, model in models.items():
        try:
            logger.info(f"Testing {name}...")
            
            # Train and test
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Test predictions
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)
            accuracy = accuracy_score(y_test, predictions)
            
            # Additional info for specific models
            extra_info = ""
            if hasattr(model, 'best_params_'):
                extra_info = f", Best params: {model.best_params_}"
            elif hasattr(model, 'quantum_kernel'):
                extra_info = f", Quantum kernel: {model.quantum_kernel}"
            
            logger.info(f"{name}: Training time={training_time:.2f}s, "
                       f"Accuracy={accuracy:.4f}, "
                       f"Proba shape={probabilities.shape}{extra_info}")
            
            # Test online learning for OnlineSVM
            if name == 'Online SVM':
                model.partial_fit(X_test[:20], y_test[:20])
                logger.info(f"Online SVM: Updated with additional 20 samples")
            
        except Exception as e:
            logger.error(f"Error testing {name}: {str(e)}")
    
    logger.info("Traditional SVM baseline testing completed!") 
