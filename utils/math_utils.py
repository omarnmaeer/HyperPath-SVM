# File: hyperpath_svm/utils/math_utils.py

"""
Mathematical Utilities and Optimizations for HyperPath-SVM

This module provides high-performance mathematical operations, optimization algorithms,
and numerical methods specifically designed for network routing and machine learning.

"""

import numpy as np
import scipy as sp
from scipy import linalg, sparse, optimize, signal, stats
from scipy.special import gamma, beta, erf, erfc
import numba
from numba import jit, njit, prange
import logging
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from .logging_utils import get_logger


@dataclass
class OptimizationResult:
    """Result container for optimization problems."""
    success: bool = False
    x: Optional[np.ndarray] = None
    fun: float = float('inf')
    nit: int = 0
    message: str = ""
    jac: Optional[np.ndarray] = None
    hess: Optional[np.ndarray] = None
    convergence_history: List[float] = None


class MathUtils:
    """Collection of mathematical utility functions."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self._cache = {}  # Simple caching mechanism
        self.numerical_precision = np.finfo(float).eps
    
    # ==================== Linear Algebra Operations ====================
    
    @staticmethod
    @njit(parallel=True, fastmath=True)
    def fast_matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    
        m, k = A.shape
        k2, n = B.shape
        
        if k != k2:
            raise ValueError("Matrix dimensions don't match for multiplication")
        
        C = np.zeros((m, n), dtype=A.dtype)
        
        for i in prange(m):
            for j in prange(n):
                for l in range(k):
                    C[i, j] += A[i, l] * B[l, j]
        
        return C
    
    def efficient_svd(self, matrix: np.ndarray, k: Optional[int] = None, 
                     method: str = 'truncated') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
      
        try:
            if method == 'randomized' and k is not None:
                # Randomized SVD for large matrices
                from sklearn.utils.extmath import randomized_svd
                return randomized_svd(matrix, n_components=k, random_state=42)
            
            elif method == 'truncated' and k is not None:
                # Truncated SVD using sparse methods
                if sparse.issparse(matrix):
                    from scipy.sparse.linalg import svds
                    U, S, Vt = svds(matrix, k=min(k, min(matrix.shape) - 1))
                    # Sort by singular values (descending)
                    idx = np.argsort(S)[::-1]
                    return U[:, idx], S[idx], Vt[idx, :]
                else:
                    U, S, Vt = linalg.svd(matrix, full_matrices=False)
                    return U[:, :k], S[:k], Vt[:k, :]
            
            else:
                # Full SVD
                return linalg.svd(matrix, full_matrices=(method == 'full'))
                
        except Exception as e:
            self.logger.error(f"SVD computation failed: {str(e)}")
            raise
    
    def stable_cholesky(self, matrix: np.ndarray, regularization: float = 1e-10) -> np.ndarray:
      
        try:
            # Add regularization to ensure positive definiteness
            regularized_matrix = matrix + regularization * np.eye(matrix.shape[0])
            
            # Attempt Cholesky decomposition
            try:
                L = linalg.cholesky(regularized_matrix, lower=True)
                return L
            except linalg.LinAlgError:
                # Fallback to eigenvalue decomposition
                eigenvals, eigenvecs = linalg.eigh(regularized_matrix)
                eigenvals = np.maximum(eigenvals, regularization)  # Ensure positive
                sqrt_eigenvals = np.sqrt(eigenvals)
                L = eigenvecs * sqrt_eigenvals
                return L
                
        except Exception as e:
            self.logger.error(f"Cholesky decomposition failed: {str(e)}")
            raise
    
    def matrix_condition_number(self, matrix: np.ndarray, norm: str = 'fro') -> float:
       
        try:
            if norm == 'fro':
                return linalg.norm(matrix, 'fro') * linalg.norm(linalg.pinv(matrix), 'fro')
            elif norm == '2':
                singular_values = linalg.svd(matrix, compute_uv=False)
                return singular_values[0] / singular_values[-1] if singular_values[-1] > 0 else float('inf')
            else:
                return linalg.norm(matrix, norm) * linalg.norm(linalg.pinv(matrix), norm)
                
        except Exception as e:
            self.logger.warning(f"Condition number computation failed: {str(e)}")
            return float('inf')
    
    # ==================== Quantum-Inspired Functions ====================
    
    def quantum_fourier_transform(self, vector: np.ndarray) -> np.ndarray:
      
        n = len(vector)
        if n & (n - 1) != 0:  # Check if n is power of 2
            # Pad to next power of 2
            next_pow2 = 1 << (n - 1).bit_length()
            padded_vector = np.zeros(next_pow2, dtype=complex)
            padded_vector[:n] = vector
            vector = padded_vector
            n = next_pow2
        
        # Apply QFT matrix
        qft_matrix = self._generate_qft_matrix(n)
        return np.dot(qft_matrix, vector)
    
    def _generate_qft_matrix(self, n: int) -> np.ndarray:
        """Generate QFT matrix for n qubits."""
        omega = np.exp(2j * np.pi / n)
        qft_matrix = np.zeros((n, n), dtype=complex)
        
        for i in range(n):
            for j in range(n):
                qft_matrix[i, j] = omega ** (i * j) / np.sqrt(n)
        
        return qft_matrix
    
    def quantum_phase_estimation(self, unitary: np.ndarray, eigenvector: np.ndarray,
                                precision_bits: int = 8) -> float:
     
        try:
            # Verify eigenvector
            result = np.dot(unitary, eigenvector)
            eigenvalue = np.dot(np.conj(eigenvector), result) / np.dot(np.conj(eigenvector), eigenvector)
            
            # Extract phase
            phase = np.angle(eigenvalue) / (2 * np.pi)
            if phase < 0:
                phase += 1
            
            # Quantize to precision bits
            quantized_phase = np.round(phase * (2 ** precision_bits)) / (2 ** precision_bits)
            
            return quantized_phase
            
        except Exception as e:
            self.logger.error(f"Quantum phase estimation failed: {str(e)}")
            return 0.0
    
    def quantum_amplitude_amplification(self, success_probability: float, 
                                      iterations: Optional[int] = None) -> float:
    
        if success_probability <= 0 or success_probability >= 1:
            return success_probability
        
        # Calculate optimal number of iterations
        if iterations is None:
            theta = 2 * np.arcsin(np.sqrt(success_probability))
            iterations = int(np.pi / (4 * theta) - 0.5)
        
        # Apply amplitude amplification
        theta = 2 * np.arcsin(np.sqrt(success_probability))
        amplified_angle = (2 * iterations + 1) * theta
        amplified_probability = np.sin(amplified_angle / 2) ** 2
        
        return min(amplified_probability, 1.0)
    
    # ==================== Optimization Algorithms ====================
    
    def gradient_descent(self, objective: Callable, gradient: Callable, x0: np.ndarray,
                        learning_rate: float = 0.01, max_iterations: int = 1000,
                        tolerance: float = 1e-6, momentum: float = 0.0) -> OptimizationResult:
        
        try:
            x = x0.copy()
            velocity = np.zeros_like(x)
            convergence_history = []
            
            for iteration in range(max_iterations):
                # Compute gradient
                grad = gradient(x)
                
                # Update with momentum
                velocity = momentum * velocity - learning_rate * grad
                x = x + velocity
                
                # Evaluate objective
                current_obj = objective(x)
                convergence_history.append(current_obj)
                
                # Check convergence
                if np.linalg.norm(grad) < tolerance:
                    return OptimizationResult(
                        success=True,
                        x=x,
                        fun=current_obj,
                        nit=iteration + 1,
                        message="Converged",
                        jac=grad,
                        convergence_history=convergence_history
                    )
                
                # Adaptive learning rate (simple decay)
                if iteration > 0 and convergence_history[-1] > convergence_history[-2]:
                    learning_rate *= 0.95
            
            return OptimizationResult(
                success=False,
                x=x,
                fun=objective(x),
                nit=max_iterations,
                message="Maximum iterations reached",
                convergence_history=convergence_history
            )
            
        except Exception as e:
            self.logger.error(f"Gradient descent failed: {str(e)}")
            return OptimizationResult(success=False, message=str(e))
    
    def conjugate_gradient(self, A: np.ndarray, b: np.ndarray, x0: Optional[np.ndarray] = None,
                          max_iterations: Optional[int] = None, tolerance: float = 1e-6) -> OptimizationResult:
     
        try:
            n = len(b)
            if x0 is None:
                x = np.zeros(n)
            else:
                x = x0.copy()
            
            if max_iterations is None:
                max_iterations = n
            
            # Initial residual
            r = b - np.dot(A, x)
            p = r.copy()
            rsold = np.dot(r, r)
            
            convergence_history = []
            
            for iteration in range(max_iterations):
                Ap = np.dot(A, p)
                alpha = rsold / np.dot(p, Ap)
                x = x + alpha * p
                r = r - alpha * Ap
                rsnew = np.dot(r, r)
                
                convergence_history.append(np.sqrt(rsnew))
                
                if np.sqrt(rsnew) < tolerance:
                    return OptimizationResult(
                        success=True,
                        x=x,
                        fun=0.5 * np.dot(x, np.dot(A, x)) - np.dot(b, x),
                        nit=iteration + 1,
                        message="Converged",
                        convergence_history=convergence_history
                    )
                
                beta = rsnew / rsold
                p = r + beta * p
                rsold = rsnew
            
            return OptimizationResult(
                success=False,
                x=x,
                fun=0.5 * np.dot(x, np.dot(A, x)) - np.dot(b, x),
                nit=max_iterations,
                message="Maximum iterations reached",
                convergence_history=convergence_history
            )
            
        except Exception as e:
            self.logger.error(f"Conjugate gradient failed: {str(e)}")
            return OptimizationResult(success=False, message=str(e))
    
    def quasi_newton_bfgs(self, objective: Callable, gradient: Callable, x0: np.ndarray,
                         max_iterations: int = 1000, tolerance: float = 1e-6) -> OptimizationResult:
    
        try:
            n = len(x0)
            x = x0.copy()
            H = np.eye(n)  # Initial Hessian approximation
            
            convergence_history = []
            
            for iteration in range(max_iterations):
                grad = gradient(x)
                convergence_history.append(objective(x))
                
                # Check convergence
                if np.linalg.norm(grad) < tolerance:
                    return OptimizationResult(
                        success=True,
                        x=x,
                        fun=objective(x),
                        nit=iteration + 1,
                        message="Converged",
                        jac=grad,
                        hess=H,
                        convergence_history=convergence_history
                    )
                
                # Compute search direction
                p = -np.dot(H, grad)
                
                # Line search (simple backtracking)
                alpha = self._backtracking_line_search(objective, gradient, x, p)
                
                # Update position
                x_new = x + alpha * p
                grad_new = gradient(x_new)
                
                # BFGS update
                s = x_new - x
                y = grad_new - grad
                
                if np.dot(y, s) > 1e-10:  # Curvature condition
                    rho = 1.0 / np.dot(y, s)
                    A1 = np.eye(n) - rho * np.outer(s, y)
                    A2 = np.eye(n) - rho * np.outer(y, s)
                    H = np.dot(A1, np.dot(H, A2)) + rho * np.outer(s, s)
                
                x = x_new
            
            return OptimizationResult(
                success=False,
                x=x,
                fun=objective(x),
                nit=max_iterations,
                message="Maximum iterations reached",
                convergence_history=convergence_history
            )
            
        except Exception as e:
            self.logger.error(f"BFGS optimization failed: {str(e)}")
            return OptimizationResult(success=False, message=str(e))
    
    def _backtracking_line_search(self, objective: Callable, gradient: Callable,
                                 x: np.ndarray, p: np.ndarray, 
                                 alpha_init: float = 1.0, c1: float = 1e-4,
                                 rho: float = 0.5, max_iterations: int = 50) -> float:
        """Backtracking line search for step size selection."""
        alpha = alpha_init
        fx = objective(x)
        gxp = np.dot(gradient(x), p)
        
        for _ in range(max_iterations):
            if objective(x + alpha * p) <= fx + c1 * alpha * gxp:
                return alpha
            alpha *= rho
        
        return alpha
    
    # ==================== Statistical Functions ====================
    
    @staticmethod
    @njit(fastmath=True)
    def fast_correlation(x: np.ndarray, y: np.ndarray) -> float:
     
        n = len(x)
        if n != len(y):
            return 0.0
        
        # Compute means
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        
        # Compute correlation
        numerator = 0.0
        sum_sq_x = 0.0
        sum_sq_y = 0.0
        
        for i in range(n):
            dx = x[i] - mean_x
            dy = y[i] - mean_y
            numerator += dx * dy
            sum_sq_x += dx * dx
            sum_sq_y += dy * dy
        
        denominator = np.sqrt(sum_sq_x * sum_sq_y)
        
        if denominator == 0.0:
            return 0.0
        
        return numerator / denominator
    
    def multivariate_normal_pdf(self, x: np.ndarray, mean: np.ndarray, 
                               cov: np.ndarray, log: bool = False) -> Union[float, np.ndarray]:
    
        try:
            d = len(mean)
            
            # Ensure proper shapes
            if x.ndim == 1:
                x = x.reshape(1, -1)
            
            diff = x - mean
            
            # Compute log probability using Cholesky decomposition
            L = self.stable_cholesky(cov)
            log_det = 2 * np.sum(np.log(np.diag(L)))
            
            # Solve L y = diff^T for each point
            y = linalg.solve_triangular(L, diff.T, lower=True)
            
            # Compute quadratic form
            quadratic_form = np.sum(y**2, axis=0)
            
            # Log probability
            log_prob = -0.5 * (d * np.log(2 * np.pi) + log_det + quadratic_form)
            
            if log:
                return log_prob
            else:
                return np.exp(log_prob)
                
        except Exception as e:
            self.logger.error(f"Multivariate normal PDF computation failed: {str(e)}")
            return np.nan
    
    def kernel_density_estimation(self, data: np.ndarray, bandwidth: Optional[float] = None,
                                 kernel: str = 'gaussian') -> Callable:
      
        try:
            n, d = data.shape if data.ndim > 1 else (len(data), 1)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
                d = 1
            
            # Silverman's rule of thumb for bandwidth
            if bandwidth is None:
                bandwidth = (4 / (d + 2)) ** (1 / (d + 4)) * n ** (-1 / (d + 4))
                if d == 1:
                    bandwidth *= np.std(data)
                else:
                    bandwidth *= np.mean(np.std(data, axis=0))
            
            def kde_function(x: np.ndarray) -> Union[float, np.ndarray]:
                """Evaluate KDE at given points."""
                if x.ndim == 1 and d > 1:
                    x = x.reshape(1, -1)
                elif x.ndim == 1 and d == 1:
                    x = x.reshape(-1, 1)
                
                density = np.zeros(x.shape[0] if x.ndim > 1 else 1)
                
                for i, xi in enumerate(x):
                    if d == 1:
                        xi = xi.reshape(-1)
                    
                    # Compute kernel values
                    distances = np.linalg.norm(data - xi, axis=1) / bandwidth
                    
                    if kernel == 'gaussian':
                        kernel_values = np.exp(-0.5 * distances**2) / np.sqrt(2 * np.pi)
                    elif kernel == 'epanechnikov':
                        kernel_values = np.where(distances <= 1, 
                                               0.75 * (1 - distances**2), 0)
                    elif kernel == 'uniform':
                        kernel_values = np.where(distances <= 1, 0.5, 0)
                    else:
                        raise ValueError(f"Unknown kernel: {kernel}")
                    
                    density[i] = np.mean(kernel_values) / (bandwidth**d)
                
                return density[0] if len(density) == 1 else density
            
            return kde_function
            
        except Exception as e:
            self.logger.error(f"KDE computation failed: {str(e)}")
            return lambda x: np.zeros_like(x)
    
    # ==================== Signal Processing ====================
    
    def adaptive_filter_lms(self, input_signal: np.ndarray, desired_signal: np.ndarray,
                           filter_length: int, step_size: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
     
        try:
            n_samples = len(input_signal)
            
            # Initialize filter coefficients
            w = np.zeros(filter_length)
            output_signal = np.zeros(n_samples)
            
            # Input buffer
            x_buffer = np.zeros(filter_length)
            
            for n in range(n_samples):
                # Update input buffer
                x_buffer[1:] = x_buffer[:-1]
                x_buffer[0] = input_signal[n]
                
                # Filter output
                y = np.dot(w, x_buffer)
                output_signal[n] = y
                
                # Error signal
                error = desired_signal[n] - y
                
                # LMS update
                w = w + step_size * error * x_buffer
            
            return output_signal, w
            
        except Exception as e:
            self.logger.error(f"LMS adaptive filter failed: {str(e)}")
            return np.zeros_like(input_signal), np.zeros(filter_length)
    
    def spectral_analysis(self, signal: np.ndarray, sampling_rate: float,
                         window: str = 'hann', nperseg: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
       
        try:
            if nperseg is None:
                nperseg = min(len(signal) // 8, 256)
            
            frequencies, psd = signal.welch(signal, fs=sampling_rate, 
                                          window=window, nperseg=nperseg)
            
            return frequencies, psd
            
        except Exception as e:
            self.logger.error(f"Spectral analysis failed: {str(e)}")
            return np.array([]), np.array([])
    
    # ==================== Numerical Methods ====================
    
    def numerical_gradient(self, func: Callable, x: np.ndarray, h: float = 1e-8) -> np.ndarray:
       
        try:
            n = len(x)
            grad = np.zeros(n)
            
            for i in range(n):
                x_plus = x.copy()
                x_minus = x.copy()
                x_plus[i] += h
                x_minus[i] -= h
                
                grad[i] = (func(x_plus) - func(x_minus)) / (2 * h)
            
            return grad
            
        except Exception as e:
            self.logger.error(f"Numerical gradient computation failed: {str(e)}")
            return np.zeros_like(x)
    
    def numerical_hessian(self, func: Callable, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
       
        try:
            n = len(x)
            hess = np.zeros((n, n))
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        # Diagonal elements
                        x_plus = x.copy()
                        x_minus = x.copy()
                        x_plus[i] += h
                        x_minus[i] -= h
                        
                        hess[i, j] = (func(x_plus) - 2*func(x) + func(x_minus)) / (h**2)
                    else:
                        # Off-diagonal elements
                        x_pp = x.copy()
                        x_pm = x.copy()
                        x_mp = x.copy()
                        x_mm = x.copy()
                        
                        x_pp[i] += h
                        x_pp[j] += h
                        x_pm[i] += h
                        x_pm[j] -= h
                        x_mp[i] -= h
                        x_mp[j] += h
                        x_mm[i] -= h
                        x_mm[j] -= h
                        
                        hess[i, j] = (func(x_pp) - func(x_pm) - func(x_mp) + func(x_mm)) / (4 * h**2)
            
            return hess
            
        except Exception as e:
            self.logger.error(f"Numerical Hessian computation failed: {str(e)}")
            return np.eye(len(x))
    
    def adaptive_quadrature(self, func: Callable, a: float, b: float, 
                           tolerance: float = 1e-8, max_depth: int = 10) -> float:
        
        try:
            def simpson_rule(f, x0, x2):
                """Simpson's rule for interval [x0, x2]."""
                x1 = (x0 + x2) / 2
                return (x2 - x0) / 6 * (f(x0) + 4*f(x1) + f(x2))
            
            def adaptive_simpson(f, x0, x2, tol, depth):
                """Recursive adaptive Simpson's rule."""
                if depth > max_depth:
                    return simpson_rule(f, x0, x2)
                
                x1 = (x0 + x2) / 2
                s1 = simpson_rule(f, x0, x2)
                s2 = simpson_rule(f, x0, x1) + simpson_rule(f, x1, x2)
                
                if abs(s2 - s1) < 15 * tol:  # Error estimate
                    return s2 + (s2 - s1) / 15
                else:
                    left = adaptive_simpson(f, x0, x1, tol/2, depth+1)
                    right = adaptive_simpson(f, x1, x2, tol/2, depth+1)
                    return left + right
            
            return adaptive_simpson(func, a, b, tolerance, 0)
            
        except Exception as e:
            self.logger.error(f"Adaptive quadrature failed: {str(e)}")
            return 0.0
    
    # ==================== Utility Functions ====================
    
    def numerical_stability_check(self, value: Union[float, np.ndarray], 
                                 name: str = "value") -> Union[float, np.ndarray]:
       
        if isinstance(value, np.ndarray):
            # Check for NaN or infinity
            if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                self.logger.warning(f"Numerical instability detected in {name}")
                value = np.nan_to_num(value, nan=0.0, posinf=1e10, neginf=-1e10)
            
            # Check for very small numbers
            small_mask = np.abs(value) < self.numerical_precision
            if np.any(small_mask):
                value[small_mask] = 0.0
        
        else:
            if np.isnan(value) or np.isinf(value):
                self.logger.warning(f"Numerical instability detected in {name}")
                value = 0.0 if np.isnan(value) else (1e10 if value > 0 else -1e10)
            
            if abs(value) < self.numerical_precision:
                value = 0.0
        
        return value
    
    def cache_result(self, key: str, result: Any):
       
        self._cache[key] = result
    
    def get_cached_result(self, key: str) -> Any:
        
        return self._cache.get(key)
    
    def clear_cache(self):
        """Clear computation cache."""
        self._cache.clear()


# Global instance for easy access
math_utils = MathUtils()


# Convenience functions
def fast_svd(matrix: np.ndarray, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  
    return math_utils.efficient_svd(matrix, k)


def stable_inverse(matrix: np.ndarray, regularization: float = 1e-10) -> np.ndarray:
   
    try:
        return linalg.inv(matrix + regularization * np.eye(matrix.shape[0]))
    except linalg.LinAlgError:
        return linalg.pinv(matrix)


def safe_log(x: Union[float, np.ndarray], epsilon: float = 1e-15) -> Union[float, np.ndarray]:
    
    if isinstance(x, np.ndarray):
        return np.log(np.maximum(x, epsilon))
    else:
        return np.log(max(x, epsilon))


def safe_divide(numerator: Union[float, np.ndarray], 
                denominator: Union[float, np.ndarray],
                epsilon: float = 1e-15) -> Union[float, np.ndarray]:
   
    if isinstance(denominator, np.ndarray):
        safe_denom = np.where(np.abs(denominator) < epsilon, epsilon, denominator)
    else:
        safe_denom = epsilon if abs(denominator) < epsilon else denominator
    
    return numerator / safe_denom


if __name__ == "__main__":
    # Test mathematical utilities
    logger = get_logger(__name__)
    logger.info("Testing mathematical utilities...")
    
    # Test matrix operations
    np.random.seed(42)
    A = np.random.random((10, 10))
    A = A @ A.T  # Make positive definite
    
    # Test SVD
    U, S, Vt = math_utils.efficient_svd(A, k=5)
    logger.info(f"SVD test: U shape={U.shape}, S shape={S.shape}, Vt shape={Vt.shape}")
    
    # Test Cholesky
    L = math_utils.stable_cholesky(A)
    reconstruction_error = np.linalg.norm(L @ L.T - A)
    logger.info(f"Cholesky test: Reconstruction error={reconstruction_error:.2e}")
    
    # Test optimization
    def quadratic(x):
        return 0.5 * x.T @ A @ x - np.ones(len(x)).T @ x
    
    def quadratic_grad(x):
        return A @ x - np.ones(len(x))
    
    x0 = np.random.random(10)
    result = math_utils.gradient_descent(quadratic, quadratic_grad, x0, 
                                       learning_rate=0.01, max_iterations=100)
    logger.info(f"Optimization test: Success={result.success}, Iterations={result.nit}")
    
    # Test statistical functions
    x = np.random.normal(0, 1, 1000)
    y = 2 * x + np.random.normal(0, 0.5, 1000)
    correlation = math_utils.fast_correlation(x, y)
    logger.info(f"Correlation test: r={correlation:.4f}")
    
    # Test quantum functions
    state_vector = np.random.random(8) + 1j * np.random.random(8)
    state_vector /= np.linalg.norm(state_vector)
    qft_result = math_utils.quantum_fourier_transform(state_vector)
    logger.info(f"QFT test: Input norm={np.linalg.norm(state_vector):.4f}, "
               f"Output norm={np.linalg.norm(qft_result):.4f}")
    
    logger.info("Mathematical utilities testing completed!") 
