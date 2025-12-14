# Comprehensive Evaluation Suite for Legal Text Understandability Classification
# This module provides ordinal-aware metrics, classification metrics, and calibration analysis.

import json
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Optional, Union, List, Dict, Any
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
)
from scipy import stats
from utils import setup_logger

logger = setup_logger(__name__)


# =============================================================================
# Label Utilities
# =============================================================================

# Default label ordering for this task (Hungarian legal text readability)
DEFAULT_LABELS = [
    "1-Nagyon nehezen érthető",  # Very hard to understand
    "2-Nehezen érthető",          # Hard to understand
    "3-Többé/kevésbé megértem",   # More or less understandable
    "4-Érthető",                   # Understandable
    "5-Könnyen érthető",           # Easy to understand
]


def get_label_ordering(labels: Optional[List[str]] = None) -> Dict[str, int]:
    """
    Get a mapping from label names to ordinal positions (0-indexed).
    
    Args:
        labels: Ordered list of label names. Uses DEFAULT_LABELS if None.
        
    Returns:
        Dictionary mapping label name to ordinal position.
    """
    if labels is None:
        labels = DEFAULT_LABELS
    return {label: idx for idx, label in enumerate(labels)}


def labels_to_ordinal(y: List[str], label_ordering: Optional[Dict[str, int]] = None) -> np.ndarray:
    """
    Convert string labels to ordinal integers.
    
    Args:
        y: List of string labels
        label_ordering: Mapping from label to ordinal. Uses default if None.
        
    Returns:
        NumPy array of ordinal integers.
    """
    if label_ordering is None:
        label_ordering = get_label_ordering()
    return np.array([label_ordering[label] for label in y])


# =============================================================================
# Ordinal Metrics
# =============================================================================

class OrdinalMetrics:
    """
    Ordinal-aware metrics that account for the natural ordering of classes.
    These metrics penalize predictions differently based on how far they are
    from the true class.
    """
    
    @staticmethod
    def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Error for ordinal classification.
        Measures the average distance between predicted and true classes.
        
        Args:
            y_true: True ordinal labels (integers)
            y_pred: Predicted ordinal labels (integers)
            
        Returns:
            MAE value (0 = perfect, higher = worse)
        """
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def off_by_one_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Percentage of predictions that are within ±1 of the true class.
        Also known as "adjacent accuracy" or "relaxed accuracy".
        
        Args:
            y_true: True ordinal labels (integers)
            y_pred: Predicted ordinal labels (integers)
            
        Returns:
            Proportion of predictions within 1 class of true (0-1)
        """
        return np.mean(np.abs(y_true - y_pred) <= 1)
    
    @staticmethod
    def off_by_k_accuracy(y_true: np.ndarray, y_pred: np.ndarray, k: int = 2) -> float:
        """
        Percentage of predictions that are within ±k of the true class.
        
        Args:
            y_true: True ordinal labels (integers)
            y_pred: Predicted ordinal labels (integers)
            k: Maximum allowed distance from true class
            
        Returns:
            Proportion of predictions within k classes of true (0-1)
        """
        return np.mean(np.abs(y_true - y_pred) <= k)
    
    @staticmethod
    def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Quadratic Weighted Kappa (QWK).
        Standard metric for ordinal classification that penalizes 
        distant misclassifications more heavily.
        
        - QWK = 1: Perfect agreement
        - QWK = 0: Agreement expected by chance
        - QWK < 0: Less than chance agreement
        
        Args:
            y_true: True ordinal labels (integers)
            y_pred: Predicted ordinal labels (integers)
            
        Returns:
            QWK score (-1 to 1, higher is better)
        """
        return cohen_kappa_score(y_true, y_pred, weights='quadratic')
    
    @staticmethod
    def linear_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Linear Weighted Kappa.
        Similar to QWK but with linear rather than quadratic weighting.
        
        Args:
            y_true: True ordinal labels (integers)
            y_pred: Predicted ordinal labels (integers)
            
        Returns:
            Linear weighted kappa score (-1 to 1, higher is better)
        """
        return cohen_kappa_score(y_true, y_pred, weights='linear')
    
    @staticmethod
    def kendall_tau(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
        """
        Kendall's Tau rank correlation coefficient.
        Measures the ordinal association between true and predicted rankings.
        
        Args:
            y_true: True ordinal labels (integers)
            y_pred: Predicted ordinal labels (integers)
            
        Returns:
            Tuple of (tau coefficient, p-value)
        """
        tau, p_value = stats.kendalltau(y_true, y_pred)
        return tau, p_value
    
    @staticmethod
    def spearman_rho(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
        """
        Spearman's rank correlation coefficient.
        
        Args:
            y_true: True ordinal labels (integers)
            y_pred: Predicted ordinal labels (integers)
            
        Returns:
            Tuple of (rho coefficient, p-value)
        """
        rho, p_value = stats.spearmanr(y_true, y_pred)
        return rho, p_value
    
    @staticmethod
    def ordinal_error_distribution(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[int, int]:
        """
        Distribution of ordinal errors (how many off by 0, 1, 2, etc.)
        
        Args:
            y_true: True ordinal labels (integers)
            y_pred: Predicted ordinal labels (integers)
            
        Returns:
            Dictionary mapping error distance to count
        """
        errors = y_pred - y_true
        return dict(Counter(errors))


# =============================================================================
# Classification Metrics
# =============================================================================

class ClassificationMetrics:
    """Standard classification metrics."""
    
    @staticmethod
    def compute_all(
        y_true: List[str],
        y_pred: List[str],
        labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compute all standard classification metrics.
        
        Args:
            y_true: True labels (strings)
            y_pred: Predicted labels (strings)
            labels: Ordered list of all possible labels
            
        Returns:
            Dictionary containing all metrics
        """
        if labels is None:
            labels = DEFAULT_LABELS
            
        # Filter to only labels present in data
        present_labels = sorted(set(y_true) | set(y_pred))
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        }
    
    @staticmethod
    def per_class_metrics(
        y_true: List[str],
        y_pred: List[str],
        labels: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute per-class precision, recall, and F1.
        
        Args:
            y_true: True labels (strings)
            y_pred: Predicted labels (strings)
            labels: Ordered list of all possible labels
            
        Returns:
            Dictionary mapping class name to its metrics
        """
        if labels is None:
            labels = DEFAULT_LABELS
            
        report = classification_report(
            y_true, y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0
        )
        
        # Extract per-class metrics
        per_class = {}
        for label in labels:
            if label in report:
                per_class[label] = {
                    'precision': report[label]['precision'],
                    'recall': report[label]['recall'],
                    'f1': report[label]['f1-score'],
                    'support': report[label]['support'],
                }
        
        return per_class


# =============================================================================
# Confusion Analysis
# =============================================================================

class ConfusionAnalysis:
    """Confusion matrix and error analysis."""
    
    @staticmethod
    def compute_confusion_matrix(
        y_true: List[str],
        y_pred: List[str],
        labels: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Compute confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Ordered list of labels
            
        Returns:
            Confusion matrix as numpy array
        """
        if labels is None:
            labels = DEFAULT_LABELS
        return confusion_matrix(y_true, y_pred, labels=labels)
    
    @staticmethod
    def confusion_matrix_normalized(
        y_true: List[str],
        y_pred: List[str],
        labels: Optional[List[str]] = None,
        normalize: str = 'true'
    ) -> np.ndarray:
        """
        Compute normalized confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels  
            labels: Ordered list of labels
            normalize: 'true' (by row), 'pred' (by column), or 'all'
            
        Returns:
            Normalized confusion matrix
        """
        if labels is None:
            labels = DEFAULT_LABELS
        return confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    
    @staticmethod
    def ordinal_error_matrix(cm: np.ndarray) -> np.ndarray:
        """
        Convert confusion matrix to ordinal error matrix.
        Each cell shows the ordinal distance (error magnitude).
        
        Args:
            cm: Confusion matrix
            
        Returns:
            Matrix where cm[i,j] is weighted by |i-j|
        """
        n = cm.shape[0]
        error_weights = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
        return cm * error_weights
    
    @staticmethod
    def analyze_errors(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze prediction errors in detail.
        
        Args:
            y_true: True ordinal labels
            y_pred: Predicted ordinal labels
            labels: Label names for interpretation
            
        Returns:
            Dictionary with error analysis
        """
        if labels is None:
            labels = DEFAULT_LABELS
            
        errors = y_pred - y_true
        abs_errors = np.abs(errors)
        
        # Find worst errors
        worst_indices = np.argsort(abs_errors)[::-1][:10]
        
        worst_errors = []
        for idx in worst_indices:
            if abs_errors[idx] > 0:
                worst_errors.append({
                    'index': int(idx),
                    'true_class': labels[y_true[idx]],
                    'pred_class': labels[y_pred[idx]],
                    'error_distance': int(errors[idx]),
                })
        
        return {
            'total_errors': int(np.sum(abs_errors > 0)),
            'error_rate': float(np.mean(abs_errors > 0)),
            'off_by_1': int(np.sum(abs_errors == 1)),
            'off_by_2': int(np.sum(abs_errors == 2)),
            'off_by_3_plus': int(np.sum(abs_errors >= 3)),
            'mean_error_distance': float(np.mean(abs_errors)),
            'max_error_distance': int(np.max(abs_errors)),
            'worst_errors': worst_errors,
            'underestimate_count': int(np.sum(errors < 0)),
            'overestimate_count': int(np.sum(errors > 0)),
        }


# =============================================================================
# Calibration Analysis
# =============================================================================

class CalibrationAnalysis:
    """
    Calibration analysis for probabilistic predictions.
    Measures how well predicted probabilities match actual outcomes.
    """
    
    @staticmethod
    def expected_calibration_error(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Expected Calibration Error (ECE).
        Measures the difference between predicted confidence and actual accuracy.
        
        Args:
            y_true: True ordinal labels (0-indexed integers)
            y_proba: Predicted probabilities shape (n_samples, n_classes)
            n_bins: Number of bins for calibration
            
        Returns:
            ECE value (0 = perfectly calibrated)
        """
        confidences = np.max(y_proba, axis=1)
        predictions = np.argmax(y_proba, axis=1)
        accuracies = (predictions == y_true)
        
        ece = 0.0
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                avg_accuracy_in_bin = np.mean(accuracies[in_bin])
                ece += np.abs(avg_accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
        
        return ece
    
    @staticmethod
    def max_calibration_error(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Maximum Calibration Error (MCE).
        The maximum difference between confidence and accuracy in any bin.
        
        Args:
            y_true: True ordinal labels
            y_proba: Predicted probabilities
            n_bins: Number of bins
            
        Returns:
            MCE value (0 = perfectly calibrated)
        """
        confidences = np.max(y_proba, axis=1)
        predictions = np.argmax(y_proba, axis=1)
        accuracies = (predictions == y_true)
        
        mce = 0.0
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            
            if np.sum(in_bin) > 0:
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                avg_accuracy_in_bin = np.mean(accuracies[in_bin])
                mce = max(mce, np.abs(avg_accuracy_in_bin - avg_confidence_in_bin))
        
        return mce
    
    @staticmethod
    def reliability_diagram_data(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Compute data for reliability diagram (calibration curve).
        
        Args:
            y_true: True ordinal labels
            y_proba: Predicted probabilities
            n_bins: Number of bins
            
        Returns:
            Dictionary with bin data for plotting
        """
        confidences = np.max(y_proba, axis=1)
        predictions = np.argmax(y_proba, axis=1)
        accuracies = (predictions == y_true).astype(float)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            bin_counts.append(np.sum(in_bin))
            
            if np.sum(in_bin) > 0:
                bin_accuracies.append(np.mean(accuracies[in_bin]))
                bin_confidences.append(np.mean(confidences[in_bin]))
            else:
                bin_accuracies.append(np.nan)
                bin_confidences.append(np.nan)
        
        return {
            'bin_centers': bin_centers,
            'bin_accuracies': np.array(bin_accuracies),
            'bin_confidences': np.array(bin_confidences),
            'bin_counts': np.array(bin_counts),
        }
    
    @staticmethod
    def average_confidence(y_proba: np.ndarray) -> float:
        """Average prediction confidence."""
        return float(np.mean(np.max(y_proba, axis=1)))
    
    @staticmethod
    def confidence_histogram_data(
        y_proba: np.ndarray,
        n_bins: int = 20
    ) -> Dict[str, np.ndarray]:
        """
        Compute histogram of prediction confidences.
        
        Args:
            y_proba: Predicted probabilities
            n_bins: Number of histogram bins
            
        Returns:
            Dictionary with histogram data
        """
        confidences = np.max(y_proba, axis=1)
        counts, bin_edges = np.histogram(confidences, bins=n_bins, range=(0, 1))
        
        return {
            'counts': counts,
            'bin_edges': bin_edges,
            'mean_confidence': float(np.mean(confidences)),
            'std_confidence': float(np.std(confidences)),
        }


# =============================================================================
# Main Evaluation Function
# =============================================================================

def evaluate_model(
    y_true: List[str],
    y_pred: List[str],
    y_proba: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    include_calibration: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation combining all metrics.
    
    This is the main API for evaluating any model's predictions.
    
    Args:
        y_true: True labels (strings)
        y_pred: Predicted labels (strings)
        y_proba: Optional predicted probabilities shape (n_samples, n_classes)
        labels: Ordered list of label names (uses DEFAULT_LABELS if None)
        include_calibration: Whether to include calibration metrics (requires y_proba)
        
    Returns:
        Dictionary containing:
        - 'ordinal': Ordinal-aware metrics (MAE, QWK, etc.)
        - 'classification': Standard classification metrics
        - 'per_class': Per-class metrics breakdown
        - 'confusion': Confusion analysis
        - 'calibration': Calibration metrics (if y_proba provided)
        
    Example:
        >>> results = evaluate_model(y_true, y_pred, y_proba)
        >>> print(f"MAE: {results['ordinal']['mae']:.3f}")
        >>> print(f"QWK: {results['ordinal']['qwk']:.3f}")
        >>> print(f"F1 Macro: {results['classification']['f1_macro']:.3f}")
    """
    if labels is None:
        labels = DEFAULT_LABELS
    
    # Convert to ordinal for ordinal metrics
    label_ordering = get_label_ordering(labels)
    
    # Filter to labels that exist in data
    valid_mask = [label in label_ordering for label in y_true]
    if not all(valid_mask):
        logger.warning(f"Some labels not in ordering, filtering {sum(not v for v in valid_mask)} samples")
        y_true = [y for y, v in zip(y_true, valid_mask) if v]
        y_pred = [y for y, v in zip(y_pred, valid_mask) if v]
        if y_proba is not None:
            y_proba = y_proba[valid_mask]
    
    y_true_ord = labels_to_ordinal(y_true, label_ordering)
    y_pred_ord = labels_to_ordinal(y_pred, label_ordering)
    
    results = {}
    
    # Ordinal metrics
    kendall_tau, kendall_p = OrdinalMetrics.kendall_tau(y_true_ord, y_pred_ord)
    spearman_rho, spearman_p = OrdinalMetrics.spearman_rho(y_true_ord, y_pred_ord)
    
    results['ordinal'] = {
        'mae': OrdinalMetrics.mean_absolute_error(y_true_ord, y_pred_ord),
        'qwk': OrdinalMetrics.quadratic_weighted_kappa(y_true_ord, y_pred_ord),
        'lwk': OrdinalMetrics.linear_weighted_kappa(y_true_ord, y_pred_ord),
        'off_by_1_accuracy': OrdinalMetrics.off_by_one_accuracy(y_true_ord, y_pred_ord),
        'off_by_2_accuracy': OrdinalMetrics.off_by_k_accuracy(y_true_ord, y_pred_ord, k=2),
        'kendall_tau': kendall_tau,
        'kendall_p_value': kendall_p,
        'spearman_rho': spearman_rho,
        'spearman_p_value': spearman_p,
        'error_distribution': OrdinalMetrics.ordinal_error_distribution(y_true_ord, y_pred_ord),
    }
    
    # Classification metrics
    results['classification'] = ClassificationMetrics.compute_all(y_true, y_pred, labels)
    
    # Per-class metrics
    results['per_class'] = ClassificationMetrics.per_class_metrics(y_true, y_pred, labels)
    
    # Confusion analysis
    cm = ConfusionAnalysis.compute_confusion_matrix(y_true, y_pred, labels)
    cm_normalized = ConfusionAnalysis.confusion_matrix_normalized(y_true, y_pred, labels)
    
    results['confusion'] = {
        'matrix': cm.tolist(),
        'matrix_normalized': cm_normalized.tolist(),
        'error_analysis': ConfusionAnalysis.analyze_errors(y_true_ord, y_pred_ord, labels),
        'labels': labels,
    }
    
    # Calibration (if probabilities provided)
    if y_proba is not None and include_calibration:
        results['calibration'] = {
            'ece': CalibrationAnalysis.expected_calibration_error(y_true_ord, y_proba),
            'mce': CalibrationAnalysis.max_calibration_error(y_true_ord, y_proba),
            'average_confidence': CalibrationAnalysis.average_confidence(y_proba),
            'reliability_diagram': CalibrationAnalysis.reliability_diagram_data(y_true_ord, y_proba),
            'confidence_histogram': CalibrationAnalysis.confidence_histogram_data(y_proba),
        }
    
    # Summary statistics
    results['summary'] = {
        'n_samples': len(y_true),
        'n_classes': len(labels),
        'class_distribution': dict(Counter(y_true)),
    }
    
    return results


# =============================================================================
# Model Comparison
# =============================================================================

def compare_models(
    model_results: Dict[str, Dict[str, Any]],
    metrics_to_compare: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple models across key metrics.
    
    Args:
        model_results: Dictionary mapping model name to its evaluation results
        metrics_to_compare: List of metrics to include. If None, uses defaults.
        
    Returns:
        Dictionary with comparison table data
        
    Example:
        >>> comparison = compare_models({
        ...     'TF-IDF': tfidf_results,
        ...     'Naive Bayes': nb_results,
        ... })
        >>> print(comparison)
    """
    if metrics_to_compare is None:
        metrics_to_compare = [
            'accuracy', 'f1_macro', 'f1_weighted',
            'mae', 'qwk', 'off_by_1_accuracy'
        ]
    
    comparison = {}
    
    for model_name, results in model_results.items():
        comparison[model_name] = {}
        
        for metric in metrics_to_compare:
            # Check ordinal metrics
            if metric in results.get('ordinal', {}):
                comparison[model_name][metric] = results['ordinal'][metric]
            # Check classification metrics
            elif metric in results.get('classification', {}):
                comparison[model_name][metric] = results['classification'][metric]
            # Check calibration
            elif metric in results.get('calibration', {}):
                comparison[model_name][metric] = results['calibration'][metric]
    
    return comparison


def format_comparison_table(comparison: Dict[str, Dict[str, float]]) -> str:
    """
    Format comparison as a pretty-printed table.
    
    Args:
        comparison: Output from compare_models()
        
    Returns:
        Formatted string table
    """
    if not comparison:
        return "No models to compare"
    
    # Get all metrics
    all_metrics = set()
    for metrics in comparison.values():
        all_metrics.update(metrics.keys())
    all_metrics = sorted(all_metrics)
    
    # Build header
    model_names = list(comparison.keys())
    header = f"{'Metric':<25}" + "".join(f"{name:>15}" for name in model_names)
    separator = "-" * len(header)
    
    lines = [separator, header, separator]
    
    for metric in all_metrics:
        row = f"{metric:<25}"
        for model_name in model_names:
            value = comparison[model_name].get(metric, float('nan'))
            if isinstance(value, float):
                row += f"{value:>15.4f}"
            else:
                row += f"{str(value):>15}"
        lines.append(row)
    
    lines.append(separator)
    return "\n".join(lines)


# =============================================================================
# Report Generation
# =============================================================================

def generate_report(
    results: Dict[str, Any],
    model_name: str = "Model",
    output_path: Optional[Path] = None
) -> str:
    """
    Generate a comprehensive evaluation report.
    
    Args:
        results: Output from evaluate_model()
        model_name: Name of the model being evaluated
        output_path: Optional path to save the report
        
    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"EVALUATION REPORT: {model_name}")
    lines.append("=" * 80)
    lines.append("")
    
    # Summary
    summary = results.get('summary', {})
    lines.append(f"Samples: {summary.get('n_samples', 'N/A')}")
    lines.append(f"Classes: {summary.get('n_classes', 'N/A')}")
    lines.append("")
    
    # Ordinal metrics (most important for this task)
    lines.append("-" * 40)
    lines.append("ORDINAL METRICS (Task-Specific)")
    lines.append("-" * 40)
    ordinal = results.get('ordinal', {})
    lines.append(f"  Mean Absolute Error:     {ordinal.get('mae', 0):.4f}")
    lines.append(f"  Quadratic Weighted Kappa:{ordinal.get('qwk', 0):.4f}")
    lines.append(f"  Linear Weighted Kappa:   {ordinal.get('lwk', 0):.4f}")
    lines.append(f"  Off-by-1 Accuracy:       {ordinal.get('off_by_1_accuracy', 0):.4f}")
    lines.append(f"  Off-by-2 Accuracy:       {ordinal.get('off_by_2_accuracy', 0):.4f}")
    lines.append(f"  Kendall's Tau:           {ordinal.get('kendall_tau', 0):.4f}")
    lines.append(f"  Spearman's Rho:          {ordinal.get('spearman_rho', 0):.4f}")
    lines.append("")
    
    # Classification metrics
    lines.append("-" * 40)
    lines.append("CLASSIFICATION METRICS")
    lines.append("-" * 40)
    classification = results.get('classification', {})
    lines.append(f"  Accuracy:                {classification.get('accuracy', 0):.4f}")
    lines.append(f"  F1 (macro):              {classification.get('f1_macro', 0):.4f}")
    lines.append(f"  F1 (weighted):           {classification.get('f1_weighted', 0):.4f}")
    lines.append(f"  Precision (macro):       {classification.get('precision_macro', 0):.4f}")
    lines.append(f"  Recall (macro):          {classification.get('recall_macro', 0):.4f}")
    lines.append("")
    
    # Per-class breakdown
    lines.append("-" * 40)
    lines.append("PER-CLASS METRICS")
    lines.append("-" * 40)
    per_class = results.get('per_class', {})
    lines.append(f"  {'Class':<35} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Supp':>8}")
    lines.append(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for label, metrics in per_class.items():
        lines.append(
            f"  {label:<35} "
            f"{metrics.get('precision', 0):>8.4f} "
            f"{metrics.get('recall', 0):>8.4f} "
            f"{metrics.get('f1', 0):>8.4f} "
            f"{metrics.get('support', 0):>8.0f}"
        )
    lines.append("")
    
    # Error analysis
    lines.append("-" * 40)
    lines.append("ERROR ANALYSIS")
    lines.append("-" * 40)
    error_analysis = results.get('confusion', {}).get('error_analysis', {})
    lines.append(f"  Total errors:            {error_analysis.get('total_errors', 0)}")
    lines.append(f"  Error rate:              {error_analysis.get('error_rate', 0):.4f}")
    lines.append(f"  Off by 1:                {error_analysis.get('off_by_1', 0)}")
    lines.append(f"  Off by 2:                {error_analysis.get('off_by_2', 0)}")
    lines.append(f"  Off by 3+:               {error_analysis.get('off_by_3_plus', 0)}")
    lines.append(f"  Underestimates:          {error_analysis.get('underestimate_count', 0)}")
    lines.append(f"  Overestimates:           {error_analysis.get('overestimate_count', 0)}")
    lines.append("")
    
    # Calibration (if available)
    if 'calibration' in results:
        lines.append("-" * 40)
        lines.append("CALIBRATION METRICS")
        lines.append("-" * 40)
        calibration = results['calibration']
        lines.append(f"  Expected Calib. Error:   {calibration.get('ece', 0):.4f}")
        lines.append(f"  Max Calibration Error:   {calibration.get('mce', 0):.4f}")
        lines.append(f"  Avg Confidence:          {calibration.get('average_confidence', 0):.4f}")
        lines.append("")
    
    lines.append("=" * 80)
    
    report = "\n".join(lines)
    
    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Report saved to {output_path}")
    
    return report


def save_results_json(
    results: Dict[str, Any],
    output_path: Path,
    model_name: Optional[str] = None
) -> None:
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Evaluation results dictionary
        output_path: Path to save JSON
        model_name: Optional model name to include in results
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare for JSON serialization
    json_results = results.copy()
    
    # Convert numpy arrays and types to Python native types
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            # Convert both keys and values
            return {
                (int(k) if isinstance(k, np.integer) else str(k) if not isinstance(k, (str, int, float, bool, type(None))) else k): convert_numpy(v)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj
    
    json_results = convert_numpy(json_results)
    
    if model_name:
        json_results['model_name'] = model_name
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {output_path}")


# =============================================================================
# Entry point for testing
# =============================================================================

if __name__ == "__main__":
    # Quick sanity check
    logger.info("Running evaluation module sanity check...")
    
    # Test with dummy perfect predictions
    y_true = DEFAULT_LABELS * 10
    y_pred = y_true.copy()  # Perfect predictions
    
    results = evaluate_model(y_true, y_pred)
    
    assert results['ordinal']['mae'] == 0.0, "MAE should be 0 for perfect predictions"
    assert results['ordinal']['qwk'] == 1.0, "QWK should be 1 for perfect predictions"
    assert results['classification']['accuracy'] == 1.0, "Accuracy should be 1 for perfect predictions"
    
    logger.info("✓ All sanity checks passed!")
    logger.info("")
    
    # Print sample report
    print(generate_report(results, model_name="Perfect Predictor"))
