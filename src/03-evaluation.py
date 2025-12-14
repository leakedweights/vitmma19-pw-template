# Model evaluation script
# This script evaluates trained models on the test set and generates comprehensive metrics.

import json
import pickle
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from utils import setup_logger
from evaluation import (
    evaluate_model,
    compare_models,
    format_comparison_table,
    generate_report,
    save_results_json,
    DEFAULT_LABELS,
)

logger = setup_logger(__name__)

# Constants - resolve paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models" / "baseline"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_test_data() -> Dict[str, Any]:
    """
    Load test dataset.
    
    Returns:
        Dictionary with texts, labels, and raw data
    """
    logger.info("Loading test data...")
    
    test_file = DATA_DIR / "test.json"
    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = [doc['content'] for doc in data]
    labels = [doc['label'] for doc in data]
    
    logger.info(f"  Loaded {len(texts)} test samples")
    
    return {'texts': texts, 'labels': labels, 'data': data}


def load_model(model_path: Path) -> Any:
    """Load a pickled model."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def evaluate_baseline_models() -> Dict[str, Dict[str, Any]]:
    """
    Evaluate all baseline models on the test set.
    
    Returns:
        Dictionary mapping model name to evaluation results
    """
    logger.info("=" * 80)
    logger.info("BASELINE MODEL EVALUATION")
    logger.info("=" * 80)
    
    # Load test data
    test_data = load_test_data()
    y_true = test_data['labels']
    texts = test_data['texts']
    
    all_results = {}
    
    # Find all baseline models
    if not MODEL_DIR.exists():
        logger.warning(f"Model directory not found: {MODEL_DIR}")
        return all_results
    
    model_files = list(MODEL_DIR.glob("*.pkl"))
    
    # Group models with their vectorizers
    model_names = set()
    for f in model_files:
        name = f.stem.replace("_vectorizer", "")
        model_names.add(name)
    
    for model_name in sorted(model_names):
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {model_name}")
        logger.info(f"{'='*60}")
        
        model_path = MODEL_DIR / f"{model_name}.pkl"
        vectorizer_path = MODEL_DIR / f"{model_name}_vectorizer.pkl"
        
        if not model_path.exists():
            logger.warning(f"  Model file not found: {model_path}")
            continue
        
        try:
            # Load model
            model = load_model(model_path)
            
            # Prepare features based on model type
            if vectorizer_path.exists():
                # Text-based model with vectorizer
                vectorizer = load_model(vectorizer_path)
                X_test = vectorizer.transform(texts)
            else:
                # Simple features model - recreate features
                X_test = extract_simple_features(texts)
            
            # Get predictions
            y_pred_list = model.predict(X_test).tolist()
            
            # Get probabilities if available
            y_proba = None
            if hasattr(model, 'predict_proba'):
                try:
                    y_proba = model.predict_proba(X_test)
                except Exception:
                    pass
            
            # Run comprehensive evaluation
            results = evaluate_model(
                y_true=y_true,
                y_pred=y_pred_list,
                y_proba=y_proba,
                labels=DEFAULT_LABELS,
                include_calibration=y_proba is not None
            )
            
            # Print report
            report = generate_report(results, model_name=model_name)
            logger.info("\n" + report)
            
            # Save detailed results
            output_dir = RESULTS_DIR / "evaluation"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            save_results_json(
                results,
                output_dir / f"{model_name}_evaluation.json",
                model_name=model_name
            )
            
            all_results[model_name] = results
            
        except Exception as e:
            logger.error(f"  Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    return all_results


def extract_simple_features(texts):
    """
    Extract simple statistical features from texts.
    Same as in baseline training for consistency.
    """
    features = []
    
    for text in texts:
        words = text.split()
        sentences = text.count('.') + text.count('!') + text.count('?')
        sentences = max(sentences, 1)
        
        feature_dict = {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': sentences,
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'avg_sentence_length': len(words) / sentences,
            'unique_word_ratio': len(set(words)) / len(words) if words else 0,
        }
        
        features.append(list(feature_dict.values()))
    
    return np.array(features)


def print_comparison_summary(all_results: Dict[str, Dict[str, Any]]) -> None:
    """Print a comparison summary of all models."""
    if not all_results:
        logger.info("No results to compare")
        return
    
    logger.info("\n" + "=" * 80)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("=" * 80)
    
    comparison = compare_models(all_results)
    table = format_comparison_table(comparison)
    logger.info("\n" + table)
    
    # Find best model by different criteria
    best_qwk = max(all_results.items(), key=lambda x: x[1].get('ordinal', {}).get('qwk', 0))
    best_mae = min(all_results.items(), key=lambda x: x[1].get('ordinal', {}).get('mae', float('inf')))
    best_f1 = max(all_results.items(), key=lambda x: x[1].get('classification', {}).get('f1_macro', 0))
    
    logger.info("\n  Best Models:")
    logger.info(f"    By QWK:        {best_qwk[0]} ({best_qwk[1]['ordinal']['qwk']:.4f})")
    logger.info(f"    By MAE:        {best_mae[0]} ({best_mae[1]['ordinal']['mae']:.4f})")
    logger.info(f"    By F1 (macro): {best_f1[0]} ({best_f1[1]['classification']['f1_macro']:.4f})")


def evaluate():
    """Main evaluation function."""
    logger.info("Starting comprehensive model evaluation...")
    
    try:
        # Evaluate all baseline models
        all_results = evaluate_baseline_models()
        
        # Print comparison
        print_comparison_summary(all_results)
        
        # Save combined results
        if all_results:
            output_dir = RESULTS_DIR / "evaluation"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save comparison summary
            comparison = compare_models(all_results)
            with open(output_dir / "comparison_summary.json", 'w') as f:
                json.dump(comparison, f, indent=2)
            
            logger.info(f"\n  Results saved to {output_dir}")
        
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    evaluate()
