# Baseline model training script
# This script trains simple baseline models for readability classification

import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    classification_report, 
    confusion_matrix
)
from sklearn.dummy import DummyClassifier
import pickle
from utils import setup_logger

logger = setup_logger()

# Constants
DATA_DIR = Path("./data/processed")
MODEL_DIR = Path("./models/baseline")
RESULTS_DIR = Path("./results/baseline")


def load_data():
    """
    Load train, validation, and test datasets.
    
    Returns:
        Dictionary with train, val, and test data
    """
    logger.info("Loading datasets...")
    
    datasets = {}
    for split in ['train', 'val', 'test']:
        file_path = DATA_DIR / f"{split}.json"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract texts and labels
        texts = [doc['content'] for doc in data]
        labels = [doc['label'] for doc in data]
        
        datasets[split] = {
            'texts': texts,
            'labels': labels,
            'data': data
        }
        
        logger.info(f"  Loaded {split}: {len(texts)} documents")
    
    return datasets


def extract_simple_features(texts):
    """
    Extract simple statistical features from texts.
    
    Args:
        texts: List of text strings
        
    Returns:
        numpy array of features
    """
    features = []
    
    for text in texts:
        words = text.split()
        sentences = text.count('.') + text.count('!') + text.count('?')
        sentences = max(sentences, 1)  # Avoid division by zero
        
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


def train_dummy_baseline(X_train, y_train, X_val, y_val):
    """
    Train a dummy classifier (most frequent class baseline).
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        
    Returns:
        Trained model and metrics
    """
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING: Dummy Classifier (Most Frequent)")
    logger.info("=" * 80)
    
    model = DummyClassifier(strategy='most_frequent', random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    # Metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    val_acc = accuracy_score(y_val, y_pred_val)
    val_f1_macro = f1_score(y_val, y_pred_val, average='macro')
    val_f1_weighted = f1_score(y_val, y_pred_val, average='weighted')
    
    logger.info(f"  Train Accuracy: {train_acc:.4f}")
    logger.info(f"  Val Accuracy: {val_acc:.4f}")
    logger.info(f"  Val F1 (macro): {val_f1_macro:.4f}")
    logger.info(f"  Val F1 (weighted): {val_f1_weighted:.4f}")
    
    return {
        'model': model,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'val_f1_macro': val_f1_macro,
        'val_f1_weighted': val_f1_weighted,
        'predictions': y_pred_val
    }


def train_simple_features_baseline(datasets):
    """
    Train baseline with simple statistical features.
    
    Args:
        datasets: Dictionary with train, val, test data
        
    Returns:
        Dictionary with model and results
    """
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING: Logistic Regression with Simple Features")
    logger.info("=" * 80)
    
    # Extract features
    logger.info("  Extracting simple statistical features...")
    X_train = extract_simple_features(datasets['train']['texts'])
    X_val = extract_simple_features(datasets['val']['texts'])
    X_test = extract_simple_features(datasets['test']['texts'])
    
    y_train = datasets['train']['labels']
    y_val = datasets['val']['labels']
    y_test = datasets['test']['labels']
    
    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    # Metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    val_acc = accuracy_score(y_val, y_pred_val)
    val_f1_macro = f1_score(y_val, y_pred_val, average='macro')
    val_f1_weighted = f1_score(y_val, y_pred_val, average='weighted')
    
    logger.info(f"  Train Accuracy: {train_acc:.4f}")
    logger.info(f"  Val Accuracy: {val_acc:.4f}")
    logger.info(f"  Val F1 (macro): {val_f1_macro:.4f}")
    logger.info(f"  Val F1 (weighted): {val_f1_weighted:.4f}")
    
    return {
        'model': model,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'val_f1_macro': val_f1_macro,
        'val_f1_weighted': val_f1_weighted,
        'predictions': y_pred_val,
        'X_test': X_test,
        'y_test': y_test
    }


def train_tfidf_baseline(datasets, max_features=1000):
    """
    Train baseline with TF-IDF features.
    
    Args:
        datasets: Dictionary with train, val, test data
        max_features: Maximum number of TF-IDF features
        
    Returns:
        Dictionary with model and results
    """
    logger.info("\n" + "=" * 80)
    logger.info(f"TRAINING: Logistic Regression with TF-IDF (max_features={max_features})")
    logger.info("=" * 80)
    
    # Extract TF-IDF features
    logger.info("  Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    X_train = vectorizer.fit_transform(datasets['train']['texts'])
    X_val = vectorizer.transform(datasets['val']['texts'])
    X_test = vectorizer.transform(datasets['test']['texts'])
    
    y_train = datasets['train']['labels']
    y_val = datasets['val']['labels']
    y_test = datasets['test']['labels']
    
    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    # Metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    val_acc = accuracy_score(y_val, y_pred_val)
    val_f1_macro = f1_score(y_val, y_pred_val, average='macro')
    val_f1_weighted = f1_score(y_val, y_pred_val, average='weighted')
    
    logger.info(f"  Train Accuracy: {train_acc:.4f}")
    logger.info(f"  Val Accuracy: {val_acc:.4f}")
    logger.info(f"  Val F1 (macro): {val_f1_macro:.4f}")
    logger.info(f"  Val F1 (weighted): {val_f1_weighted:.4f}")
    
    return {
        'model': model,
        'vectorizer': vectorizer,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'val_f1_macro': val_f1_macro,
        'val_f1_weighted': val_f1_weighted,
        'predictions': y_pred_val,
        'X_test': X_test,
        'y_test': y_test
    }


def train_naive_bayes_baseline(datasets):
    """
    Train Naive Bayes baseline with count features.
    
    Args:
        datasets: Dictionary with train, val, test data
        
    Returns:
        Dictionary with model and results
    """
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING: Naive Bayes with Count Features")
    logger.info("=" * 80)
    
    # Extract count features
    logger.info("  Extracting count features...")
    vectorizer = CountVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    X_train = vectorizer.fit_transform(datasets['train']['texts'])
    X_val = vectorizer.transform(datasets['val']['texts'])
    X_test = vectorizer.transform(datasets['test']['texts'])
    
    y_train = datasets['train']['labels']
    y_val = datasets['val']['labels']
    y_test = datasets['test']['labels']
    
    # Train model
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    
    # Metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    val_acc = accuracy_score(y_val, y_pred_val)
    val_f1_macro = f1_score(y_val, y_pred_val, average='macro')
    val_f1_weighted = f1_score(y_val, y_pred_val, average='weighted')
    
    logger.info(f"  Train Accuracy: {train_acc:.4f}")
    logger.info(f"  Val Accuracy: {val_acc:.4f}")
    logger.info(f"  Val F1 (macro): {val_f1_macro:.4f}")
    logger.info(f"  Val F1 (weighted): {val_f1_weighted:.4f}")
    
    return {
        'model': model,
        'vectorizer': vectorizer,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'val_f1_macro': val_f1_macro,
        'val_f1_weighted': val_f1_weighted,
        'predictions': y_pred_val,
        'X_test': X_test,
        'y_test': y_test
    }


def evaluate_on_test(model_name, model_dict, y_test=None):
    """
    Evaluate model on test set.
    
    Args:
        model_name: Name of the model
        model_dict: Dictionary containing model and test data
        y_test: Optional test labels (if not in model_dict)
    """
    logger.info(f"\n  Evaluating {model_name} on test set...")
    
    model = model_dict['model']
    X_test = model_dict.get('X_test')
    if y_test is None:
        y_test = model_dict.get('y_test')
    
    if X_test is None:
        logger.warning(f"  No test data available for {model_name}")
        return None
    
    y_pred_test = model.predict(X_test)
    
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1_macro = f1_score(y_test, y_pred_test, average='macro')
    test_f1_weighted = f1_score(y_test, y_pred_test, average='weighted')
    
    logger.info(f"    Test Accuracy: {test_acc:.4f}")
    logger.info(f"    Test F1 (macro): {test_f1_macro:.4f}")
    logger.info(f"    Test F1 (weighted): {test_f1_weighted:.4f}")
    
    return {
        'test_acc': test_acc,
        'test_f1_macro': test_f1_macro,
        'test_f1_weighted': test_f1_weighted,
        'predictions': y_pred_test,
        'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist()
    }


def save_results(results, output_dir=RESULTS_DIR):
    """
    Save baseline results to JSON.
    
    Args:
        results: Dictionary with all baseline results
        output_dir: Directory to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare results for JSON (remove non-serializable objects)
    json_results = {}
    for model_name, model_dict in results.items():
        json_results[model_name] = {
            'train_acc': model_dict.get('train_acc'),
            'val_acc': model_dict.get('val_acc'),
            'val_f1_macro': model_dict.get('val_f1_macro'),
            'val_f1_weighted': model_dict.get('val_f1_weighted'),
            'test_acc': model_dict.get('test_acc'),
            'test_f1_macro': model_dict.get('test_f1_macro'),
            'test_f1_weighted': model_dict.get('test_f1_weighted'),
        }
    
    output_file = output_dir / "baseline_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"\n  Saved results to {output_file}")


def save_best_model(model_name, model_dict, output_dir=MODEL_DIR):
    """
    Save the best baseline model.
    
    Args:
        model_name: Name of the model
        model_dict: Dictionary containing model
        output_dir: Directory to save model
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_file = output_dir / f"{model_name.replace(' ', '_').lower()}.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(model_dict['model'], f)
    
    # Save vectorizer if present
    if 'vectorizer' in model_dict:
        vectorizer_file = output_dir / f"{model_name.replace(' ', '_').lower()}_vectorizer.pkl"
        with open(vectorizer_file, 'wb') as f:
            pickle.dump(model_dict['vectorizer'], f)
    
    logger.info(f"  Saved {model_name} to {model_file}")


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("BASELINE MODEL TRAINING")
    logger.info("=" * 80)
    
    # Load data
    datasets = load_data()
    
    # Get dummy baseline (for reference)
    y_train = datasets['train']['labels']
    y_val = datasets['val']['labels']
    dummy_result = train_dummy_baseline(
        np.zeros((len(y_train), 1)), y_train,
        np.zeros((len(y_val), 1)), y_val
    )
    
    # Train baselines
    results = {
        'Dummy (Most Frequent)': dummy_result,
        'Simple Features': train_simple_features_baseline(datasets),
        'TF-IDF': train_tfidf_baseline(datasets, max_features=1000),
        'Naive Bayes': train_naive_bayes_baseline(datasets),
    }
    
    # Evaluate on test set
    logger.info("\n" + "=" * 80)
    logger.info("TEST SET EVALUATION")
    logger.info("=" * 80)
    
    y_test = datasets['test']['labels']
    for model_name, model_dict in results.items():
        if model_name != 'Dummy (Most Frequent)':
            test_results = evaluate_on_test(model_name, model_dict, y_test)
            if test_results:
                results[model_name].update(test_results)
    
    # Find best model
    logger.info("\n" + "=" * 80)
    logger.info("BASELINE COMPARISON")
    logger.info("=" * 80)
    
    logger.info(f"\n  {'Model':<25} {'Val Acc':>10} {'Val F1':>10} {'Test Acc':>10} {'Test F1':>10}")
    logger.info(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    best_model_name = None
    best_val_f1 = 0
    
    for model_name, model_dict in results.items():
        val_acc = model_dict.get('val_acc', 0)
        val_f1 = model_dict.get('val_f1_weighted', 0)
        test_acc = model_dict.get('test_acc', 0)
        test_f1 = model_dict.get('test_f1_weighted', 0)
        
        logger.info(f"  {model_name:<25} {val_acc:>10.4f} {val_f1:>10.4f} {test_acc:>10.4f} {test_f1:>10.4f}")
        
        if val_f1 > best_val_f1 and model_name != 'Dummy (Most Frequent)':
            best_val_f1 = val_f1
            best_model_name = model_name
    
    logger.info(f"\n  ✓ Best baseline: {best_model_name} (Val F1: {best_val_f1:.4f})")
    
    # Save results
    save_results(results)
    
    # Save best model
    if best_model_name:
        save_best_model(best_model_name, results[best_model_name])
    
    logger.info("\n" + "=" * 80)
    logger.info("BASELINE TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info("  Next steps:")
    logger.info("  1. Analyze baseline performance")
    logger.info("  2. Implement feature engineering")
    logger.info("  3. Train advanced models to beat the baseline")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
