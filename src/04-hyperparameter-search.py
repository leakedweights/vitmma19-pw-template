# Hyperparameter Tuning Script
# Grid search over key hyperparameters for TextCNN with CORAL loss

import json
import itertools
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

import sys
sys.path.insert(0, str(Path(__file__).parent))

from utils import setup_logger

logger = setup_logger(__name__)

# Import after path setup
from dataset import Vocabulary, LabelEncoder, TextClassificationDataset, collate_fn, load_split
from features import FeatureExtractor
from models import create_model, CORALLoss
from evaluation import evaluate_model

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "hyperparameter_search"


# =============================================================================
# Hyperparameter Grid
# =============================================================================

PARAM_GRID = {
    'embedding_dim': [32, 64, 128],
    'num_filters': [16, 32, 64],
    'dropout': [0.2, 0.3, 0.5],
    'learning_rate': [1e-3, 5e-4],
}

# Fixed parameters
FIXED_PARAMS = {
    'model_type': 'textcnn',
    'use_coral': True,
    'use_features': True,
    'filter_sizes': (2, 3, 4, 5),
    'batch_size': 32,
    'epochs': 30,
    'patience': 8,
    'vocab_min_freq': 2,
    'vocab_max_size': 8000,
    'max_seq_length': 256,
}


# =============================================================================
# Quick Training Function
# =============================================================================

def train_with_config(
    config: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    vocab_size: int,
    feature_dim: int,
    label_encoder: LabelEncoder,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """Train a model with given configuration and return validation metrics."""
    
    # Create model
    model = create_model(
        model_type=config['model_type'],
        vocab_size=vocab_size,
        num_classes=5,
        feature_dim=feature_dim,
        use_coral=config['use_coral'],
        embedding_dim=config['embedding_dim'],
        num_filters=config['num_filters'],
        filter_sizes=config['filter_sizes'],
        dropout=config['dropout'],
    )
    model = model.to(device)
    
    # Criterion
    criterion = CORALLoss(num_classes=5)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=1e-4
    )
    
    # Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    best_val_qwk = -float('inf')
    patience_counter = 0
    
    for epoch in range(1, config['epochs'] + 1):
        # Train
        model.train()
        for batch in train_loader:
            token_ids = batch['token_ids'].to(device)
            labels = batch['labels'].to(device)
            lengths = batch['lengths']
            features = batch.get('features')
            if features is not None:
                features = features.to(device)
            
            optimizer.zero_grad()
            logits = model(token_ids, lengths, features)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Validate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                token_ids = batch['token_ids'].to(device)
                labels = batch['labels'].to(device)
                lengths = batch['lengths']
                features = batch.get('features')
                if features is not None:
                    features = features.to(device)
                
                logits = model(token_ids, lengths, features)
                preds = model.predict(logits)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        
        # Evaluate
        pred_labels = [label_encoder.decode(p) for p in all_preds]
        true_labels = [label_encoder.decode(l) for l in all_labels]
        results = evaluate_model(true_labels, pred_labels, include_calibration=False)
        
        val_qwk = results['ordinal']['qwk']
        scheduler.step(val_qwk)
        
        # Early stopping
        if val_qwk > best_val_qwk + 0.001:
            best_val_qwk = val_qwk
            best_results = results
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            break
    
    return {
        'best_val_qwk': best_val_qwk,
        'best_val_mae': best_results['ordinal']['mae'],
        'best_val_acc': best_results['classification']['accuracy'],
        'off_by_1': best_results['ordinal']['off_by_1_accuracy'],
        'epochs_trained': epoch,
        'parameters': model.count_parameters(),
    }


# =============================================================================
# Main Search
# =============================================================================

def run_hyperparameter_search(max_trials: int = None):
    """Run grid search over hyperparameters."""
    
    logger.info("=" * 80)
    logger.info("HYPERPARAMETER SEARCH")
    logger.info("=" * 80)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load data once
    logger.info("\n1. Loading data...")
    train_texts, train_labels = load_split('train', DATA_DIR)
    val_texts, val_labels = load_split('val', DATA_DIR)
    
    # Extract features once
    logger.info("\n2. Extracting features...")
    extractor = FeatureExtractor()
    train_features = extractor.extract_batch(train_texts)
    val_features = extractor.extract_batch(val_texts)
    feature_dim = train_features.shape[1]
    
    # Normalize
    train_mean = train_features.mean(axis=0)
    train_std = train_features.std(axis=0) + 1e-8
    train_features = (train_features - train_mean) / train_std
    val_features = (val_features - train_mean) / train_std
    
    # Build vocabulary once
    logger.info("\n3. Building vocabulary...")
    vocab = Vocabulary(min_freq=FIXED_PARAMS['vocab_min_freq'], max_size=FIXED_PARAMS['vocab_max_size'])
    vocab.build(train_texts)
    
    label_encoder = LabelEncoder()
    
    # Create datasets
    train_dataset = TextClassificationDataset(
        train_texts, train_labels, vocab, label_encoder,
        max_length=FIXED_PARAMS['max_seq_length'],
        features=train_features
    )
    val_dataset = TextClassificationDataset(
        val_texts, val_labels, vocab, label_encoder,
        max_length=FIXED_PARAMS['max_seq_length'],
        features=val_features
    )
    
    train_loader = DataLoader(train_dataset, batch_size=FIXED_PARAMS['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=FIXED_PARAMS['batch_size'], shuffle=False, collate_fn=collate_fn)
    
    # Generate all configurations
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    all_configs = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    if max_trials and max_trials < len(all_configs):
        # Random sample
        np.random.shuffle(all_configs)
        all_configs = all_configs[:max_trials]
    
    logger.info(f"\n4. Running {len(all_configs)} configurations...")
    
    results = []
    
    for i, params in enumerate(all_configs, 1):
        config = {**FIXED_PARAMS, **params}
        
        logger.info(f"\n[{i}/{len(all_configs)}] Testing: emb={params['embedding_dim']}, filters={params['num_filters']}, dropout={params['dropout']}, lr={params['learning_rate']}")
        
        try:
            metrics = train_with_config(
                config, train_loader, val_loader,
                len(vocab), feature_dim, label_encoder, device
            )
            
            result = {**params, **metrics}
            results.append(result)
            
            logger.info(f"  → QWK: {metrics['best_val_qwk']:.4f}, MAE: {metrics['best_val_mae']:.4f}, Off-by-1: {metrics['off_by_1']:.4f}, Params: {metrics['parameters']:,}")
            
        except Exception as e:
            logger.error(f"  → Error: {e}")
            continue
    
    # Sort by QWK
    results.sort(key=lambda x: x['best_val_qwk'], reverse=True)
    
    # Print top 5
    logger.info("\n" + "=" * 80)
    logger.info("TOP 5 CONFIGURATIONS")
    logger.info("=" * 80)
    
    for i, r in enumerate(results[:5], 1):
        logger.info(
            f"{i}. QWK={r['best_val_qwk']:.4f} | MAE={r['best_val_mae']:.4f} | "
            f"emb={r['embedding_dim']}, filters={r['num_filters']}, "
            f"dropout={r['dropout']}, lr={r['learning_rate']}"
        )
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = RESULTS_DIR / f'search_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_file}")
    
    # Return best config
    if results:
        best = results[0]
        logger.info(f"\n✓ Best configuration: {best}")
        return best
    return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-trials', type=int, default=None, help='Limit number of trials')
    args = parser.parse_args()
    
    run_hyperparameter_search(max_trials=args.max_trials)
