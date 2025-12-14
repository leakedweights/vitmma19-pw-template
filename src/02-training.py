# Deep Learning Training Script
# Trains models for legal text understandability classification.

import json
import time
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import setup_logger
from dataset import (
    Vocabulary, LabelEncoder, TextClassificationDataset,
    collate_fn, load_split, create_data_loaders
)
from features import FeatureExtractor, extract_features_from_dataset
from models import create_model, CORALLoss, OrdinalCrossEntropyLoss, coral_predict
from evaluation import evaluate_model, generate_report, save_results_json

logger = setup_logger(__name__)

# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


# =============================================================================
# Training Configuration
# =============================================================================

class TrainingConfig:
    """Training configuration with sensible defaults."""
    
    def __init__(self, **kwargs):
        # Model
        self.model_type = kwargs.get('model_type', 'textcnn')
        self.use_features = kwargs.get('use_features', True)
        self.use_coral = kwargs.get('use_coral', False)
        
        # Architecture
        self.embedding_dim = kwargs.get('embedding_dim', 64)
        self.hidden_dim = kwargs.get('hidden_dim', 64)
        self.num_filters = kwargs.get('num_filters', 32)
        self.filter_sizes = kwargs.get('filter_sizes', (2, 3, 4, 5))
        self.dropout = kwargs.get('dropout', 0.3)
        
        # Vocabulary
        self.vocab_min_freq = kwargs.get('vocab_min_freq', 2)
        self.vocab_max_size = kwargs.get('vocab_max_size', 8000)
        self.max_seq_length = kwargs.get('max_seq_length', 256)
        
        # Training
        self.batch_size = kwargs.get('batch_size', 32)
        self.learning_rate = kwargs.get('learning_rate', 1e-3)
        self.weight_decay = kwargs.get('weight_decay', 1e-4)
        self.epochs = kwargs.get('epochs', 50)
        self.patience = kwargs.get('patience', 10)
        self.min_delta = kwargs.get('min_delta', 0.001)
        
        # Device
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Experiment name
        self.experiment_name = kwargs.get('experiment_name', None)
        if self.experiment_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.experiment_name = f"{self.model_type}_{timestamp}"
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def __repr__(self):
        return f"TrainingConfig({self.to_dict()})"


# =============================================================================
# Training Loop
# =============================================================================

class Trainer:
    """Handles model training with early stopping and metrics tracking."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_mae': [],
            'val_qwk': [],
            'learning_rates': [],
        }
        self.best_val_loss = float('inf')
        self.best_val_qwk = -float('inf')
        self.patience_counter = 0
        self.best_model_state = None
    
    def train_epoch(
        self,
        model: nn.Module,
        train_loader,
        optimizer,
        criterion,
        label_encoder: LabelEncoder
    ) -> float:
        """Train for one epoch."""
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            token_ids = batch['token_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            lengths = batch['lengths']
            
            features = None
            if 'features' in batch:
                features = batch['features'].to(self.device)
            
            optimizer.zero_grad()
            
            logits = model(token_ids, lengths, features)
            loss = criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def evaluate(
        self,
        model: nn.Module,
        val_loader,
        criterion,
        label_encoder: LabelEncoder
    ) -> dict:
        """Evaluate model on validation set."""
        model.eval()
        total_loss = 0
        num_batches = 0
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in val_loader:
                token_ids = batch['token_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                lengths = batch['lengths']
                
                features = None
                if 'features' in batch:
                    features = batch['features'].to(self.device)
                
                logits = model(token_ids, lengths, features)
                loss = criterion(logits, labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Get predictions
                preds = model.predict(logits)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                
                # Get probabilities
                if self.config.use_coral:
                    probs = torch.sigmoid(logits)
                    # Convert CORAL probs to class probs (approximate)
                    probs_full = torch.zeros(logits.size(0), 5, device=logits.device)
                    for i in range(5):
                        if i == 0:
                            probs_full[:, i] = 1 - probs[:, 0]
                        elif i == 4:
                            probs_full[:, i] = probs[:, 3]
                        else:
                            probs_full[:, i] = probs[:, i-1] - probs[:, i]
                    all_probs.extend(probs_full.cpu().tolist())
                else:
                    probs = torch.softmax(logits, dim=-1)
                    all_probs.extend(probs.cpu().tolist())
        
        # Convert to label strings for evaluation
        pred_labels = [label_encoder.decode(p) for p in all_preds]
        true_labels = [label_encoder.decode(l) for l in all_labels]
        
        # Get metrics
        eval_results = evaluate_model(
            true_labels, pred_labels, 
            np.array(all_probs) if all_probs else None,
            include_calibration=True
        )
        
        return {
            'loss': total_loss / num_batches,
            'accuracy': eval_results['classification']['accuracy'],
            'mae': eval_results['ordinal']['mae'],
            'qwk': eval_results['ordinal']['qwk'],
            'f1_macro': eval_results['classification']['f1_macro'],
            'full_results': eval_results
        }
    
    def train(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        label_encoder: LabelEncoder
    ) -> dict:
        """Full training loop with early stopping."""
        model = model.to(self.device)
        
        # Criterion
        if self.config.use_coral:
            criterion = CORALLoss(num_classes=5)
        else:
            criterion = OrdinalCrossEntropyLoss(num_classes=5)
        
        # Optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Scheduler
        scheduler = ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Training: {self.config.experiment_name}")
        logger.info(f"{'='*60}")
        logger.info(f"Model parameters: {model.count_parameters():,}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
        logger.info(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(1, self.config.epochs + 1):
            # Train
            train_loss = self.train_epoch(
                model, train_loader, optimizer, criterion, label_encoder
            )
            
            # Evaluate
            val_metrics = self.evaluate(model, val_loader, criterion, label_encoder)
            
            # Get current LR
            current_lr = optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['val_qwk'].append(val_metrics['qwk'])
            self.history['learning_rates'].append(current_lr)
            
            # Log progress
            logger.info(
                f"Epoch {epoch:3d}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val MAE: {val_metrics['mae']:.4f} | "
                f"Val QWK: {val_metrics['qwk']:.4f} | "
                f"LR: {current_lr:.6f}"
            )
            
            # Update scheduler (using QWK as metric)
            scheduler.step(val_metrics['qwk'])
            
            # Check for improvement
            if val_metrics['qwk'] > self.best_val_qwk + self.config.min_delta:
                self.best_val_qwk = val_metrics['qwk']
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                self.best_model_state = model.state_dict().copy()
                logger.info(f"  ✓ New best QWK: {self.best_val_qwk:.4f}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                logger.info(f"\nEarly stopping after {epoch} epochs (no improvement for {self.config.patience} epochs)")
                break
        
        # Restore best model
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            logger.info(f"\nRestored best model (QWK: {self.best_val_qwk:.4f})")
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time/60:.2f} minutes")
        
        return {
            'best_val_qwk': self.best_val_qwk,
            'best_val_loss': self.best_val_loss,
            'training_time': training_time,
            'epochs_trained': len(self.history['train_loss']),
            'history': self.history
        }


# =============================================================================
# Main Training Function
# =============================================================================

def run_training(config: TrainingConfig) -> dict:
    """
    Run full training pipeline.
    
    Args:
        config: Training configuration
        
    Returns:
        Dictionary with training results and paths to saved artifacts
    """
    logger.info("=" * 80)
    logger.info("DEEP LEARNING TRAINING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Configuration: {config.to_dict()}")
    
    # Create output directories
    experiment_dir = MODEL_DIR / config.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    results_dir = RESULTS_DIR / "training" / config.experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # ==========================================================================
    # Load Data
    # ==========================================================================
    
    logger.info("\n1. Loading data...")
    train_texts, train_labels = load_split('train', DATA_DIR)
    val_texts, val_labels = load_split('val', DATA_DIR)
    test_texts, test_labels = load_split('test', DATA_DIR)
    
    logger.info(f"  Train: {len(train_texts)}")
    logger.info(f"  Val: {len(val_texts)}")
    logger.info(f"  Test: {len(test_texts)}")
    
    # ==========================================================================
    # Extract Features
    # ==========================================================================
    
    train_features = None
    val_features = None
    test_features = None
    feature_dim = 0
    
    if config.use_features:
        logger.info("\n2. Extracting features...")
        extractor = FeatureExtractor()
        train_features = extractor.extract_batch(train_texts)
        val_features = extractor.extract_batch(val_texts)
        test_features = extractor.extract_batch(test_texts)
        feature_dim = train_features.shape[1]
        logger.info(f"  Feature dimension: {feature_dim}")
        
        # Normalize features
        train_mean = train_features.mean(axis=0)
        train_std = train_features.std(axis=0) + 1e-8
        train_features = (train_features - train_mean) / train_std
        val_features = (val_features - train_mean) / train_std
        test_features = (test_features - train_mean) / train_std
        
        # Save normalization params
        np.savez(
            experiment_dir / 'feature_normalization.npz',
            mean=train_mean, std=train_std,
            feature_names=extractor.get_feature_names()
        )
    
    # ==========================================================================
    # Build Vocabulary
    # ==========================================================================
    
    logger.info("\n3. Building vocabulary...")
    vocab = Vocabulary(
        min_freq=config.vocab_min_freq,
        max_size=config.vocab_max_size
    )
    vocab.build(train_texts)
    vocab.save(experiment_dir / 'vocabulary.json')
    logger.info(f"  Vocabulary size: {len(vocab)}")
    
    # Label encoder
    label_encoder = LabelEncoder()
    
    # ==========================================================================
    # Create Data Loaders
    # ==========================================================================
    
    logger.info("\n4. Creating data loaders...")
    
    from torch.utils.data import DataLoader
    
    train_dataset = TextClassificationDataset(
        train_texts, train_labels, vocab, label_encoder,
        max_length=config.max_seq_length,
        features=train_features
    )
    val_dataset = TextClassificationDataset(
        val_texts, val_labels, vocab, label_encoder,
        max_length=config.max_seq_length,
        features=val_features
    )
    test_dataset = TextClassificationDataset(
        test_texts, test_labels, vocab, label_encoder,
        max_length=config.max_seq_length,
        features=test_features
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size,
        shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size,
        shuffle=False, collate_fn=collate_fn
    )
    
    # ==========================================================================
    # Create Model
    # ==========================================================================
    
    logger.info("\n5. Creating model...")
    
    model_kwargs = {
        'dropout': config.dropout,
    }
    
    if config.model_type == 'textcnn':
        model_kwargs['embedding_dim'] = config.embedding_dim
        model_kwargs['num_filters'] = config.num_filters
        model_kwargs['filter_sizes'] = config.filter_sizes
    elif config.model_type == 'bilstm':
        model_kwargs['embedding_dim'] = config.embedding_dim
        model_kwargs['hidden_dim'] = config.hidden_dim
    elif config.model_type == 'features_only':
        model_kwargs['hidden_dim'] = config.hidden_dim
    
    model = create_model(
        model_type=config.model_type,
        vocab_size=len(vocab),
        num_classes=5,
        feature_dim=feature_dim,
        use_coral=config.use_coral,
        **model_kwargs
    )
    
    # ==========================================================================
    # Train
    # ==========================================================================
    
    logger.info("\n6. Training...")
    trainer = Trainer(config)
    training_results = trainer.train(model, train_loader, val_loader, label_encoder)
    
    # ==========================================================================
    # Save Model
    # ==========================================================================
    
    logger.info("\n7. Saving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.to_dict(),
        'vocab_size': len(vocab),
        'feature_dim': feature_dim,
        'training_results': training_results,
    }, experiment_dir / 'model.pt')
    logger.info(f"  Saved to {experiment_dir / 'model.pt'}")
    
    # ==========================================================================
    # Final Evaluation on Test Set
    # ==========================================================================
    
    logger.info("\n8. Evaluating on test set...")
    
    if config.use_coral:
        criterion = CORALLoss(num_classes=5)
    else:
        criterion = OrdinalCrossEntropyLoss(num_classes=5)
    
    test_metrics = trainer.evaluate(model, test_loader, criterion, label_encoder)
    
    logger.info("\n" + "=" * 60)
    logger.info("TEST SET RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Accuracy:   {test_metrics['accuracy']:.4f}")
    logger.info(f"  F1 (macro): {test_metrics['f1_macro']:.4f}")
    logger.info(f"  MAE:        {test_metrics['mae']:.4f}")
    logger.info(f"  QWK:        {test_metrics['qwk']:.4f}")
    logger.info("=" * 60)
    
    # Generate full report
    report = generate_report(
        test_metrics['full_results'],
        model_name=config.experiment_name,
        output_path=results_dir / 'evaluation_report.txt'
    )
    logger.info(f"\n{report}")
    
    # Save all results
    save_results_json(
        test_metrics['full_results'],
        results_dir / 'test_results.json',
        model_name=config.experiment_name
    )
    
    # Save training history
    with open(results_dir / 'training_history.json', 'w') as f:
        json.dump(training_results, f, indent=2)
    
    # Save config
    with open(results_dir / 'config.json', 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    logger.info(f"\n✓ All results saved to {results_dir}")
    
    return {
        'config': config.to_dict(),
        'model_path': str(experiment_dir / 'model.pt'),
        'results_path': str(results_dir),
        'training_results': training_results,
        'test_metrics': {
            'accuracy': test_metrics['accuracy'],
            'f1_macro': test_metrics['f1_macro'],
            'mae': test_metrics['mae'],
            'qwk': test_metrics['qwk'],
        },
        'parameters': model.count_parameters(),
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train deep learning model')
    parser.add_argument('--model', type=str, default='textcnn',
                       choices=['textcnn', 'bilstm', 'features_only'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--no-features', action='store_true')
    parser.add_argument('--coral', action='store_true')
    parser.add_argument('--embedding-dim', type=int, default=64)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--name', type=str, default=None)
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_features=not args.no_features,
        use_coral=args.coral,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        experiment_name=args.name,
    )
    
    results = run_training(config)
    
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")  
    logger.info("=" * 80)
    logger.info(f"Model: {config.model_type}")
    logger.info(f"Parameters: {results['parameters']:,}")
    logger.info(f"Test MAE: {results['test_metrics']['mae']:.4f}")
    logger.info(f"Test QWK: {results['test_metrics']['qwk']:.4f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
