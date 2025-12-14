# Overfitting on Single Batch Test
# Tests minimum model capacity by trying to overfit a single batch.

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from utils import setup_logger
logger = setup_logger(__name__)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import Vocabulary, LabelEncoder, TextClassificationDataset, collate_fn, load_split
from features import FeatureExtractor
from models import create_model, CORALLoss

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def overfit_single_batch(
    model_type: str = 'textcnn',
    embedding_dim: int = 32,
    num_filters: int = 16,
    hidden_dim: int = 32,
    use_features: bool = True,
    batch_size: int = 32,
    max_epochs: int = 200,
    target_loss: float = 0.01,
):
    """
    Try to overfit a single batch to test model capacity.
    
    If the model can reach near-zero loss, it has sufficient capacity.
    If it plateaus at high loss, the model is too small.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load data
    train_texts, train_labels = load_split('train', DATA_DIR)
    
    # Extract features
    feature_dim = 0
    train_features = None
    if use_features:
        extractor = FeatureExtractor()
        train_features = extractor.extract_batch(train_texts[:batch_size])
        # Normalize
        train_features = (train_features - train_features.mean(axis=0)) / (train_features.std(axis=0) + 1e-8)
        feature_dim = train_features.shape[1]
    
    # Build vocab (small, just from this batch)
    vocab = Vocabulary(min_freq=1, max_size=5000)
    vocab.build(train_texts[:batch_size])
    
    label_encoder = LabelEncoder()
    
    # Create single-batch dataset
    dataset = TextClassificationDataset(
        train_texts[:batch_size],
        train_labels[:batch_size],
        vocab, label_encoder,
        max_length=256,
        features=train_features
    )
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(loader))
    
    # Move to device
    token_ids = batch['token_ids'].to(device)
    labels = batch['labels'].to(device)
    lengths = batch['lengths']
    features = batch.get('features')
    if features is not None:
        features = features.to(device)
    
    # Create model
    model_kwargs = {'dropout': 0.0}  # No dropout for overfitting test
    if model_type == 'textcnn':
        model_kwargs['embedding_dim'] = embedding_dim
        model_kwargs['num_filters'] = num_filters
        model_kwargs['filter_sizes'] = (2, 3, 4)
    elif model_type == 'bilstm':
        model_kwargs['embedding_dim'] = embedding_dim
        model_kwargs['hidden_dim'] = hidden_dim
    elif model_type == 'features_only':
        model_kwargs['hidden_dim'] = hidden_dim
    
    model = create_model(
        model_type=model_type,
        vocab_size=len(vocab),
        num_classes=5,
        feature_dim=feature_dim,
        use_coral=False,  # Use standard CE for clearer loss
        **model_kwargs
    )
    model = model.to(device)
    
    params = model.count_parameters()
    logger.info(f"\n{'='*60}")
    logger.info(f"OVERFIT TEST: {model_type}")
    logger.info(f"{'='*60}")
    logger.info(f"Parameters: {params:,}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Features: {feature_dim if use_features else 'None'}")
    logger.info(f"{'='*60}\n")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    
    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad()
        logits = model(token_ids, lengths, features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        # Get accuracy
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean().item()
        
        if epoch % 20 == 0 or epoch == 1 or loss.item() < target_loss:
            logger.info(f"Epoch {epoch:4d} | Loss: {loss.item():.6f} | Acc: {acc:.2%}")
        
        if loss.item() < target_loss:
            logger.info(f"\n✓ Reached target loss {target_loss} at epoch {epoch}")
            logger.info(f"  Model can overfit with {params:,} parameters")
            return True, params, epoch
    
    logger.info(f"\n✗ Did not reach target loss after {max_epochs} epochs")
    logger.info(f"  Final loss: {loss.item():.6f}")
    logger.info(f"  Model may be too small with {params:,} parameters")
    return False, params, max_epochs


def find_minimum_capacity():
    """Find the minimum model size that can overfit a batch."""
    
    logger.info("=" * 70)
    logger.info("FINDING MINIMUM MODEL CAPACITY")
    logger.info("=" * 70)
    
    results = []
    
    # Test different configurations
    configs = [
        # TextCNN variants
        {'model_type': 'textcnn', 'embedding_dim': 8, 'num_filters': 4},
        {'model_type': 'textcnn', 'embedding_dim': 8, 'num_filters': 8},
        {'model_type': 'textcnn', 'embedding_dim': 16, 'num_filters': 8},
        {'model_type': 'textcnn', 'embedding_dim': 16, 'num_filters': 16},
        {'model_type': 'textcnn', 'embedding_dim': 32, 'num_filters': 16},
        {'model_type': 'textcnn', 'embedding_dim': 32, 'num_filters': 32},
        
        # Features only
        {'model_type': 'features_only', 'hidden_dim': 16},
        {'model_type': 'features_only', 'hidden_dim': 32},
        {'model_type': 'features_only', 'hidden_dim': 64},
    ]
    
    for config in configs:
        success, params, epochs = overfit_single_batch(**config, max_epochs=200)
        results.append({
            **config,
            'params': params,
            'success': success,
            'epochs': epochs
        })
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Model':<20} {'Config':<25} {'Params':>10} {'Overfit?':>10}")
    logger.info("-" * 70)
    
    for r in results:
        model = r['model_type']
        if model == 'textcnn':
            config = f"emb={r['embedding_dim']}, filt={r['num_filters']}"
        else:
            config = f"hidden={r['hidden_dim']}"
        status = "✓ Yes" if r['success'] else "✗ No"
        logger.info(f"{model:<20} {config:<25} {r['params']:>10,} {status:>10}")
    
    # Find minimum
    successful = [r for r in results if r['success']]
    if successful:
        min_model = min(successful, key=lambda x: x['params'])
        logger.info(f"\n✓ Minimum capacity: {min_model['params']:,} params")
    else:
        logger.info("\n✗ No configuration could overfit the batch")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--full-search', action='store_true', help='Run full search')
    parser.add_argument('--model', type=str, default='textcnn')
    parser.add_argument('--embedding-dim', type=int, default=16)
    parser.add_argument('--num-filters', type=int, default=8)
    parser.add_argument('--hidden-dim', type=int, default=32)
    args = parser.parse_args()
    
    if args.full_search:
        find_minimum_capacity()
    else:
        overfit_single_batch(
            model_type=args.model,
            embedding_dim=args.embedding_dim,
            num_filters=args.num_filters,
            hidden_dim=args.hidden_dim,
        )
