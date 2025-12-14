# Inference Script
# Demonstrates model inference on sample texts

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from utils import setup_logger

logger = setup_logger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def find_best_model():
    """Find the most recently trained model."""
    model_dirs = [d for d in MODEL_DIR.iterdir() if d.is_dir() and (d / 'model.pt').exists()]
    if not model_dirs:
        return None
    # Sort by modification time
    model_dirs.sort(key=lambda x: (x / 'model.pt').stat().st_mtime, reverse=True)
    return model_dirs[0]


def run_inference():
    """Run inference demo on sample texts."""
    
    logger.info("=" * 80)
    logger.info("INFERENCE DEMO")
    logger.info("=" * 80)
    
    # Find model
    model_path = find_best_model()
    if model_path is None:
        logger.warning("No trained model found. Please run training first.")
        return
    
    logger.info(f"Using model: {model_path.name}")
    
    # Sample texts for demonstration
    sample_texts = [
        "A szerződés felmondása írásban történik.",
        "Az adásvételi szerződés érvényességéhez a felek egybehangzó akaratnyilatkozata szükséges, amelyet írásba kell foglalni.",
        "A bíróság ítélete ellen fellebbezésnek van helye a másodfokú bírósághoz a kézbesítéstől számított tizenöt napon belül.",
    ]
    
    logger.info(f"\nSample texts for inference:")
    for i, text in enumerate(sample_texts, 1):
        logger.info(f"  {i}. {text[:80]}{'...' if len(text) > 80 else ''}")
    
    # Try to load model and run inference
    try:
        import torch
        from dataset import Vocabulary, LabelEncoder
        from models import create_model
        
        # Load model checkpoint
        checkpoint = torch.load(model_path / 'model.pt', map_location='cpu')
        config = checkpoint['config']
        vocab_size = checkpoint['vocab_size']
        feature_dim = checkpoint['feature_dim']
        
        logger.info(f"\nModel configuration:")
        logger.info(f"  Model type: {config['model_type']}")
        logger.info(f"  Embedding dim: {config.get('embedding_dim', 'N/A')}")
        logger.info(f"  CORAL loss: {config.get('use_coral', False)}")
        
        # Load vocabulary
        vocab = Vocabulary()
        vocab.load(model_path / 'vocabulary.json')
        
        label_encoder = LabelEncoder()
        
        # Create model
        model_kwargs = {'dropout': 0.0}
        if config['model_type'] == 'textcnn':
            model_kwargs['embedding_dim'] = config.get('embedding_dim', 64)
            model_kwargs['num_filters'] = config.get('num_filters', 32)
            model_kwargs['filter_sizes'] = tuple(config.get('filter_sizes', (2, 3, 4, 5)))
        
        model = create_model(
            model_type=config['model_type'],
            vocab_size=vocab_size,
            num_classes=5,
            feature_dim=feature_dim,
            use_coral=config.get('use_coral', False),
            **model_kwargs
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Feature extraction
        if feature_dim > 0:
            from features import FeatureExtractor
            import numpy as np
            extractor = FeatureExtractor()
            features = extractor.extract_batch(sample_texts)
            # Load normalization params
            norm_path = model_path / 'feature_normalization.npz'
            if norm_path.exists():
                norm = np.load(norm_path)
                features = (features - norm['mean']) / norm['std']
            features = torch.tensor(features, dtype=torch.float32)
        else:
            features = None
        
        # Tokenize
        token_ids = []
        for text in sample_texts:
            tokens = text.lower().split()[:256]
            ids = [vocab.word2idx.get(t, vocab.word2idx['<UNK>']) for t in tokens]
            token_ids.append(ids)
        
        # Pad
        max_len = max(len(ids) for ids in token_ids)
        lengths = [len(ids) for ids in token_ids]
        padded = [ids + [0] * (max_len - len(ids)) for ids in token_ids]
        token_ids_tensor = torch.tensor(padded, dtype=torch.long)
        
        # Predict
        with torch.no_grad():
            logits = model(token_ids_tensor, lengths, features)
            predictions = model.predict(logits)
        
        # Decode predictions
        logger.info("\n" + "=" * 60)
        logger.info("PREDICTIONS")
        logger.info("=" * 60)
        
        for i, (text, pred) in enumerate(zip(sample_texts, predictions), 1):
            label = label_encoder.decode(pred.item())
            logger.info(f"\n  Text {i}: {text[:60]}{'...' if len(text) > 60 else ''}")
            logger.info(f"  Predicted: {label}")
        
        logger.info("\n" + "=" * 60)
        logger.info("Inference completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        logger.info("Skipping inference demo due to missing dependencies or model.")


if __name__ == "__main__":
    run_inference()
