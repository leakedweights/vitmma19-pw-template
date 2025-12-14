# Configuration Settings for Legal Text Understandability Classification
# Central location for all project configurations

from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, List, Optional

# =============================================================================
# Path Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"


# =============================================================================
# Label Configuration
# =============================================================================

# 5-class ordinal labels (1 = hardest, 5 = easiest)
LABELS = [
    "1-Nagyon nehezen érthető",  # Very hard to understand
    "2-Nehezen érthető",          # Hard to understand
    "3-Többé/kevésbé megértem",   # Somewhat understandable
    "4-Érthető",                   # Understandable
    "5-Könnyen érthető",          # Easy to understand
]

NUM_CLASSES = len(LABELS)


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Model architecture
    model_type: str = 'textcnn'
    embedding_dim: int = 32
    hidden_dim: int = 64
    num_filters: int = 32
    filter_sizes: Tuple[int, ...] = (2, 3, 4, 5)
    dropout: float = 0.3
    
    # Features
    use_features: bool = True
    use_coral: bool = True
    
    # Vocabulary
    vocab_min_freq: int = 2
    vocab_max_size: int = 8000
    max_seq_length: int = 256
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    patience: int = 10
    min_delta: float = 0.001
    
    # Experiment
    experiment_name: Optional[str] = None


# =============================================================================
# Model Presets
# =============================================================================

# Best configurations found through hyperparameter search
MODEL_PRESETS = {
    # Best efficiency: 64K params, MAE=0.67
    'textcnn_micro': TrainingConfig(
        model_type='textcnn',
        embedding_dim=8,
        num_filters=32,
        use_coral=True,
        learning_rate=1e-3,
    ),
    
    # Balanced: 126K params, MAE=0.68
    'textcnn_tiny': TrainingConfig(
        model_type='textcnn',
        embedding_dim=16,
        num_filters=32,
        use_coral=True,
        learning_rate=1e-3,
    ),
    
    # Small: 249K params, MAE=0.70
    'textcnn_small': TrainingConfig(
        model_type='textcnn',
        embedding_dim=32,
        num_filters=32,
        use_coral=True,
        learning_rate=5e-4,
    ),
    
    # Standard: 496K params, QWK=0.53
    'textcnn_standard': TrainingConfig(
        model_type='textcnn',
        embedding_dim=64,
        num_filters=32,
        use_coral=True,
        learning_rate=1e-3,
    ),
    
    # Minimal params: 4.7K params, MAE=0.76
    'features_only': TrainingConfig(
        model_type='features_only',
        hidden_dim=64,
        use_coral=False,
        use_features=True,
    ),
    
    # Nano: 33K params, MAE=0.70
    'textcnn_nano': TrainingConfig(
        model_type='textcnn',
        embedding_dim=4,
        num_filters=32,
        use_coral=True,
        learning_rate=1e-3,
    ),
}

# Default preset
DEFAULT_PRESET = 'textcnn_micro'


# =============================================================================
# Hyperparameter Search Configuration
# =============================================================================

HP_SEARCH_GRID = {
    'embedding_dim': [4, 8, 16, 32, 64, 128],
    'num_filters': [16, 32, 64],
    'dropout': [0.2, 0.3, 0.5],
    'learning_rate': [1e-3, 5e-4, 1e-4],
}


# =============================================================================
# Feature Configuration
# =============================================================================

FEATURE_GROUPS = [
    'lexical',      # Word-level features (avg word length, unique ratio, etc.)
    'syntactic',    # Sentence-level features (avg sentence length, punctuation)
    'legal',        # Domain-specific (legal terms, references)
    'readability',  # Readability indices (Flesch, ARI, etc.)
    'structural',   # Document structure (paragraphs, lists, etc.)
]

TOTAL_FEATURES = 37  # Number of extracted features


# =============================================================================
# Data Split Configuration
# =============================================================================

DATA_SPLIT = {
    'train': 0.70,
    'val': 0.15,
    'test': 0.15,
    'random_state': 42,
}


# =============================================================================
# Evaluation Configuration
# =============================================================================

# Primary metrics for model comparison
PRIMARY_METRICS = ['mae', 'qwk', 'off_by_1_accuracy']

# Ordinal error thresholds
ORDINAL_ERROR_THRESHOLDS = [1, 2, 3]  # For off-by-k accuracy


# =============================================================================
# Experiment Results Summary
# =============================================================================

# Best results achieved during development
EXPERIMENT_RESULTS = {
    'baseline': {
        'params': 1000,
        'mae': 0.94,
        'qwk': 0.48,
        'f1': 0.26,
        'off_by_1': 0.75,
    },
    'features_only': {
        'params': 4677,
        'mae': 0.76,
        'qwk': 0.45,
        'f1': 0.33,
        'off_by_1': 0.81,
    },
    'textcnn_nano': {
        'params': 33038,
        'mae': 0.70,
        'qwk': 0.49,
        'f1': 0.35,
        'off_by_1': 0.87,
    },
    'textcnn_micro': {
        'params': 63878,
        'mae': 0.67,
        'qwk': 0.50,
        'f1': 0.36,
        'off_by_1': 0.87,
    },
    'textcnn_tiny': {
        'params': 125558,
        'mae': 0.68,
        'qwk': 0.49,
        'f1': 0.36,
        'off_by_1': 0.87,
    },
    'textcnn_small': {
        'params': 248918,
        'mae': 0.70,
        'qwk': 0.51,
        'f1': 0.35,
        'off_by_1': 0.87,
    },
    'textcnn_standard': {
        'params': 495638,
        'mae': 0.71,
        'qwk': 0.53,
        'f1': 0.35,
        'off_by_1': 0.88,
    },
}


def get_config(preset: str = None) -> TrainingConfig:
    """Get a training configuration by preset name."""
    if preset is None:
        preset = DEFAULT_PRESET
    if preset not in MODEL_PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(MODEL_PRESETS.keys())}")
    return MODEL_PRESETS[preset]


def list_presets():
    """List all available model presets with their key metrics."""
    print("\nAvailable Model Presets:")
    print("-" * 80)
    print(f"{'Preset':<20} {'Params':>10} {'MAE':>8} {'QWK':>8} {'F1':>8} {'Off-by-1':>10}")
    print("-" * 80)
    
    for name, config in MODEL_PRESETS.items():
        # Get results if available
        result = EXPERIMENT_RESULTS.get(name, {})
        params = result.get('params', '?')
        mae = result.get('mae', '?')
        qwk = result.get('qwk', '?')
        f1 = result.get('f1', '?')
        off_by_1 = result.get('off_by_1', '?')
        
        mae_str = f"{mae:.2f}" if isinstance(mae, float) else str(mae)
        qwk_str = f"{qwk:.2f}" if isinstance(qwk, float) else str(qwk)
        f1_str = f"{f1:.2f}" if isinstance(f1, float) else str(f1)
        off_by_1_str = f"{off_by_1:.0%}" if isinstance(off_by_1, float) else str(off_by_1)
        params_str = f"{params:,}" if isinstance(params, int) else str(params)
        
        print(f"{name:<20} {params_str:>10} {mae_str:>8} {qwk_str:>8} {f1_str:>8} {off_by_1_str:>10}")


if __name__ == "__main__":
    list_presets()
