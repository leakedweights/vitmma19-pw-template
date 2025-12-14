# PyTorch Dataset Utilities for Legal Text Classification
# Handles data loading, tokenization, and batching.

import json
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from utils import setup_logger

logger = setup_logger(__name__)


# =============================================================================
# Vocabulary
# =============================================================================

class Vocabulary:
    """
    Simple vocabulary class for mapping words to indices.
    """
    
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    
    def __init__(self, min_freq: int = 2, max_size: Optional[int] = None):
        """
        Args:
            min_freq: Minimum frequency for a word to be included
            max_size: Maximum vocabulary size (None = unlimited)
        """
        self.min_freq = min_freq
        self.max_size = max_size
        self.word2idx = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.idx2word = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}
        self.word_freq = Counter()
        self._built = False
    
    def build(self, texts: List[str]) -> 'Vocabulary':
        """
        Build vocabulary from a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            self (for chaining)
        """
        logger.info("Building vocabulary...")
        
        # Count word frequencies
        for text in texts:
            words = self._tokenize(text)
            self.word_freq.update(words)
        
        # Filter by frequency and sort
        vocab_words = [
            word for word, freq in self.word_freq.most_common()
            if freq >= self.min_freq
        ]
        
        # Apply max size limit
        if self.max_size is not None:
            vocab_words = vocab_words[:self.max_size - 2]  # Reserve for PAD, UNK
        
        # Build mappings
        for idx, word in enumerate(vocab_words, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        self._built = True
        logger.info(f"  Vocabulary size: {len(self.word2idx)}")
        
        return self
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization by splitting on non-alphanumeric."""
        return re.findall(r'\b\w+\b', text.lower())
    
    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """
        Encode text to list of word indices.
        
        Args:
            text: Input text
            max_length: Maximum sequence length (truncate if longer)
            
        Returns:
            List of word indices
        """
        words = self._tokenize(text)
        indices = [self.word2idx.get(w, self.word2idx[self.UNK_TOKEN]) for w in words]
        
        if max_length is not None and len(indices) > max_length:
            indices = indices[:max_length]
        
        return indices
    
    def __len__(self) -> int:
        return len(self.word2idx)
    
    def save(self, path: Path) -> None:
        """Save vocabulary to file."""
        data = {
            'word2idx': self.word2idx,
            'min_freq': self.min_freq,
            'max_size': self.max_size,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: Path) -> 'Vocabulary':
        """Load vocabulary from file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        vocab = cls(min_freq=data['min_freq'], max_size=data['max_size'])
        vocab.word2idx = data['word2idx']
        vocab.idx2word = {int(v): k for k, v in data['word2idx'].items()}
        vocab._built = True
        
        return vocab


# =============================================================================
# Label Encoder
# =============================================================================

class LabelEncoder:
    """Encode ordinal labels to integers."""
    
    # Default ordering for this task
    DEFAULT_LABELS = [
        "1-Nagyon nehezen érthető",
        "2-Nehezen érthető",
        "3-Többé/kevésbé megértem",
        "4-Érthető",
        "5-Könnyen érthető",
    ]
    
    def __init__(self, labels: Optional[List[str]] = None):
        self.labels = labels or self.DEFAULT_LABELS
        self.label2idx = {label: idx for idx, label in enumerate(self.labels)}
        self.idx2label = {idx: label for idx, label in enumerate(self.labels)}
    
    def encode(self, label: str) -> int:
        return self.label2idx[label]
    
    def decode(self, idx: int) -> str:
        return self.idx2label[idx]
    
    def encode_batch(self, labels: List[str]) -> List[int]:
        return [self.encode(l) for l in labels]
    
    def decode_batch(self, indices: List[int]) -> List[str]:
        return [self.decode(i) for i in indices]
    
    @property
    def num_classes(self) -> int:
        return len(self.labels)


# =============================================================================
# Dataset Classes
# =============================================================================

class TextClassificationDataset(Dataset):
    """
    PyTorch Dataset for text classification.
    Returns tokenized text and label.
    """
    
    def __init__(
        self,
        texts: List[str],
        labels: List[str],
        vocab: Vocabulary,
        label_encoder: LabelEncoder,
        max_length: int = 256,
        features: Optional[np.ndarray] = None
    ):
        """
        Args:
            texts: List of text strings
            labels: List of label strings
            vocab: Vocabulary for encoding
            label_encoder: Label encoder
            max_length: Maximum sequence length
            features: Optional pre-extracted features array
        """
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.label_encoder = label_encoder
        self.max_length = max_length
        self.features = features
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Encode text
        token_ids = self.vocab.encode(text, max_length=self.max_length)
        
        # Encode label
        label_idx = self.label_encoder.encode(label)
        
        result = {
            'token_ids': torch.tensor(token_ids, dtype=torch.long),
            'label': torch.tensor(label_idx, dtype=torch.long),
            'length': len(token_ids),
        }
        
        # Add features if available
        if self.features is not None:
            result['features'] = torch.tensor(self.features[idx], dtype=torch.float32)
        
        return result


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    Pads sequences to the same length within a batch.
    """
    # Get max length in batch
    max_len = max(item['length'] for item in batch)
    
    # Pad token_ids
    padded_ids = []
    for item in batch:
        ids = item['token_ids']
        padding = torch.zeros(max_len - len(ids), dtype=torch.long)
        padded_ids.append(torch.cat([ids, padding]))
    
    result = {
        'token_ids': torch.stack(padded_ids),
        'labels': torch.stack([item['label'] for item in batch]),
        'lengths': torch.tensor([item['length'] for item in batch]),
    }
    
    # Add features if present
    if 'features' in batch[0]:
        result['features'] = torch.stack([item['features'] for item in batch])
    
    return result


# =============================================================================
# Data Loading Utilities
# =============================================================================

def load_split(
    split: str,
    data_dir: Path = Path("data/processed")
) -> Tuple[List[str], List[str]]:
    """
    Load a data split (train/val/test).
    
    Args:
        split: 'train', 'val', or 'test'
        data_dir: Path to processed data directory
        
    Returns:
        Tuple of (texts, labels)
    """
    file_path = data_dir / f"{split}.json"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    texts = [doc['content'] for doc in data]
    labels = [doc['label'] for doc in data]
    
    return texts, labels


def create_data_loaders(
    train_texts: List[str],
    train_labels: List[str],
    val_texts: List[str],
    val_labels: List[str],
    vocab: Vocabulary,
    label_encoder: LabelEncoder,
    batch_size: int = 32,
    max_length: int = 256,
    train_features: Optional[np.ndarray] = None,
    val_features: Optional[np.ndarray] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = TextClassificationDataset(
        train_texts, train_labels, vocab, label_encoder, max_length, train_features
    )
    val_dataset = TextClassificationDataset(
        val_texts, val_labels, vocab, label_encoder, max_length, val_features
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    
    return train_loader, val_loader


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    from pathlib import Path
    
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data" / "processed"
    
    # Load data
    train_texts, train_labels = load_split('train', DATA_DIR)
    val_texts, val_labels = load_split('val', DATA_DIR)
    
    logger.info(f"Train: {len(train_texts)}, Val: {len(val_texts)}")
    
    # Build vocabulary
    vocab = Vocabulary(min_freq=2, max_size=10000)
    vocab.build(train_texts)
    
    # Create label encoder
    label_encoder = LabelEncoder()
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_texts, train_labels,
        val_texts, val_labels,
        vocab, label_encoder,
        batch_size=16,
        max_length=128
    )
    
    # Test batch
    batch = next(iter(train_loader))
    logger.info(f"Batch token_ids shape: {batch['token_ids'].shape}")
    logger.info(f"Batch labels shape: {batch['labels'].shape}")
    logger.info(f"Batch lengths: {batch['lengths'][:5]}")
    
    logger.info("✓ Dataset utilities working correctly!")
