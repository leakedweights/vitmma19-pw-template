# Deep Learning Models for Legal Text Classification
# Lightweight architectures optimized for minimal parameter count.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from utils import setup_logger

logger = setup_logger(__name__)


# =============================================================================
# Ordinal Loss Functions
# =============================================================================

class OrdinalCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss with ordinal distance weighting.
    Penalizes predictions more heavily if they are farther from the true class.
    """
    
    def __init__(self, num_classes: int, weight_power: float = 2.0):
        """
        Args:
            num_classes: Number of ordinal classes
            weight_power: Power for distance weighting (2.0 = quadratic)
        """
        super().__init__()
        self.num_classes = num_classes
        self.weight_power = weight_power
        
        # Precompute distance weights for each (true, pred) pair
        # Shape: (num_classes, num_classes)
        weights = torch.zeros(num_classes, num_classes)
        for i in range(num_classes):
            for j in range(num_classes):
                weights[i, j] = 1.0 + abs(i - j) ** weight_power
        self.register_buffer('weights', weights)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model output (batch_size, num_classes)
            targets: True labels (batch_size,)
        """
        # Get per-sample weights based on the difference between pred and true
        batch_weights = self.weights[targets]  # (batch_size, num_classes)
        
        # Weighted cross-entropy
        log_probs = F.log_softmax(logits, dim=-1)
        weighted_log_probs = log_probs * batch_weights
        
        # Standard cross-entropy on weighted log probs
        loss = F.nll_loss(weighted_log_probs, targets)
        return loss


class CORALLoss(nn.Module):
    """
    CORAL (Consistent Ordinal Ranking) loss for ordinal regression.
    Treats ordinal classification as K-1 binary classification problems.
    
    Reference: Cao et al., "Rank Consistent Ordinal Regression for Neural Networks"
    """
    
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model output (batch_size, num_classes - 1) representing cumulative probabilities
            targets: True ordinal labels (batch_size,) with values 0 to num_classes-1
        """
        batch_size = targets.size(0)
        
        # Create binary labels for each threshold
        # For class k, binary label at threshold i is 1 if k > i
        levels = torch.arange(self.num_classes - 1, device=targets.device)
        binary_labels = (targets.unsqueeze(1) > levels).float()  # (batch_size, num_classes - 1)
        
        # Binary cross-entropy for each threshold
        loss = F.binary_cross_entropy_with_logits(logits, binary_labels)
        return loss


def coral_predict(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert CORAL logits to class predictions.
    
    Args:
        logits: (batch_size, num_classes - 1)
        
    Returns:
        Class predictions (batch_size,)
    """
    probs = torch.sigmoid(logits)
    # Count how many thresholds are exceeded
    predictions = (probs > 0.5).sum(dim=1)
    return predictions


# =============================================================================
# Model Components
# =============================================================================

class AttentionLayer(nn.Module):
    """Simple attention mechanism for sequence classification."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            mask: (batch_size, seq_len) with 1 for valid positions, 0 for padding
            
        Returns:
            Context vector (batch_size, hidden_size)
        """
        # Compute attention scores
        scores = self.attention(hidden_states).squeeze(-1)  # (batch_size, seq_len)
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax over sequence
        weights = F.softmax(scores, dim=-1)  # (batch_size, seq_len)
        
        # Weighted sum
        context = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        return context


# =============================================================================
# TextCNN Model
# =============================================================================

class TextCNN(nn.Module):
    """
    Convolutional Neural Network for text classification.
    Very parameter-efficient architecture.
    
    Architecture:
        Embedding -> Conv1D (multiple filter sizes) -> MaxPool -> Dropout -> Dense
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        num_classes: int = 5,
        num_filters: int = 32,
        filter_sizes: tuple = (2, 3, 4, 5),
        dropout: float = 0.3,
        use_coral: bool = False,
        feature_dim: int = 0,
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            num_classes: Number of output classes
            num_filters: Number of filters per filter size
            filter_sizes: Tuple of filter sizes (n-grams)
            dropout: Dropout probability
            use_coral: Whether to use CORAL ordinal regression
            feature_dim: Dimension of additional features (0 = no extra features)
        """
        super().__init__()
        
        self.use_coral = use_coral
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, fs)
            for fs in filter_sizes
        ])
        
        # Calculate total features before classifier
        cnn_out_dim = num_filters * len(filter_sizes)
        classifier_in_dim = cnn_out_dim + feature_dim
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Feature projection (if using extra features)
        if feature_dim > 0:
            self.feature_proj = nn.Linear(feature_dim, feature_dim)
        
        # Classifier
        if use_coral:
            self.classifier = nn.Linear(classifier_in_dim, num_classes - 1)
        else:
            self.classifier = nn.Linear(classifier_in_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight)
        nn.init.xavier_uniform_(self.classifier.weight)
    
    def forward(
        self,
        token_ids: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            token_ids: (batch_size, seq_len)
            lengths: (batch_size,) - not used but kept for API consistency
            features: (batch_size, feature_dim) - additional features
            
        Returns:
            Logits (batch_size, num_classes) or (batch_size, num_classes-1) for CORAL
        """
        # Embedding: (batch_size, seq_len, embedding_dim)
        x = self.embedding(token_ids)
        
        # Transpose for conv1d: (batch_size, embedding_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Apply convolutions and max pooling
        conv_outputs = []
        for conv in self.convs:
            # Conv: (batch_size, num_filters, seq_len - filter_size + 1)
            c = F.relu(conv(x))
            # Max pool over time
            c = F.max_pool1d(c, c.size(2)).squeeze(2)  # (batch_size, num_filters)
            conv_outputs.append(c)
        
        # Concatenate all filter outputs
        x = torch.cat(conv_outputs, dim=1)  # (batch_size, num_filters * len(filter_sizes))
        
        # Add extra features if provided
        if features is not None and self.feature_dim > 0:
            feat_proj = F.relu(self.feature_proj(features))
            x = torch.cat([x, feat_proj], dim=1)
        
        # Dropout and classify
        x = self.dropout(x)
        logits = self.classifier(x)
        
        return logits
    
    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        """Get class predictions from logits."""
        if self.use_coral:
            return coral_predict(logits)
        return logits.argmax(dim=-1)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# BiLSTM with Attention
# =============================================================================

class BiLSTMAttention(nn.Module):
    """
    Bidirectional LSTM with attention for text classification.
    Slightly more parameters than TextCNN but potentially better for sequences.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 1,
        num_classes: int = 5,
        dropout: float = 0.3,
        use_coral: bool = False,
        feature_dim: int = 0,
    ):
        super().__init__()
        
        self.use_coral = use_coral
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # BiLSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Attention
        self.attention = AttentionLayer(hidden_dim * 2)
        
        # Calculate classifier input dimension
        lstm_out_dim = hidden_dim * 2  # bidirectional
        classifier_in_dim = lstm_out_dim + feature_dim
        
        # Feature projection
        if feature_dim > 0:
            self.feature_proj = nn.Linear(feature_dim, feature_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Classifier
        if use_coral:
            self.classifier = nn.Linear(classifier_in_dim, num_classes - 1)
        else:
            self.classifier = nn.Linear(classifier_in_dim, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.classifier.weight)
    
    def forward(
        self,
        token_ids: torch.Tensor,
        lengths: torch.Tensor,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            token_ids: (batch_size, seq_len)
            lengths: (batch_size,) actual lengths (before padding)
            features: (batch_size, feature_dim)
        """
        batch_size, seq_len = token_ids.shape
        
        # Embedding
        x = self.embedding(token_ids)  # (batch_size, seq_len, embedding_dim)
        
        # Pack sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # LSTM
        lstm_out, _ = self.lstm(packed)
        
        # Unpack
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True, total_length=seq_len
        )
        
        # Create mask
        mask = (token_ids != 0).float()
        
        # Attention
        x = self.attention(lstm_out, mask)
        
        # Add features
        if features is not None and self.feature_dim > 0:
            feat_proj = F.relu(self.feature_proj(features))
            x = torch.cat([x, feat_proj], dim=1)
        
        # Classify
        x = self.dropout(x)
        logits = self.classifier(x)
        
        return logits
    
    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        if self.use_coral:
            return coral_predict(logits)
        return logits.argmax(dim=-1)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Feature-Only Model (as baseline comparison)
# =============================================================================

class FeatureOnlyModel(nn.Module):
    """
    Simple MLP that only uses engineered features.
    Minimal parameters, good for comparison.
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        num_classes: int = 5,
        dropout: float = 0.3,
        use_coral: bool = False,
    ):
        super().__init__()
        
        self.use_coral = use_coral
        self.num_classes = num_classes
        
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        output_dim = num_classes - 1 if use_coral else num_classes
        self.classifier = nn.Linear(hidden_dim // 2, output_dim)
    
    def forward(
        self,
        token_ids: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        features: torch.Tensor = None,
    ) -> torch.Tensor:
        x = self.layers(features)
        return self.classifier(x)
    
    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        if self.use_coral:
            return coral_predict(logits)
        return logits.argmax(dim=-1)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Model Factory
# =============================================================================

def create_model(
    model_type: str,
    vocab_size: int,
    num_classes: int = 5,
    feature_dim: int = 0,
    use_coral: bool = False,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: 'textcnn', 'bilstm', or 'features_only'
        vocab_size: Vocabulary size
        num_classes: Number of output classes
        feature_dim: Dimension of additional features
        use_coral: Whether to use CORAL ordinal regression
        **kwargs: Additional model-specific arguments
    """
    if model_type == 'textcnn':
        model = TextCNN(
            vocab_size=vocab_size,
            num_classes=num_classes,
            feature_dim=feature_dim,
            use_coral=use_coral,
            **kwargs
        )
    elif model_type == 'bilstm':
        model = BiLSTMAttention(
            vocab_size=vocab_size,
            num_classes=num_classes,
            feature_dim=feature_dim,
            use_coral=use_coral,
            **kwargs
        )
    elif model_type == 'features_only':
        model = FeatureOnlyModel(
            feature_dim=feature_dim,
            num_classes=num_classes,
            use_coral=use_coral,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    logger.info(f"Created {model_type} model with {model.count_parameters():,} parameters")
    return model


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    # Test model creation and forward pass
    batch_size = 4
    seq_len = 32
    vocab_size = 5000
    feature_dim = 37
    
    # Random inputs
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    lengths = torch.tensor([seq_len, seq_len-5, seq_len-10, seq_len-15])
    features = torch.randn(batch_size, feature_dim)
    
    logger.info("Testing TextCNN...")
    model = create_model('textcnn', vocab_size, feature_dim=feature_dim)
    logits = model(token_ids, lengths, features)
    logger.info(f"  Output shape: {logits.shape}")
    logger.info(f"  Parameters: {model.count_parameters():,}")
    
    logger.info("\nTesting BiLSTM...")
    model = create_model('bilstm', vocab_size, feature_dim=feature_dim)
    logits = model(token_ids, lengths, features)
    logger.info(f"  Output shape: {logits.shape}")
    logger.info(f"  Parameters: {model.count_parameters():,}")
    
    logger.info("\nTesting FeatureOnly...")
    model = create_model('features_only', vocab_size, feature_dim=feature_dim)
    logits = model(features=features)
    logger.info(f"  Output shape: {logits.shape}")
    logger.info(f"  Parameters: {model.count_parameters():,}")
    
    logger.info("\nTesting CORAL loss...")
    model = create_model('textcnn', vocab_size, use_coral=True)
    logits = model(token_ids)
    logger.info(f"  CORAL output shape: {logits.shape}")
    
    loss_fn = CORALLoss(num_classes=5)
    targets = torch.randint(0, 5, (batch_size,))
    loss = loss_fn(logits, targets)
    logger.info(f"  CORAL loss: {loss.item():.4f}")
    
    logger.info("\n✓ All models working correctly!")
