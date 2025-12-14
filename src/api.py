# ML as a Service - FastAPI Backend
# Serves the trained model for inference

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np

from utils import setup_logger

logger = setup_logger(__name__)

# Initialize app
app = FastAPI(
    title="Legal Text Understandability API",
    description="Classify Hungarian legal text by readability level",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
model = None
vocab = None
feature_extractor = None
feature_norm = None
label_encoder = None
config = None


# =============================================================================
# Request/Response Models
# =============================================================================

class PredictionRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    text: str
    prediction: str
    confidence: float
    class_probabilities: dict
    

class BatchPredictionRequest(BaseModel):
    texts: List[str]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]


class ModelInfo(BaseModel):
    model_type: str
    parameters: int
    embedding_dim: int
    use_coral: bool
    labels: List[str]


# =============================================================================
# Model Loading
# =============================================================================

def load_model():
    """Load the trained model."""
    global model, vocab, feature_extractor, feature_norm, label_encoder, config
    
    import torch
    from dataset import Vocabulary, LabelEncoder
    from models import create_model
    from features import FeatureExtractor
    
    PROJECT_ROOT = Path(__file__).parent.parent
    MODEL_DIR = PROJECT_ROOT / "models"
    
    # Find best model
    model_dirs = [d for d in MODEL_DIR.iterdir() if d.is_dir() and (d / 'model.pt').exists()]
    if not model_dirs:
        logger.error("No trained model found!")
        return False
    
    # Use most recent
    model_dirs.sort(key=lambda x: (x / 'model.pt').stat().st_mtime, reverse=True)
    model_path = model_dirs[0]
    
    logger.info(f"Loading model from {model_path.name}...")
    
    # Load checkpoint
    checkpoint = torch.load(model_path / 'model.pt', map_location='cpu')
    config = checkpoint['config']
    vocab_size = checkpoint['vocab_size']
    feature_dim = checkpoint['feature_dim']
    
    # Load vocabulary
    vocab = Vocabulary()
    vocab_data = checkpoint.get('vocab')
    if vocab_data is None and (model_path / 'vocabulary.json').exists():
        vocab = Vocabulary.load(model_path / 'vocabulary.json')
    else:
        vocab.word2idx = vocab_data['word2idx'] if vocab_data else {}
        vocab._built = True
    
    # Load feature normalization
    norm_path = model_path / 'feature_normalization.npz'
    if norm_path.exists():
        feature_norm = np.load(norm_path)
    else:
        feature_norm = None
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor()
    
    # Initialize label encoder
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
    
    logger.info(f"Model loaded: {config['model_type']} with {model.count_parameters():,} parameters")
    return True


# =============================================================================
# Inference
# =============================================================================

def predict_single(text: str) -> PredictionResponse:
    """Run inference on a single text."""
    import torch
    import torch.nn.functional as F
    
    # Extract features
    features = feature_extractor.extract_batch([text])
    if feature_norm is not None:
        features = (features - feature_norm['mean']) / feature_norm['std']
    features_tensor = torch.tensor(features, dtype=torch.float32)
    
    # Tokenize
    tokens = text.lower().split()[:256]
    ids = [vocab.word2idx.get(t, vocab.word2idx.get('<UNK>', 1)) for t in tokens]
    if len(ids) == 0:
        ids = [1]  # UNK token
    
    # Ensure minimum length for CNN kernels (max kernel size is 5)
    min_length = 6
    while len(ids) < min_length:
        ids.append(0)  # PAD token
    
    token_ids = torch.tensor([ids], dtype=torch.long)
    lengths = [len(ids)]
    
    # Predict
    with torch.no_grad():
        logits = model(token_ids, lengths, features_tensor)
        pred_idx = model.predict(logits).item()
        
        # Get probabilities
        if config.get('use_coral', False):
            # CORAL: convert cumulative probs to class probs
            cumulative_probs = torch.sigmoid(logits)[0]  # (num_classes - 1,)
            # P(class = k) = P(Y > k-1) - P(Y > k)
            # Add boundaries: P(Y > -1) = 1, P(Y > K-1) = 0
            extended = torch.cat([
                torch.ones(1),
                cumulative_probs,
                torch.zeros(1)
            ])
            probs = extended[:-1] - extended[1:]  # Class probabilities
            probs = torch.clamp(probs, min=0.0001)  # Avoid negative/zero probs
            probs = probs / probs.sum()  # Normalize
        else:
            probs = F.softmax(logits, dim=-1)[0]
    
    # Get label
    pred_label = label_encoder.decode(pred_idx)
    confidence = probs[pred_idx].item()
    
    # Class probabilities
    class_probs = {
        label_encoder.decode(i): round(probs[i].item(), 4)
        for i in range(5)
    }
    
    return PredictionResponse(
        text=text[:200] + "..." if len(text) > 200 else text,
        prediction=pred_label,
        confidence=round(confidence, 4),
        class_probabilities=class_probs
    )


# =============================================================================
# API Endpoints
# =============================================================================

@app.on_event("startup")
async def startup():
    """Load model on startup."""
    load_model()


@app.get("/")
async def root():
    """Serve the frontend."""
    frontend_path = Path(__file__).parent / "frontend" / "index.html"
    if frontend_path.exists():
        return FileResponse(frontend_path)
    return {"message": "Legal Text Understandability API", "docs": "/docs"}


@app.get("/api/health")
async def health():
    """Health check."""
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/api/info", response_model=ModelInfo)
async def model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfo(
        model_type=config['model_type'],
        parameters=model.count_parameters(),
        embedding_dim=config.get('embedding_dim', 64),
        use_coral=config.get('use_coral', False),
        labels=label_encoder.labels
    )


@app.post("/api/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict understandability for a single text."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    return predict_single(request.text)


@app.post("/api/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Predict understandability for multiple texts."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(request.texts) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 texts per batch")
    
    predictions = [predict_single(text) for text in request.texts]
    return BatchPredictionResponse(predictions=predictions)


# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
