# Feature Engineering Module for Legal Text Readability
# Extracts readability-specific features from Hungarian legal texts.

import re
import numpy as np
from typing import List, Dict, Any, Optional
from collections import Counter
from utils import setup_logger

logger = setup_logger(__name__)


# =============================================================================
# Hungarian Language Utilities
# =============================================================================

# Common Hungarian stop words (simplified set)
HUNGARIAN_STOP_WORDS = {
    'a', 'az', 'és', 'vagy', 'de', 'hogy', 'is', 'nem', 'ha', 'ki', 'mi',
    'ez', 'az', 'egy', 'van', 'volt', 'lesz', 'meg', 'csak', 'már', 'el',
    'le', 'fel', 'ki', 'be', 'még', 'sem', 'mint', 'úgy', 'igen', 'jó',
    'mely', 'ami', 'aki', 'ahol', 'mert', 'pedig', 'után', 'alatt', 'között'
}

# Hungarian vowels for syllable counting
HUNGARIAN_VOWELS = set('aáeéiíoóöőuúüű')

# Common legal terms (Hungarian)
LEGAL_TERMS = {
    'szerződés', 'jogosult', 'kötelezett', 'felmondás', 'törvény', 'rendelet',
    'hatályos', 'érvényes', 'jogorvoslat', 'bíróság', 'illetékes', 'hatóság',
    'felszólítás', 'teljesítés', 'kártérítés', 'kötbér', 'záradék', 'hatálya',
    'rendelkezés', 'előírás', 'szabályzat', 'feltétel', 'kikötés', 'jogviszony',
    'felelősség', 'képviselő', 'meghatalmazás', 'okirat', 'igazolás', 'nyilatkozat',
    'jogszabály', 'módosítás', 'megszűnés', 'felmondási', 'peres', 'peren',
    'végrehajtás', 'végrehajtó', 'eljárás', 'fellebbezés', 'jogerős', 'határidő',
    'fogyasztó', 'szolgáltató', 'vállalkozás', 'garanciális', 'jótállás',
    'panasz', 'reklamáció', 'elállás', 'visszatérítés', 'kifizetés'
}


def count_syllables_hungarian(word: str) -> int:
    """
    Estimate syllable count for a Hungarian word.
    Hungarian syllables are roughly equal to vowel count.
    """
    word = word.lower()
    return sum(1 for char in word if char in HUNGARIAN_VOWELS)


def count_syllables_text(text: str) -> int:
    """Count total syllables in text."""
    words = re.findall(r'\b\w+\b', text.lower())
    return sum(count_syllables_hungarian(w) for w in words)


# =============================================================================
# Feature Extractors
# =============================================================================

class LexicalFeatures:
    """Features based on word-level statistics."""
    
    @staticmethod
    def extract(text: str) -> Dict[str, float]:
        """Extract lexical features from text."""
        words = re.findall(r'\b\w+\b', text.lower())
        
        if not words:
            return {
                'word_count': 0,
                'avg_word_length': 0,
                'max_word_length': 0,
                'long_word_ratio': 0,  # words > 10 chars
                'very_long_word_ratio': 0,  # words > 15 chars
                'unique_word_ratio': 0,
                'stop_word_ratio': 0,
                'avg_syllables_per_word': 0,
            }
        
        word_lengths = [len(w) for w in words]
        syllable_counts = [count_syllables_hungarian(w) for w in words]
        unique_words = set(words)
        stop_words = [w for w in words if w in HUNGARIAN_STOP_WORDS]
        
        return {
            'word_count': len(words),
            'avg_word_length': np.mean(word_lengths),
            'max_word_length': max(word_lengths),
            'long_word_ratio': sum(1 for l in word_lengths if l > 10) / len(words),
            'very_long_word_ratio': sum(1 for l in word_lengths if l > 15) / len(words),
            'unique_word_ratio': len(unique_words) / len(words),
            'stop_word_ratio': len(stop_words) / len(words),
            'avg_syllables_per_word': np.mean(syllable_counts) if syllable_counts else 0,
        }


class SyntacticFeatures:
    """Features based on sentence and clause structure."""
    
    @staticmethod
    def extract(text: str) -> Dict[str, float]:
        """Extract syntactic features from text."""
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        words = re.findall(r'\b\w+\b', text)
        char_count = len(text)
        
        if not sentences:
            return {
                'sentence_count': 0,
                'avg_sentence_length_words': 0,
                'avg_sentence_length_chars': 0,
                'max_sentence_length': 0,
                'comma_ratio': 0,
                'semicolon_ratio': 0,
                'colon_ratio': 0,
                'parenthesis_ratio': 0,
                'punctuation_density': 0,
            }
        
        # Sentence lengths
        sentence_word_counts = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
        sentence_char_counts = [len(s) for s in sentences]
        
        # Punctuation counts
        comma_count = text.count(',')
        semicolon_count = text.count(';')
        colon_count = text.count(':')
        parenthesis_count = text.count('(') + text.count(')')
        total_punct = sum(1 for c in text if c in '.,;:!?()[]{}"-–')
        
        return {
            'sentence_count': len(sentences),
            'avg_sentence_length_words': np.mean(sentence_word_counts),
            'avg_sentence_length_chars': np.mean(sentence_char_counts),
            'max_sentence_length': max(sentence_word_counts) if sentence_word_counts else 0,
            'comma_ratio': comma_count / len(words) if words else 0,
            'semicolon_ratio': semicolon_count / max(len(sentences), 1),
            'colon_ratio': colon_count / max(len(sentences), 1),
            'parenthesis_ratio': parenthesis_count / max(char_count, 1),
            'punctuation_density': total_punct / max(char_count, 1),
        }


class LegalDomainFeatures:
    """Features specific to legal text domain."""
    
    @staticmethod
    def extract(text: str) -> Dict[str, float]:
        """Extract legal domain-specific features."""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        if not words:
            return {
                'legal_term_ratio': 0,
                'legal_term_count': 0,
                'number_ratio': 0,
                'has_paragraph_ref': 0,
                'has_law_ref': 0,
                'uppercase_ratio': 0,
                'roman_numeral_count': 0,
            }
        
        # Legal term matching (simple stem matching)
        legal_matches = 0
        for word in words:
            # Check if word starts with any legal term stem
            for term in LEGAL_TERMS:
                if word.startswith(term[:min(6, len(term))]):
                    legal_matches += 1
                    break
        
        # Numbers and references
        numbers = re.findall(r'\d+', text)
        paragraph_refs = re.findall(r'\d+\.\s*§|\d+\.§|§\s*\d+', text)
        law_refs = re.findall(r'\d{4}\.\s*évi\s+[IVXLCDM]+\.?\s*törvény', text, re.IGNORECASE)
        
        # Roman numerals (common in legal docs)
        roman_numerals = re.findall(r'\b[IVXLCDM]+\b', text)
        
        # Case analysis
        uppercase_chars = sum(1 for c in text if c.isupper())
        alpha_chars = sum(1 for c in text if c.isalpha())
        
        return {
            'legal_term_ratio': legal_matches / len(words),
            'legal_term_count': legal_matches,
            'number_ratio': len(numbers) / len(words),
            'has_paragraph_ref': 1 if paragraph_refs else 0,
            'has_law_ref': 1 if law_refs else 0,
            'uppercase_ratio': uppercase_chars / max(alpha_chars, 1),
            'roman_numeral_count': len(roman_numerals),
        }


class ReadabilityIndices:
    """Readability formulas adapted for Hungarian."""
    
    @staticmethod
    def extract(text: str) -> Dict[str, float]:
        """Calculate readability indices."""
        words = re.findall(r'\b\w+\b', text)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        word_count = len(words)
        sentence_count = max(len(sentences), 1)
        syllable_count = count_syllables_text(text)
        char_count = len(text)
        
        if word_count == 0:
            return {
                'flesch_reading_ease': 0,
                'automated_readability_index': 0,
                'coleman_liau_index': 0,
                'words_per_sentence': 0,
                'syllables_per_word': 0,
                'chars_per_word': 0,
            }
        
        # Basic ratios
        words_per_sentence = word_count / sentence_count
        syllables_per_word = syllable_count / word_count
        chars_per_word = sum(len(w) for w in words) / word_count
        
        # Flesch Reading Ease (adapted)
        # Original: 206.835 - 1.015*(words/sentences) - 84.6*(syllables/words)
        flesch = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
        flesch = max(0, min(100, flesch))  # Clamp to 0-100
        
        # Automated Readability Index
        # 4.71*(chars/words) + 0.5*(words/sentences) - 21.43
        ari = 4.71 * chars_per_word + 0.5 * words_per_sentence - 21.43
        
        # Coleman-Liau Index
        # 0.0588*L - 0.296*S - 15.8 where L=letters per 100 words, S=sentences per 100 words
        letters_per_100 = (sum(len(w) for w in words) / word_count) * 100
        sentences_per_100 = (sentence_count / word_count) * 100
        cli = 0.0588 * letters_per_100 - 0.296 * sentences_per_100 - 15.8
        
        return {
            'flesch_reading_ease': flesch,
            'automated_readability_index': ari,
            'coleman_liau_index': cli,
            'words_per_sentence': words_per_sentence,
            'syllables_per_word': syllables_per_word,
            'chars_per_word': chars_per_word,
        }


class StructuralFeatures:
    """Features about text structure and formatting."""
    
    @staticmethod
    def extract(text: str) -> Dict[str, float]:
        """Extract structural features."""
        char_count = max(len(text), 1)
        lines = text.split('\n')
        
        # List patterns (bullets, numbering)
        list_items = re.findall(r'^\s*[-•*]\s+|\d+[.)]\s+|[a-z][.)]\s+', text, re.MULTILINE)
        
        # Quoted text
        quoted_matches = re.findall(r'[„""\'](.*?)["""\']', text)
        quoted_chars = sum(len(m) for m in quoted_matches)
        
        return {
            'char_count': len(text),
            'line_count': len(lines),
            'has_list_items': 1 if list_items else 0,
            'list_item_count': len(list_items),
            'quoted_text_ratio': quoted_chars / char_count,
            'whitespace_ratio': sum(1 for c in text if c.isspace()) / char_count,
            'digit_ratio': sum(1 for c in text if c.isdigit()) / char_count,
        }


# =============================================================================
# Main Feature Extractor
# =============================================================================

class FeatureExtractor:
    """
    Main class for extracting all features from text.
    
    Usage:
        extractor = FeatureExtractor()
        features = extractor.extract(text)  # Returns dict
        feature_vector = extractor.extract_vector(text)  # Returns numpy array
    """
    
    def __init__(self):
        self.feature_names = None
        self._initialized = False
    
    def extract(self, text: str) -> Dict[str, float]:
        """
        Extract all features from a single text.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary of feature name -> value
        """
        features = {}
        
        # Extract all feature groups
        features.update(LexicalFeatures.extract(text))
        features.update(SyntacticFeatures.extract(text))
        features.update(LegalDomainFeatures.extract(text))
        features.update(ReadabilityIndices.extract(text))
        features.update(StructuralFeatures.extract(text))
        
        # Store feature names on first call
        if not self._initialized:
            self.feature_names = list(features.keys())
            self._initialized = True
        
        return features
    
    def extract_vector(self, text: str) -> np.ndarray:
        """Extract features as a numpy vector."""
        features = self.extract(text)
        return np.array([features[name] for name in self.feature_names])
    
    def extract_batch(self, texts: List[str]) -> np.ndarray:
        """
        Extract features for a batch of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            2D numpy array of shape (n_texts, n_features)
        """
        vectors = []
        for i, text in enumerate(texts):
            vec = self.extract_vector(text)
            vectors.append(vec)
            if (i + 1) % 500 == 0:
                logger.info(f"  Extracted features for {i+1}/{len(texts)} texts")
        
        return np.array(vectors)
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names (must call extract first)."""
        if not self._initialized:
            # Initialize by extracting from dummy text
            self.extract("dummy text for initialization")
        return self.feature_names.copy()


def extract_features_from_dataset(
    documents: List[Dict[str, Any]],
    text_key: str = 'content'
) -> np.ndarray:
    """
    Convenience function to extract features from a list of documents.
    
    Args:
        documents: List of document dictionaries
        text_key: Key for text content in each document
        
    Returns:
        Feature matrix of shape (n_docs, n_features)
    """
    extractor = FeatureExtractor()
    texts = [doc[text_key] for doc in documents]
    
    logger.info(f"Extracting features from {len(texts)} documents...")
    features = extractor.extract_batch(texts)
    logger.info(f"  Extracted {features.shape[1]} features")
    
    return features, extractor.get_feature_names()


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    # Test with sample Hungarian legal text
    sample_text = """
    81. Az Alza főszabály szerint nem köt folyamatos teljesítésű szerződéseket. 
    Amennyiben az Alza folyamatos teljesítésű szerződést kötne, úgy a Fogyasztó 
    megfelelő információt kap a szerződés legrövidebb időtartamáról, valamint 
    a szerződés megszüntetésének feltételeiről.
    """
    
    extractor = FeatureExtractor()
    features = extractor.extract(sample_text)
    
    logger.info("Sample feature extraction:")
    for name, value in sorted(features.items()):
        logger.info(f"  {name}: {value:.4f}")
    
    logger.info(f"\nTotal features: {len(features)}")
