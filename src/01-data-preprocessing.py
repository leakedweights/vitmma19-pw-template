# Data preprocessing script
# This script handles data loading, cleaning, and transformation.
import os
import zipfile
import json
import re
from pathlib import Path
from collections import Counter
import requests
from sklearn.model_selection import train_test_split
from utils import setup_logger

logger = setup_logger()

# Constants
DATA_DIR = Path("./data")
DATASET_DIR = DATA_DIR / "legaltextdecoder"
DATASET_URL = "https://bmeedu-my.sharepoint.com/:u:/g/personal/gyires-toth_balint_vik_bme_hu/IQDYwXUJcB_jQYr0bDfNT5RKARYgfKoH97zho3rxZ46KA1I?e=iFp3iz&download=1"
DATASET_ZIP = DATA_DIR / "legaltextdecoder.zip"


def ensure_data_directory():
    """Ensure the data directory exists."""
    if not DATA_DIR.exists():
        logger.info(f"Creating data directory: {DATA_DIR}")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    else:
        logger.info(f"Data directory already exists: {DATA_DIR}")


def download_dataset():
    """Download the dataset from SharePoint."""
    logger.info(f"Downloading dataset from {DATASET_URL}")
    
    try:
        response = requests.get(DATASET_URL, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0
        
        with open(DATASET_ZIP, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        logger.info(f"Download progress: {progress:.1f}%")
        
        logger.info(f"Dataset downloaded successfully to {DATASET_ZIP}")
        return True
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        return False


def extract_dataset():
    """Extract the downloaded zip file."""
    logger.info(f"Extracting dataset to {DATASET_DIR}")
    
    try:
        with zipfile.ZipFile(DATASET_ZIP, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        
        logger.info(f"Dataset extracted successfully to {DATASET_DIR}")
        
        # Clean up the zip file
        DATASET_ZIP.unlink()
        logger.info(f"Removed zip file: {DATASET_ZIP}")
        return True
    except Exception as e:
        logger.error(f"Failed to extract dataset: {e}")
        return False


def list_text_files():
    """List all .txt files in the dataset subdirectories."""
    logger.info(f"Listing all .txt files in {DATASET_DIR}")
    
    txt_files = []
    
    if not DATASET_DIR.exists():
        logger.error(f"Dataset directory does not exist: {DATASET_DIR}")
        return txt_files
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(DATASET_DIR):
        for file in files:
            if file.endswith('.txt'):
                abs_path = Path(root) / file
                txt_files.append(abs_path.resolve())
    
    logger.info(f"Found {len(txt_files)} .txt files")
    
    # Log the first few files as examples
    for i, file_path in enumerate(txt_files[:5]):
        logger.info(f"  Example {i+1}: {file_path}")
    
    if len(txt_files) > 5:
        logger.info(f"  ... and {len(txt_files) - 5} more files")
    
    return txt_files


def merge_documents(txt_files, labels=None, output_file="data/dataset.json"):
    """
    Merge all documents from text files into a single JSON file.
    Each line in each text file becomes a separate document entry.
    If labels are provided, they will be merged into the documents.
    
    Args:
        txt_files: List of paths to text files
        labels: Optional list of label dictionaries to merge with documents
        output_file: Path to output JSON file
    
    Returns:
        List of document dictionaries
    """
    logger.info(f"Merging documents from {len(txt_files)} files...")
    
    documents = []
    doc_id = 0
    
    for file_path in txt_files:
        # Get subdirectory name (parent directory of the file)
        subdirectory = file_path.parent.name
        
        # Skip the consensus directory
        if subdirectory == "consensus":
            logger.info(f"Skipping consensus directory file: {file_path.name}")
            continue
        
        # Get filename without extension
        filename = file_path.stem
        
        logger.info(f"Processing {subdirectory}/{filename}.txt")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Process each line as a separate document
            for line_num, line in enumerate(lines, start=1):
                # Strip whitespace and skip empty lines
                content = line.strip()
                if not content:
                    continue
                
                # Create document entry
                doc_entry = {
                    "id": doc_id,
                    "subdirectory": subdirectory,
                    "filename": filename,
                    "line_number": line_num,
                    "content": content,
                    "label": None,  # Will be filled if labels are provided
                    "label_metadata": None  # Additional label info
                }
                
                documents.append(doc_entry)
                doc_id += 1
        
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue
    
    logger.info(f"Merged {len(documents)} documents from {len([f for f in txt_files if f.parent.name != 'consensus'])} files")
    
    # If labels are provided, merge them into documents
    if labels:
        logger.info(f"Merging {len(labels)} labels into documents...")
        documents = _merge_labels_into_documents(documents, labels)
    
    # Save to JSON file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Successfully saved merged documents to {output_path}")
        logger.info(f"Total documents: {len(documents)}")
        
        # Log some statistics
        subdirs = set(doc['subdirectory'] for doc in documents)
        logger.info(f"Documents from {len(subdirs)} subdirectories")
        
        if labels:
            labeled_count = sum(1 for doc in documents if doc['label'] is not None)
            logger.info(f"Labeled documents: {labeled_count} ({labeled_count/len(documents)*100:.2f}%)")
        
    except Exception as e:
        logger.error(f"Error saving merged documents: {e}")
        return []
    
    return documents


def _merge_labels_into_documents(documents, labels):
    """
    Helper function to merge label information into document entries.
    
    Args:
        documents: List of document dictionaries
        labels: List of label dictionaries
    
    Returns:
        Updated list of documents with label information
    """
    # Create a lookup dictionary for documents by content
    doc_by_content = {}
    for doc in documents:
        content = doc['content'].strip()
        if content not in doc_by_content:
            doc_by_content[content] = []
        doc_by_content[content].append(doc)
    
    label_stats = {
        'matched_labels': 0,
        'unmatched_labels': 0,
        'label_distribution': {}
    }
    
    # Process each label and try to match with documents
    for label_item in labels:
        text_content = label_item.get('data', {}).get('text', '').strip()
        
        # Try to find matching document(s)
        matching_docs = doc_by_content.get(text_content, [])
        
        if matching_docs:
            label_stats['matched_labels'] += 1
            
            # Extract the annotation/label value
            annotations = label_item.get('annotations', [])
            label_value = None
            
            if annotations and len(annotations) > 0:
                result = annotations[0].get('result', [])
                if result and len(result) > 0:
                    choices = result[0].get('value', {}).get('choices', [])
                    if choices:
                        label_value = choices[0]
                        
                        # Track label distribution
                        if label_value not in label_stats['label_distribution']:
                            label_stats['label_distribution'][label_value] = 0
                        label_stats['label_distribution'][label_value] += 1
            
            # Add label info to each matching document
            for doc in matching_docs:
                doc['label'] = label_value
                doc['label_metadata'] = {
                    'label_id': label_item.get('id'),
                    'file_upload': label_item.get('file_upload'),
                    'source_subdirectory': label_item.get('source_subdirectory'),
                    'source_file': label_item.get('source_file'),
                    'created_at': label_item.get('created_at'),
                    'updated_at': label_item.get('updated_at')
                }
        else:
            label_stats['unmatched_labels'] += 1
    
    logger.info("Label merging statistics:")
    logger.info(f"  Matched labels: {label_stats['matched_labels']}")
    logger.info(f"  Unmatched labels: {label_stats['unmatched_labels']}")
    logger.info(f"  Label distribution: {label_stats['label_distribution']}")
    
    return documents


def find_consensus_files():
    """Find all consensus JSON files."""
    logger.info("Finding consensus label files...")
    
    consensus_dir = DATASET_DIR / "consensus"
    consensus_files = []
    
    if not consensus_dir.exists():
        logger.warning(f"Consensus directory does not exist: {consensus_dir}")
        return consensus_files
    
    for file_path in consensus_dir.glob("*.json"):
        if file_path.name != "legaltextdecoder-consensus.txt":  # Skip non-JSON files
            consensus_files.append(file_path)
    
    logger.info(f"Found {len(consensus_files)} consensus files")
    
    return consensus_files


def impute_labels_from_consensus(documents, consensus_files):
    """
    Impute missing labels using consensus data.
    
    Args:
        documents: List of document dictionaries
        consensus_files: List of paths to consensus JSON files
    
    Returns:
        Updated list of documents with imputed labels
    """
    logger.info("Starting label imputation from consensus data...")
    
    # Load all consensus labels
    consensus_labels = []
    for file_path in consensus_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                labels = json.load(f)
                for label in labels:
                    label['consensus_source_file'] = file_path.name
                consensus_labels.extend(labels)
        except Exception as e:
            logger.error(f"Error loading consensus file {file_path}: {e}")
    
    logger.info(f"Loaded {len(consensus_labels)} consensus labels")
    
    # Create lookup by content for consensus labels
    consensus_by_content = {}
    for label in consensus_labels:
        text_content = label.get('data', {}).get('text', '').strip()
        if text_content:
            if text_content not in consensus_by_content:
                consensus_by_content[text_content] = []
            consensus_by_content[text_content].append(label)
    
    # Track imputation statistics
    imputation_stats = {
        'total_unlabeled_before': 0,
        'imputed_count': 0,
        'still_unlabeled': 0,
        'already_labeled': 0,
        'imputed_by_subdirectory': {},
        'imputed_label_distribution': {}
    }
    
    # Count unlabeled documents before imputation
    for doc in documents:
        if doc['label'] is None:
            imputation_stats['total_unlabeled_before'] += 1
        else:
            imputation_stats['already_labeled'] += 1
    
    logger.info("Documents before imputation:")
    logger.info(f"  Already labeled: {imputation_stats['already_labeled']}")
    logger.info(f"  Unlabeled: {imputation_stats['total_unlabeled_before']}")
    
    # Impute labels for unlabeled documents
    for doc in documents:
        # Skip if already labeled
        if doc['label'] is not None:
            continue
        
        content = doc['content'].strip()
        consensus_matches = consensus_by_content.get(content, [])
        
        if consensus_matches:
            # Use the first consensus match
            consensus_label = consensus_matches[0]
            
            # Extract label value
            annotations = consensus_label.get('annotations', [])
            label_value = None
            
            if annotations and len(annotations) > 0:
                result = annotations[0].get('result', [])
                if result and len(result) > 0:
                    choices = result[0].get('value', {}).get('choices', [])
                    if choices:
                        label_value = choices[0]
            
            if label_value:
                # Impute the label
                doc['label'] = label_value
                doc['label_metadata'] = {
                    'label_id': consensus_label.get('id'),
                    'file_upload': consensus_label.get('file_upload'),
                    'source_subdirectory': 'consensus',
                    'source_file': consensus_label.get('consensus_source_file'),
                    'created_at': consensus_label.get('created_at'),
                    'updated_at': consensus_label.get('updated_at'),
                    'imputed': True  # Mark as imputed
                }
                
                # Update statistics
                imputation_stats['imputed_count'] += 1
                
                # Track by subdirectory
                subdir = doc['subdirectory']
                if subdir not in imputation_stats['imputed_by_subdirectory']:
                    imputation_stats['imputed_by_subdirectory'][subdir] = 0
                imputation_stats['imputed_by_subdirectory'][subdir] += 1
                
                # Track label distribution
                if label_value not in imputation_stats['imputed_label_distribution']:
                    imputation_stats['imputed_label_distribution'][label_value] = 0
                imputation_stats['imputed_label_distribution'][label_value] += 1
                
                logger.debug(f"Imputed label '{label_value}' for doc {doc['id']} from {doc['subdirectory']}/{doc['filename']}")
    
    # Count still unlabeled documents
    imputation_stats['still_unlabeled'] = sum(1 for doc in documents if doc['label'] is None)
    
    # Log detailed statistics
    logger.info("Label imputation complete!")
    logger.info(f"  Total unlabeled before: {imputation_stats['total_unlabeled_before']}")
    logger.info(f"  Successfully imputed: {imputation_stats['imputed_count']}")
    logger.info(f"  Still unlabeled: {imputation_stats['still_unlabeled']}")
    logger.info(f"  Imputation rate: {imputation_stats['imputed_count'] / imputation_stats['total_unlabeled_before'] * 100:.2f}%" if imputation_stats['total_unlabeled_before'] > 0 else "  Imputation rate: N/A")
    
    if imputation_stats['imputed_by_subdirectory']:
        logger.info("  Imputed labels by subdirectory:")
        for subdir, count in sorted(imputation_stats['imputed_by_subdirectory'].items(), key=lambda x: x[1], reverse=True):
            logger.info(f"    {subdir}: {count}")
    
    if imputation_stats['imputed_label_distribution']:
        logger.info("  Imputed label distribution:")
        for label, count in sorted(imputation_stats['imputed_label_distribution'].items()):
            logger.info(f"    {label}: {count}")
    
    return documents


def find_label_files():
    """Find all label JSON files in subdirectories (excluding consensus)."""
    logger.info(f"Finding label JSON files in {DATASET_DIR}")
    
    label_files = []
    
    if not DATASET_DIR.exists():
        logger.error(f"Dataset directory does not exist: {DATASET_DIR}")
        return label_files
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(DATASET_DIR):
        # Skip consensus directory
        if "consensus" in Path(root).parts:
            continue
            
        for file in files:
            if file.endswith('.json'):
                abs_path = Path(root) / file
                label_files.append(abs_path.resolve())
    
    logger.info(f"Found {len(label_files)} label JSON files")
    
    return label_files


def merge_labels(label_files, output_file="data/labels.json"):
    """
    Merge all label JSON files into a single file.
    
    Args:
        label_files: List of paths to label JSON files
        output_file: Path to output JSON file
    
    Returns:
        List of all labeled items
    """
    logger.info(f"Merging labels from {len(label_files)} files...")
    
    all_labels = []
    
    for file_path in label_files:
        subdirectory = file_path.parent.name
        
        logger.info(f"Processing labels from {subdirectory}/{file_path.name}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                labels = json.load(f)
            
            # Add subdirectory and source file info to each label
            for label in labels:
                label['source_subdirectory'] = subdirectory
                label['source_file'] = file_path.name
                all_labels.append(label)
        
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue
    
    logger.info(f"Merged {len(all_labels)} labeled items from {len(label_files)} files")
    
    # Save to JSON file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_labels, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Successfully saved merged labels to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving merged labels: {e}")
        return []
    
    return all_labels


def create_label_mapping(documents, labels, output_file="data/labeled_dataset.json"):
    """
    Create a mapping between documents and their labels.
    
    Args:
        documents: List of document dictionaries from merge_documents
        labels: List of label dictionaries from merge_labels
        output_file: Path to output JSON file
    
    Returns:
        Dictionary with labeled and unlabeled documents
    """
    logger.info("Creating mapping between documents and labels...")
    
    # Create a lookup dictionary for documents by content
    doc_by_content = {}
    for doc in documents:
        # Use content as key for matching
        content = doc['content'].strip()
        if content not in doc_by_content:
            doc_by_content[content] = []
        doc_by_content[content].append(doc)
    
    labeled_documents = []
    label_stats = {
        'total_labels': len(labels),
        'matched_labels': 0,
        'unmatched_labels': 0,
        'label_distribution': {}
    }
    
    # Process each label and try to match with documents
    for label_item in labels:
        text_content = label_item.get('data', {}).get('text', '').strip()
        
        # Try to find matching document(s)
        matching_docs = doc_by_content.get(text_content, [])
        
        if matching_docs:
            label_stats['matched_labels'] += 1
            
            # Extract the annotation/label value
            annotations = label_item.get('annotations', [])
            label_value = None
            
            if annotations and len(annotations) > 0:
                result = annotations[0].get('result', [])
                if result and len(result) > 0:
                    choices = result[0].get('value', {}).get('choices', [])
                    if choices:
                        label_value = choices[0]
                        
                        # Track label distribution
                        if label_value not in label_stats['label_distribution']:
                            label_stats['label_distribution'][label_value] = 0
                        label_stats['label_distribution'][label_value] += 1
            
            # Add label info to each matching document
            for doc in matching_docs:
                labeled_doc = doc.copy()
                labeled_doc['label'] = label_value
                labeled_doc['label_id'] = label_item.get('id')
                labeled_doc['file_upload'] = label_item.get('file_upload')
                labeled_doc['label_source_subdirectory'] = label_item.get('source_subdirectory')
                labeled_doc['label_source_file'] = label_item.get('source_file')
                labeled_documents.append(labeled_doc)
        else:
            label_stats['unmatched_labels'] += 1
            logger.debug(f"No matching document found for label {label_item.get('id')}: {text_content[:50]}...")
    
    # Find unlabeled documents
    labeled_contents = set(doc['content'] for doc in labeled_documents)
    unlabeled_documents = [doc for doc in documents if doc['content'] not in labeled_contents]
    
    result = {
        'labeled_documents': labeled_documents,
        'unlabeled_documents': unlabeled_documents,
        'statistics': {
            **label_stats,
            'total_documents': len(documents),
            'labeled_count': len(labeled_documents),
            'unlabeled_count': len(unlabeled_documents),
            'labeling_coverage': f"{len(labeled_documents) / len(documents) * 100:.2f}%" if documents else "0%"
        }
    }
    
    logger.info("Mapping complete:")
    logger.info(f"  Total documents: {len(documents)}")
    logger.info(f"  Labeled documents: {len(labeled_documents)}")
    logger.info(f"  Unlabeled documents: {len(unlabeled_documents)}")
    logger.info(f"  Coverage: {result['statistics']['labeling_coverage']}")
    logger.info(f"  Matched labels: {label_stats['matched_labels']}")
    logger.info(f"  Unmatched labels: {label_stats['unmatched_labels']}")
    logger.info(f"  Label distribution: {label_stats['label_distribution']}")
    
    # Save to JSON file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Successfully saved labeled dataset to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving labeled dataset: {e}")
    
    return result


def clean_text(text):
    """
    Clean and normalize Hungarian text.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Remove zero-width characters and other invisible characters
    text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    # Normalize dashes
    text = text.replace('–', '-').replace('—', '-')
    
    # Remove multiple spaces again after replacements
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def check_text_quality(text, min_length=10, max_length=10000):
    """
    Check if text meets quality criteria.
    
    Args:
        text: Text to check
        min_length: Minimum character length
        max_length: Maximum character length
        
    Returns:
        Tuple of (is_valid, reason)
    """
    if not text or len(text.strip()) == 0:
        return False, "empty_text"
    
    if len(text) < min_length:
        return False, "too_short"
    
    if len(text) > max_length:
        return False, "too_long"
    
    # Check if text is mostly non-alphabetic (likely corrupted)
    alpha_ratio = sum(c.isalpha() for c in text) / len(text)
    if alpha_ratio < 0.5:
        return False, "low_alpha_ratio"
    
    return True, "valid"


def find_and_remove_duplicates(documents):
    """
    Find and remove duplicate documents.
    
    Args:
        documents: List of document dictionaries
        
    Returns:
        Tuple of (cleaned_documents, duplicate_count)
    """
    logger.info("Checking for duplicates...")
    
    seen_content = set()
    cleaned_docs = []
    duplicate_count = 0
    
    for doc in documents:
        content = doc['content']
        
        if content in seen_content:
            duplicate_count += 1
            continue
        
        seen_content.add(content)
        cleaned_docs.append(doc)
    
    logger.info(f"  Found and removed {duplicate_count} duplicate documents")
    
    return cleaned_docs, duplicate_count


def clean_and_validate_documents(documents):
    """
    Clean text and validate document quality.
    
    Args:
        documents: List of document dictionaries
        
    Returns:
        Tuple of (cleaned_documents, statistics)
    """
    logger.info("Cleaning and validating documents...")
    
    stats = {
        'original_count': len(documents),
        'removed_empty_text': 0,
        'removed_too_short': 0,
        'removed_too_long': 0,
        'removed_low_alpha_ratio': 0,
        'removed_duplicates': 0,
        'cleaned_count': 0
    }
    
    cleaned_docs = []
    
    for doc in documents:
        # Clean the text
        original_content = doc['content']
        cleaned_content = clean_text(original_content)
        
        # Check quality
        is_valid, reason = check_text_quality(cleaned_content)
        
        if not is_valid:
            stats[f'removed_{reason}'] += 1
            continue
        
        # Update document with cleaned content
        doc['content'] = cleaned_content
        doc['original_content'] = original_content
        doc['content_length'] = len(cleaned_content)
        doc['word_count'] = len(cleaned_content.split())
        
        cleaned_docs.append(doc)
    
    # Remove duplicates
    cleaned_docs, duplicate_count = find_and_remove_duplicates(cleaned_docs)
    stats['removed_duplicates'] = duplicate_count
    stats['cleaned_count'] = len(cleaned_docs)
    
    return cleaned_docs, stats


def split_dataset(documents, test_size=0.15, val_size=0.15, random_state=42):
    """
    Split dataset into train/validation/test sets with stratification.
    
    Args:
        documents: List of cleaned documents
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with train, val, test, and unlabeled splits
    """
    logger.info("Splitting dataset into train/val/test...")
    
    # Separate labeled and unlabeled documents
    labeled_docs = [doc for doc in documents if doc['label'] is not None]
    unlabeled_docs = [doc for doc in documents if doc['label'] is None]
    
    if len(labeled_docs) == 0:
        raise ValueError("No labeled documents found!")
    
    # Extract labels for stratification
    labels = [doc['label'] for doc in labeled_docs]
    
    # First split: separate test set
    train_val_docs, test_docs, train_val_labels, test_labels = train_test_split(
        labeled_docs,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )
    
    # Second split: separate validation from training
    val_size_adjusted = val_size / (1 - test_size)
    train_docs, val_docs, train_labels, val_labels = train_test_split(
        train_val_docs,
        train_val_labels,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=train_val_labels
    )
    
    return {
        'train': train_docs,
        'val': val_docs,
        'test': test_docs,
        'unlabeled': unlabeled_docs,
        'train_labels': train_labels,
        'val_labels': val_labels,
        'test_labels': test_labels
    }


def save_splits(splits, output_dir="data/processed"):
    """
    Save train/val/test splits to separate JSON files.
    
    Args:
        splits: Dictionary with train/val/test/unlabeled splits
        output_dir: Directory to save files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for split_name in ['train', 'val', 'test', 'unlabeled']:
        split_docs = splits[split_name]
        output_file = output_path / f"{split_name}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(split_docs, f, ensure_ascii=False, indent=2)
        
        logger.info(f"  Saved {len(split_docs)} documents to {output_file}")
    
    # Also save combined cleaned dataset
    all_labeled = splits['train'] + splits['val'] + splits['test']
    combined_file = output_path / "dataset_cleaned.json"
    
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(all_labeled, f, ensure_ascii=False, indent=2)
    
    logger.info(f"  Saved {len(all_labeled)} labeled documents to {combined_file}")


def preprocess():
    """Main preprocessing function."""
    logger.info("=" * 80)
    logger.info("STARTING DATA PREPROCESSING PIPELINE")
    logger.info("=" * 80)
    
    # Step 1: Ensure data directory exists
    ensure_data_directory()
    
    # Step 2: Check if preprocessed dataset exists
    if DATASET_DIR.exists() and any(DATASET_DIR.iterdir()):
        logger.info(f"Preprocessed dataset already exists at {DATASET_DIR}")
    else:
        logger.info("Preprocessed dataset not found. Starting download and extraction...")
        
        # Step 3: Download the dataset
        if not download_dataset():
            logger.error("Failed to download dataset. Exiting.")
            return
        
        # Step 4: Extract the dataset
        if not extract_dataset():
            logger.error("Failed to extract dataset. Exiting.")
            return
    
    # Step 5: List all .txt files
    txt_files = list_text_files()
    
    # Step 6: Find and load label files
    label_files = find_label_files()
    labels = []
    
    if label_files:
        logger.info(f"Loading labels from {len(label_files)} files...")
        for file_path in label_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_labels = json.load(f)
                    # Add source metadata to each label
                    for label in file_labels:
                        label['source_subdirectory'] = file_path.parent.name
                        label['source_file'] = file_path.name
                    labels.extend(file_labels)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        logger.info(f"Loaded {len(labels)} labels total")
    
    # Step 7: Merge documents with labels into a single unified dataset
    documents = merge_documents(txt_files, labels=labels if labels else None)
    
    # Step 8: Impute missing labels from consensus data
    consensus_files = find_consensus_files()
    if consensus_files:
        documents = impute_labels_from_consensus(documents, consensus_files)
        # Re-save the dataset with imputed labels
        output_path = Path("data/dataset.json")
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(documents, f, ensure_ascii=False, indent=2)
            logger.info(f"Updated dataset saved with imputed labels to {output_path}")
        except Exception as e:
            logger.error(f"Error saving updated dataset: {e}")
    
    # Step 9: Clean and validate documents
    logger.info("\n" + "=" * 80)
    logger.info("CLEANING AND VALIDATION")
    logger.info("=" * 80)
    cleaned_docs, cleaning_stats = clean_and_validate_documents(documents)
    
    logger.info(f"  Original documents: {cleaning_stats['original_count']}")
    logger.info(f"  Removed (empty): {cleaning_stats['removed_empty_text']}")
    logger.info(f"  Removed (too short): {cleaning_stats['removed_too_short']}")
    logger.info(f"  Removed (too long): {cleaning_stats['removed_too_long']}")
    logger.info(f"  Removed (low quality): {cleaning_stats['removed_low_alpha_ratio']}")
    logger.info(f"  Removed (duplicates): {cleaning_stats['removed_duplicates']}")
    logger.info(f"  ✓ Final cleaned documents: {cleaning_stats['cleaned_count']}")
    logger.info(f"  ✓ Retention rate: {cleaning_stats['cleaned_count']/cleaning_stats['original_count']*100:.2f}%")
    
    # Step 10: Split dataset
    logger.info("\n" + "=" * 80)
    logger.info("DATASET SPLITTING")
    logger.info("=" * 80)
    splits = split_dataset(cleaned_docs)
    
    labeled_count = len(splits['train']) + len(splits['val']) + len(splits['test'])
    logger.info(f"  Total labeled documents: {labeled_count}")
    logger.info(f"  ✓ Training set: {len(splits['train'])} ({len(splits['train'])/labeled_count*100:.1f}%)")
    logger.info(f"  ✓ Validation set: {len(splits['val'])} ({len(splits['val'])/labeled_count*100:.1f}%)")
    logger.info(f"  ✓ Test set: {len(splits['test'])} ({len(splits['test'])/labeled_count*100:.1f}%)")
    logger.info(f"  Unlabeled documents: {len(splits['unlabeled'])}")
    
    # Log label distribution
    logger.info("\n  Label Distribution Across Splits:")
    label_counts_train = Counter(splits['train_labels'])
    label_counts_val = Counter(splits['val_labels'])
    label_counts_test = Counter(splits['test_labels'])
    
    all_labels = sorted(set(splits['train_labels'] + splits['val_labels'] + splits['test_labels']))
    logger.info(f"  {'Label':<30} {'Train':>8} {'Val':>8} {'Test':>8}")
    logger.info(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8}")
    for label in all_labels:
        train_pct = label_counts_train.get(label, 0) / len(splits['train']) * 100
        val_pct = label_counts_val.get(label, 0) / len(splits['val']) * 100
        test_pct = label_counts_test.get(label, 0) / len(splits['test']) * 100
        logger.info(f"  {label:<30} {train_pct:>7.1f}% {val_pct:>7.1f}% {test_pct:>7.1f}%")
    
    # Step 11: Save splits
    logger.info("\n" + "=" * 80)
    logger.info("SAVING PROCESSED DATA")
    logger.info("=" * 80)
    save_splits(splits)
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("PREPROCESSING COMPLETE - SUMMARY")
    logger.info("=" * 80)
    logger.info(f"  ✓ Downloaded and extracted raw data")
    logger.info(f"  ✓ Merged {len(txt_files)} text files")
    logger.info(f"  ✓ Applied {len(labels)} labels")
    logger.info(f"  ✓ Imputed labels from consensus data")
    logger.info(f"  ✓ Cleaned and validated {cleaning_stats['cleaned_count']} documents")
    logger.info(f"  ✓ Split into train ({len(splits['train'])}), val ({len(splits['val'])}), test ({len(splits['test'])})")
    logger.info(f"  ✓ Saved to data/processed/")
    logger.info("=" * 80)
    logger.info("Ready for feature engineering and model training!")
    logger.info("=" * 80)
    
    return cleaned_docs


if __name__ == "__main__":
    preprocess()
