"""
Hateful Memes Dataset for ViLBERT/VisualBERT

Dataset loader for the Facebook Hateful Memes Challenge.
Supports:
- Loading from HuggingFace or local JSONL files
- On-the-fly feature extraction or pre-extracted features
- Data augmentation (optional)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from PIL import Image
import numpy as np
import json
import os
from typing import Dict, List, Optional, Callable, Union
from pathlib import Path
import h5py

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class HatefulMemesDataset(Dataset):
    """
    PyTorch Dataset for the Hateful Memes Challenge.

    Supports two modes:
    1. Pre-extracted features (faster, recommended for training)
    2. On-the-fly feature extraction (slower, but more flexible)
    """

    def __init__(
        self,
        data_path: str,
        image_dir: str,
        feature_dir: Optional[str] = None,
        tokenizer: Optional[BertTokenizer] = None,
        max_seq_length: int = 128,
        max_regions: int = 36,
        feature_extractor: Optional[Callable] = None,
        split: str = 'train',
        use_huggingface: bool = False,
        transform: Optional[Callable] = None
    ):
        """
        Initialize the dataset.

        Args:
            data_path: Path to JSONL annotation file or HuggingFace dataset name
            image_dir: Directory containing meme images
            feature_dir: Directory containing pre-extracted visual features (optional)
            tokenizer: BERT tokenizer (created if not provided)
            max_seq_length: Maximum sequence length for text
            max_regions: Maximum number of visual regions
            feature_extractor: Feature extraction callable (for on-the-fly)
            split: Data split ('train', 'dev', 'test')
            use_huggingface: Whether to load from HuggingFace
            transform: Optional image transforms
        """
        self.image_dir = image_dir
        self.feature_dir = feature_dir
        self.max_seq_length = max_seq_length
        self.max_regions = max_regions
        self.feature_extractor = feature_extractor
        self.split = split
        self.transform = transform

        # Initialize tokenizer
        self.tokenizer = tokenizer or BertTokenizer.from_pretrained('bert-base-uncased')

        # Load annotations
        if use_huggingface and HF_AVAILABLE:
            self._load_from_huggingface(data_path, split)
        else:
            self._load_from_jsonl(data_path)

        # Load feature index if using pre-extracted features
        self.feature_index = {}
        if feature_dir and os.path.exists(feature_dir):
            self._build_feature_index()

    def _load_from_jsonl(self, data_path: str):
        """Load annotations from JSONL file."""
        self.samples = []
        with open(data_path, 'r') as f:
            for line in f:
                sample = json.loads(line.strip())
                self.samples.append(sample)
        print(f"Loaded {len(self.samples)} samples from {data_path}")

    def _load_from_huggingface(self, dataset_name: str, split: str):
        """Load from HuggingFace datasets."""
        dataset = load_dataset(dataset_name, split=split)
        self.samples = []
        for item in dataset:
            self.samples.append({
                'id': item.get('id', item.get('idx')),
                'img': item.get('img', item.get('image')),
                'text': item.get('text', item.get('sentence')),
                'label': item.get('label', 0)
            })
        print(f"Loaded {len(self.samples)} samples from HuggingFace: {dataset_name}")

    def _build_feature_index(self):
        """Build index of pre-extracted features."""
        for filename in os.listdir(self.feature_dir):
            if filename.endswith('.h5') or filename.endswith('.npy'):
                image_id = os.path.splitext(filename)[0]
                self.feature_index[image_id] = os.path.join(self.feature_dir, filename)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Get sample info
        sample_id = sample.get('id', idx)
        text = sample.get('text', '')
        label = sample.get('label', 0)
        img_path = sample.get('img', '')

        # Process text
        text_inputs = self._process_text(text)

        # Get visual features
        visual_inputs = self._get_visual_features(sample_id, img_path)

        return {
            'id': sample_id,
            'input_ids': text_inputs['input_ids'],
            'attention_mask': text_inputs['attention_mask'],
            'token_type_ids': text_inputs['token_type_ids'],
            'visual_features': visual_inputs['features'],
            'visual_attention_mask': visual_inputs['attention_mask'],
            'spatial_features': visual_inputs.get('spatial_features'),
            'label': torch.tensor(label, dtype=torch.long)
        }

    def _process_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize and encode text."""
        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'token_type_ids': encoding.get('token_type_ids',
                                           torch.zeros_like(encoding['input_ids'])).squeeze(0)
        }

    def _get_visual_features(
        self,
        sample_id: Union[int, str],
        img_path: str
    ) -> Dict[str, torch.Tensor]:
        """Get visual features (pre-extracted or on-the-fly)."""

        # Convert sample_id to string for indexing
        str_id = str(sample_id)

        # Try to load pre-extracted features
        if str_id in self.feature_index:
            return self._load_preextracted_features(str_id)

        # Try image filename as key
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        if img_id in self.feature_index:
            return self._load_preextracted_features(img_id)

        # Extract features on-the-fly
        if self.feature_extractor is not None:
            return self._extract_features_online(img_path)

        # Return dummy features if no extractor available
        return self._get_dummy_features()

    def _load_preextracted_features(self, feature_id: str) -> Dict[str, torch.Tensor]:
        """Load pre-extracted features from file."""
        filepath = self.feature_index[feature_id]

        if filepath.endswith('.h5'):
            with h5py.File(filepath, 'r') as f:
                features = torch.from_numpy(f['features'][:])
                spatial = torch.from_numpy(f['normalized_boxes'][:]) if 'normalized_boxes' in f else None
        else:
            data = np.load(filepath, allow_pickle=True).item()
            features = torch.from_numpy(data['features'])
            spatial = torch.from_numpy(data.get('normalized_boxes', np.zeros((36, 5))))

        # Pad or truncate to max_regions
        num_regions = features.shape[0]
        if num_regions > self.max_regions:
            features = features[:self.max_regions]
            spatial = spatial[:self.max_regions] if spatial is not None else None
            num_regions = self.max_regions
        elif num_regions < self.max_regions:
            pad_size = self.max_regions - num_regions
            features = torch.cat([features, torch.zeros(pad_size, features.shape[1])], dim=0)
            if spatial is not None:
                spatial = torch.cat([spatial, torch.zeros(pad_size, spatial.shape[1])], dim=0)

        # Create attention mask
        attention_mask = torch.zeros(self.max_regions)
        attention_mask[:num_regions] = 1

        return {
            'features': features.float(),
            'attention_mask': attention_mask.float(),
            'spatial_features': spatial.float() if spatial is not None else torch.zeros(self.max_regions, 5)
        }

    def _extract_features_online(self, img_path: str) -> Dict[str, torch.Tensor]:
        """Extract features on-the-fly."""
        full_path = os.path.join(self.image_dir, img_path)

        # Load and optionally transform image
        image = Image.open(full_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Extract features
        result = self.feature_extractor.extract_features(image)

        # Format output
        features = result['features']
        spatial = result.get('normalized_boxes', torch.zeros(features.shape[0], 5))

        # Pad or truncate
        num_regions = features.shape[0]
        if num_regions > self.max_regions:
            features = features[:self.max_regions]
            spatial = spatial[:self.max_regions]
            num_regions = self.max_regions
        elif num_regions < self.max_regions:
            pad_size = self.max_regions - num_regions
            features = torch.cat([features, torch.zeros(pad_size, features.shape[1])], dim=0)
            spatial = torch.cat([spatial, torch.zeros(pad_size, spatial.shape[1])], dim=0)

        attention_mask = torch.zeros(self.max_regions)
        attention_mask[:num_regions] = 1

        return {
            'features': features.float(),
            'attention_mask': attention_mask.float(),
            'spatial_features': spatial.float()
        }

    def _get_dummy_features(self) -> Dict[str, torch.Tensor]:
        """Return dummy features when no extractor is available."""
        return {
            'features': torch.randn(self.max_regions, 2048),
            'attention_mask': torch.ones(self.max_regions),
            'spatial_features': torch.zeros(self.max_regions, 5)
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for DataLoader."""
    return {
        'id': [item['id'] for item in batch],
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
        'token_type_ids': torch.stack([item['token_type_ids'] for item in batch]),
        'visual_features': torch.stack([item['visual_features'] for item in batch]),
        'visual_attention_mask': torch.stack([item['visual_attention_mask'] for item in batch]),
        'spatial_features': torch.stack([item['spatial_features'] for item in batch]),
        'labels': torch.stack([item['label'] for item in batch])
    }


def create_dataloaders(
    train_path: str,
    val_path: str,
    test_path: str,
    image_dir: str,
    feature_dir: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    max_seq_length: int = 128,
    max_regions: int = 36
) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        train_path: Path to training JSONL
        val_path: Path to validation JSONL
        test_path: Path to test JSONL
        image_dir: Directory containing images
        feature_dir: Directory containing pre-extracted features
        batch_size: Batch size
        num_workers: Number of data loading workers
        max_seq_length: Maximum text sequence length
        max_regions: Maximum visual regions

    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    datasets = {}
    dataloaders = {}

    # Training set
    if train_path and os.path.exists(train_path):
        datasets['train'] = HatefulMemesDataset(
            data_path=train_path,
            image_dir=image_dir,
            feature_dir=feature_dir,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            max_regions=max_regions,
            split='train'
        )
        dataloaders['train'] = DataLoader(
            datasets['train'],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

    # Validation set
    if val_path and os.path.exists(val_path):
        datasets['val'] = HatefulMemesDataset(
            data_path=val_path,
            image_dir=image_dir,
            feature_dir=feature_dir,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            max_regions=max_regions,
            split='val'
        )
        dataloaders['val'] = DataLoader(
            datasets['val'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

    # Test set
    if test_path and os.path.exists(test_path):
        datasets['test'] = HatefulMemesDataset(
            data_path=test_path,
            image_dir=image_dir,
            feature_dir=feature_dir,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            max_regions=max_regions,
            split='test'
        )
        dataloaders['test'] = DataLoader(
            datasets['test'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

    return dataloaders


class HatefulMemesHuggingFace:
    """
    Helper class to download and setup Hateful Memes from HuggingFace.

    Dataset: neuralcatcher/hateful_memes
    """

    @staticmethod
    def download_and_prepare(
        output_dir: str,
        save_images: bool = True
    ) -> Dict[str, str]:
        """
        Download the dataset and prepare it for training.

        Args:
            output_dir: Directory to save the prepared dataset
            save_images: Whether to save images locally

        Returns:
            Dictionary with paths to train/val/test JSONL files
        """
        if not HF_AVAILABLE:
            raise ImportError("Please install datasets: pip install datasets")

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'img'), exist_ok=True)

        # Load from HuggingFace
        dataset = load_dataset('neuralcatcher/hateful_memes')

        paths = {}

        for split_name, split_data in dataset.items():
            jsonl_path = os.path.join(output_dir, f'{split_name}.jsonl')

            with open(jsonl_path, 'w') as f:
                for idx, item in enumerate(split_data):
                    # Save image if requested
                    if save_images and 'image' in item:
                        img = item['image']
                        img_filename = f'img/{item.get("id", idx)}.png'
                        img_path = os.path.join(output_dir, img_filename)
                        img.save(img_path)
                    else:
                        img_filename = item.get('img', f'img/{idx}.png')

                    # Write annotation
                    sample = {
                        'id': item.get('id', idx),
                        'img': img_filename,
                        'text': item.get('text', ''),
                        'label': item.get('label', 0)
                    }
                    f.write(json.dumps(sample) + '\n')

            paths[split_name] = jsonl_path
            print(f"Saved {split_name} split to {jsonl_path}")

        return paths


if __name__ == '__main__':
    # Test the dataset with dummy data
    print("Testing HatefulMemesDataset...")

    # Create a dummy dataset
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Test text processing
    text = "This is a test meme text"
    encoding = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    print(f"Input IDs shape: {encoding['input_ids'].shape}")
    print(f"Attention mask shape: {encoding['attention_mask'].shape}")

    # Test dummy features
    dummy_dataset = HatefulMemesDataset.__new__(HatefulMemesDataset)
    dummy_dataset.max_regions = 36
    features = dummy_dataset._get_dummy_features()
    print(f"Visual features shape: {features['features'].shape}")
    print(f"Visual attention mask shape: {features['attention_mask'].shape}")

    print("\nDataset tests passed!")
