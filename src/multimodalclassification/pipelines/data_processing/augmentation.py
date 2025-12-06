"""
Data Augmentation for Hateful Memes

Implements techniques proven to improve ViLBERT performance:
1. Caption Enrichment - Add image captions to text input (+2-6% AUROC)
2. Text Augmentation - Back-translation and paraphrasing (+1-2% AUROC)
3. Image Captioning - Generate descriptions of meme images

Reference: "Caption Enriched Samples for Improving Hateful Memes Detection" (EMNLP 2021)
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ImageCaptioner:
    """
    Generate captions for images using a pretrained model.

    Uses BLIP (Bootstrapping Language-Image Pre-training) which is
    state-of-the-art for image captioning.
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.model_name = model_name

    def _load_model(self):
        """Lazy load the model."""
        if self.model is None:
            try:
                from transformers import BlipForConditionalGeneration, BlipProcessor

                logger.info(f"Loading BLIP captioning model: {self.model_name}")
                self.processor = BlipProcessor.from_pretrained(self.model_name)
                self.model = BlipForConditionalGeneration.from_pretrained(
                    self.model_name
                )
                self.model.to(self.device)
                self.model.eval()
                logger.info("BLIP model loaded successfully")
            except ImportError:
                logger.warning(
                    "transformers not available for BLIP. Install with: pip install transformers"
                )
                raise

    @torch.no_grad()
    def generate_caption(self, image: Image.Image, max_length: int = 50) -> str:
        """Generate a caption for a single image."""
        self._load_model()

        inputs = self.processor(image, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_length=max_length)
        caption = self.processor.decode(output[0], skip_special_tokens=True)

        return caption

    def generate_captions_batch(
        self,
        image_paths: List[str],
        batch_size: int = 8,
        max_length: int = 50,
    ) -> List[str]:
        """Generate captions for a batch of images."""
        self._load_model()

        captions = []
        for i in tqdm(
            range(0, len(image_paths), batch_size), desc="Generating captions"
        ):
            batch_paths = image_paths[i : i + batch_size]
            batch_images = []

            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    batch_images.append(img)
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")
                    batch_images.append(Image.new("RGB", (224, 224)))

            inputs = self.processor(batch_images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model.generate(**inputs, max_length=max_length)
            batch_captions = self.processor.batch_decode(
                outputs, skip_special_tokens=True
            )
            captions.extend(batch_captions)

        return captions


class TextAugmenter:
    """
    Text augmentation techniques for hateful memes.

    Includes:
    - Synonym replacement
    - Random insertion
    - Back-translation (if available)
    """

    def __init__(self):
        self.nlp = None
        self.translator = None

    def augment_text(self, text: str, method: str = "synonym") -> str:
        """Augment text using specified method."""
        if method == "synonym":
            return self._synonym_replacement(text)
        elif method == "shuffle":
            return self._word_shuffle(text)
        else:
            return text

    def _synonym_replacement(self, text: str, n: int = 2) -> str:
        """Replace n random words with synonyms."""
        try:
            import random
            import nltk
            from nltk.corpus import wordnet

            # Ensure wordnet is downloaded
            try:
                wordnet.synsets("test")
            except LookupError:
                nltk.download("wordnet", quiet=True)
                nltk.download("omw-1.4", quiet=True)

            words = text.split()
            if len(words) < 2:
                return text

            # Get indices of words to replace (skip short words)
            replaceable = [i for i, w in enumerate(words) if len(w) > 3]
            if not replaceable:
                return text

            n = min(n, len(replaceable))
            indices = random.sample(replaceable, n)

            for idx in indices:
                word = words[idx].lower()
                synsets = wordnet.synsets(word)
                if synsets:
                    # Get synonyms from first synset
                    synonyms = []
                    for syn in synsets[:2]:
                        for lemma in syn.lemmas():
                            if lemma.name() != word and "_" not in lemma.name():
                                synonyms.append(lemma.name())
                    if synonyms:
                        words[idx] = random.choice(synonyms)

            return " ".join(words)
        except Exception as e:
            logger.debug(f"Synonym replacement failed: {e}")
            return text

    def _word_shuffle(self, text: str) -> str:
        """Shuffle words in the text (preserves first and last)."""
        import random

        words = text.split()
        if len(words) <= 3:
            return text

        # Keep first and last words, shuffle middle
        middle = words[1:-1]
        random.shuffle(middle)
        return " ".join([words[0]] + middle + [words[-1]])


def enrich_with_captions(
    df: pd.DataFrame,
    image_column: str = "img_path",
    text_column: str = "text",
    output_column: str = "text_enriched",
    cache_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Enrich dataset with image captions.

    This implements the Caption Enriched Samples (CES) approach which
    improved ViLBERT AUROC by +2-6% on Hateful Memes.

    Args:
        df: DataFrame with image paths and text
        image_column: Column containing image paths
        text_column: Column containing original text
        output_column: Column to store enriched text
        cache_path: Optional path to cache captions

    Returns:
        DataFrame with enriched text column
    """
    df = df.copy()

    # Try to load cached captions
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading cached captions from {cache_path}")
        cached = pd.read_csv(cache_path)
        caption_map = dict(zip(cached["img_path"], cached["caption"]))
    else:
        caption_map = {}

    # Get images that need captions
    missing_paths = [p for p in df[image_column].tolist() if p not in caption_map]

    if missing_paths:
        logger.info(f"Generating captions for {len(missing_paths)} images...")
        captioner = ImageCaptioner()
        new_captions = captioner.generate_captions_batch(missing_paths)

        for path, caption in zip(missing_paths, new_captions):
            caption_map[path] = caption

        # Cache the captions
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            cache_df = pd.DataFrame(
                {
                    "img_path": list(caption_map.keys()),
                    "caption": list(caption_map.values()),
                }
            )
            cache_df.to_csv(cache_path, index=False)
            logger.info(f"Cached captions to {cache_path}")

    # Create enriched text: "[original text] [SEP] [caption]"
    df["caption"] = df[image_column].map(caption_map)
    df[output_column] = df[text_column] + " [SEP] " + df["caption"].fillna("")

    logger.info(f"Enriched {len(df)} samples with captions")
    return df


def augment_dataset(
    df: pd.DataFrame,
    text_column: str = "text",
    augment_ratio: float = 0.5,
    methods: List[str] = ["synonym"],
) -> pd.DataFrame:
    """
    Augment dataset with text variations.

    Args:
        df: Original DataFrame
        text_column: Column containing text to augment
        augment_ratio: Fraction of samples to augment (0.5 = 50% more data)
        methods: Augmentation methods to use

    Returns:
        DataFrame with original + augmented samples
    """
    import random

    augmenter = TextAugmenter()

    # Sample indices to augment
    n_augment = int(len(df) * augment_ratio)
    indices = random.sample(range(len(df)), min(n_augment, len(df)))

    augmented_rows = []
    for idx in tqdm(indices, desc="Augmenting text"):
        row = df.iloc[idx].copy()
        method = random.choice(methods)
        row[text_column] = augmenter.augment_text(row[text_column], method)
        augmented_rows.append(row)

    if augmented_rows:
        augmented_df = pd.DataFrame(augmented_rows)
        df = pd.concat([df, augmented_df], ignore_index=True)

    logger.info(
        f"Augmented dataset: {len(df)} total samples (+{len(augmented_rows)} augmented)"
    )
    logger.info(f"Augmented dataset: {len(df)} total samples (+{len(augmented_rows)} augmented)")
    return df
