"""FinBERT sentiment analysis with GPU acceleration."""

from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm


class SentimentAnalyzer:
    """FinBERT sentiment analysis."""
    
    def __init__(
        self,
        model_name: str = "yiyanghkust/finbert-tone",
        device: str = "cuda",
        batch_size: int = 16,
        max_length: int = 512
    ):
        """Initialize sentiment analyzer.
        
        Args:
            model_name: Model name/path
            device: Device (cuda, cpu)
            batch_size: Batch size for inference
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        
        # Get label mapping
        self.id2label = self.model.config.id2label
        self.label2id = self.model.config.label2id
    
    def analyze_batch(self, texts: List[str]) -> List[Tuple[str, float, float, float]]:
        """Analyze sentiment for a batch of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of (label, p_positive, p_neutral, p_negative) tuples
        """
        results = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Sentiment analysis"):
            batch = texts[i:i + self.batch_size]
            batch_results = self._process_batch(batch)
            results.extend(batch_results)
        
        return results
    
    def _process_batch(self, texts: List[str]) -> List[Tuple[str, float, float, float]]:
        """Process a single batch.
        
        Args:
            texts: Batch of texts
            
        Returns:
            List of results
        """
        # Tokenize
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Compute probabilities via softmax
            probs = F.softmax(logits, dim=-1).cpu().numpy()
        
        # Parse results
        results = []
        for prob_dist in probs:
            # Map to labels
            label_probs = {}
            for idx, prob in enumerate(prob_dist):
                label = self.id2label.get(idx, f"LABEL_{idx}").lower()
                label_probs[label] = float(prob)
            
            # Get dominant label
            max_label = max(label_probs.keys(), key=lambda k: label_probs[k])
            
            # Extract individual probabilities
            p_positive = label_probs.get("positive", 0.0)
            p_neutral = label_probs.get("neutral", 0.0)
            p_negative = label_probs.get("negative", 0.0)
            
            results.append((max_label, p_positive, p_neutral, p_negative))
        
        return results
    
    def analyze_single(self, text: str) -> Tuple[str, float, float, float]:
        """Analyze sentiment for a single text.
        
        Args:
            text: Input text
            
        Returns:
            (label, p_positive, p_neutral, p_negative)
        """
        results = self.analyze_batch([text])
        return results[0]

