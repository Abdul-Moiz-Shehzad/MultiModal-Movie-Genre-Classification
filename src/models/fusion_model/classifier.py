import numpy as np
import joblib
from typing import Dict
from src.models.text_model.classifier import TextGenreClassifier
from src.models.image_model.classifier import ImageGenreClassifier
from src.utils.config_loader import load_config
import tensorflow as tf
class FusionGenreClassifier:
    def __init__(self):
        self.config = load_config("configs/base_config.yaml")
        print(f"Image Model Path from Fusion: {self.config['image_model']['model_path']}")
        print(f"TensorFlow Version (Image Classifier): {tf.__version__}")
        self.text_classifier=TextGenreClassifier()
        self.image_classifier = ImageGenreClassifier()
        self.label_encoder = joblib.load(self.config['fusion_model']["label_encoder_path"])
        self.weights = {
            'text': self.config['fusion_model']["text_weight"],
            'image': self.config['fusion_model']["image_weight"]
        }

    def _combine_probs(self, text_probs: Dict, image_probs: Dict) -> Dict:
        """Combine probabilities using configured weights"""
        combined = {}
        for label in self.label_encoder.classes_:
            combined[label] = (
                self.weights['text'] * text_probs[label] +
                self.weights['image'] * image_probs[label]
            )
        return combined

    def predict(self, raw_text: str, raw_image: np.ndarray) -> str:
        """Get final prediction using fused probabilities"""
        text_probs = self.text_classifier.predict_proba(raw_text)
        image_probs = self.image_classifier.predict_proba(raw_image)
        combined = self._combine_probs(text_probs, image_probs)
        return max(combined, key=combined.get)

    def predict_proba(self, raw_text: str, raw_image: np.ndarray) -> Dict:
        """Get combined probability distribution"""
        text_probs = self.text_classifier.predict_proba(raw_text)
        image_probs = self.image_classifier.predict_proba(raw_image)
        return self._combine_probs(text_probs, image_probs)
