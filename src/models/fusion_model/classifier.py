import numpy as np
import joblib
from typing import Dict
from src.models.text_model.classifier import TextGenreClassifier
from src.models.image_model.classifier import ImageGenreClassifier
from src.utils.config_loader import load_config
from src.data.text.preprocessing import preprocess_text
import tensorflow as tf
from lime.lime_text import LimeTextExplainer
class FusionGenreClassifier:
    def __init__(self, dynamic_weights: bool = False):
        self.config = load_config("configs/base_config.yaml")
        print(f"Image Model Path from Fusion: {self.config['image_model']['model_path']}")
        print(f"TensorFlow Version (Image Classifier): {tf.__version__}")
        self.text_classifier = TextGenreClassifier()
        self.image_classifier = ImageGenreClassifier()
        self.label_encoder =  joblib.load(self.config['fusion_model']["label_encoder_path"])
        self.dynamic_weights = dynamic_weights
        self.static_weights = {
            'text': self.config['fusion_model']["text_weight"],
            'image': self.config['fusion_model']["image_weight"]
        }

    def _get_dynamic_weights(self, text_probs: dict, image_probs: dict) -> dict:
        text_conf = max(text_probs.values())
        image_conf = max(image_probs.values())
        total = text_conf + image_conf
        return {'text': text_conf / total, 'image': image_conf / total} if total > 0 else self.static_weights

    def _combine_probs(self, text_probs: dict, image_probs: dict) -> dict:
        weights = self._get_dynamic_weights(text_probs, image_probs) if self.dynamic_weights else self.static_weights
        return {
            label: weights['text'] * text_probs[label] + weights['image'] * image_probs[label]
            for label in self.label_encoder.classes_
        }

    def predict(self, raw_text: str, raw_image: np.ndarray) -> str:
        text_probs = self.text_classifier.predict_proba(raw_text)
        image_probs = self.image_classifier.predict_proba(raw_image) if raw_image is not None else {k: 0 for k in self.label_encoder.classes_}
        return max((self._combine_probs(text_probs, image_probs)), key=lambda x: self._combine_probs(text_probs, image_probs)[x])

    def predict_proba(self, raw_text: str, raw_image: np.ndarray) -> dict:
        text_probs = self.text_classifier.predict_proba(raw_text)
        image_probs = self.image_classifier.predict_proba(raw_image) if raw_image is not None else {k: 0 for k in self.label_encoder.classes_}
        return self._combine_probs(text_probs, image_probs)
    
class FusionGenreClassifierWithLIME(FusionGenreClassifier):
    def __init__(self, dynamic_weights: bool = False):
        super().__init__(dynamic_weights)
        self.class_names = list(self.label_encoder.classes_)
        self.explainer = LimeTextExplainer(class_names=self.class_names, bow=False)

    def _lime_predict_wrapper(self, texts):
        return np.array([
            list(self.text_classifier.predict_proba(text).values())
            for text in texts
        ])

    def explain_text(self, raw_text: str):
        explanation = self.explainer.explain_instance(
            raw_text,
            self._lime_predict_wrapper,
            num_features=10,
            num_samples=500,
            labels=list(range(len(self.class_names)))
        )

        print("\n🔍 LIME Explanations per Class:")
        for label_idx in explanation.available_labels():
            label_name = self.class_names[label_idx]
            print(f"\n📌 Class: {label_name}")
            for word, weight in explanation.as_list(label=label_idx):
                print(f"  {word}: {weight:.4f}")

        return explanation
