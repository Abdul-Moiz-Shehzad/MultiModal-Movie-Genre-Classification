import tensorflow as tf
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from src.data.text.preprocessing import preprocess_text
from src.utils.config_loader import load_config

class TextGenreClassifier:
    def __init__(self):
        self.config = load_config("configs/base_config.yaml")["text_model"]
        self.encoder = SentenceTransformer(self.config["encoder_path"])
        self.model = tf.keras.models.load_model(self.config["model_path"], compile=False)
        self.label_encoder = joblib.load(self.config["label_encoder_path"])
    
    def predict(self, raw_text: str) -> str:
        preprocess_params = self.config.get("preprocess_params", {})
        
        clean_text = preprocess_text(
            raw_text, 
            return_lst=preprocess_params.get("return_lst", False)
        )
        embedding = self.encoder.encode([clean_text])
        pred = self.model.predict(embedding).argmax()
        return self.label_encoder.inverse_transform([pred])[0]

    def predict_proba(self, raw_text: str) -> dict:
        """
        Predicts genre probabilities for the given text.

        Args:
            raw_text: A string containing the text description.

        Returns:
            A dictionary where keys are genre labels and values are probabilities.
        """
        preprocess_params = self.config.get("preprocess_params", {})
        clean_text = preprocess_text(
            raw_text, 
            return_lst=preprocess_params.get("return_lst", False)
        )
        embedding = self.encoder.encode([clean_text])
        preds = self.model.predict(embedding)[0]

        return {label: float(prob) for label, prob in zip(self.label_encoder.classes_, preds)}
