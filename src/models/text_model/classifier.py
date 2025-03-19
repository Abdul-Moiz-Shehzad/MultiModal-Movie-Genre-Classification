import tensorflow as tf
import joblib
from sentence_transformers import SentenceTransformer
from src.data.text.preprocessing import preprocess_text
from src.utils.config_loader import load_config

class TextGenreClassifier:
    def __init__(self):
        self.config = load_config("configs/base_config.yaml")["text_model"]
        self.encoder = SentenceTransformer(self.config["encoder_path"])
        self.model = tf.keras.models.load_model(self.config["model_path"])
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