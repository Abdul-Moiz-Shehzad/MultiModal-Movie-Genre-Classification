import tensorflow as tf
import joblib
import cv2
import numpy as np
from src.utils.config_loader import load_config
from src.data.Images.preprocessing import preprocess_image

class ImageGenreClassifier:
    def __init__(self):
        self.config = load_config("configs/base_config.yaml")["image_model"]
        
        self.model = tf.keras.models.load_model(self.config["model_path"])
        self.label_encoder = joblib.load(self.config["label_encoder_path"])
        
    def predict(self, raw_image: np.ndarray) -> str:
        """
        Predict genre from raw BGR image (OpenCV format)
        
        Args:
            raw_image: Input image in BGR format with any size
            
        Returns:
            Predicted genre label
        """
        processed_image = preprocess_image(raw_image)

        predictions = self.model.predict(processed_image)
        predicted_class = np.argmax(predictions, axis=1)
        
        return self.label_encoder.inverse_transform(predicted_class)[0]

    def predict_proba(self, raw_image: np.ndarray) -> dict:
        """
        Get prediction probabilities for all classes
        
        Args:
            raw_image: Input image in BGR format with any size
            
        Returns:
            Dictionary of class probabilities
        """
        processed_image = preprocess_image(raw_image)
        predictions = self.model.predict(processed_image)[0]
        
        return {
            label: float(prob) 
            for label, prob in zip(self.label_encoder.classes_, predictions)
        }