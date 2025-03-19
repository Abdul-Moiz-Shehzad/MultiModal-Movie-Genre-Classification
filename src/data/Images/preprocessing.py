import cv2
import numpy as np
import tensorflow as tf

def preprocess_image(bgr_image: np.ndarray) -> np.ndarray:
    """
    Preprocess BGR image for EfficientNetB3 model input.
    
    Args:
        bgr_image: Input image in BGR format (OpenCV default)
        
    Returns:
        Preprocessed RGB image in format expected by EfficientNetB3
    """
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(rgb_image, (224, 224))
    float_image = tf.cast(resized_image, tf.float32)
    preprocessed_image = tf.keras.applications.efficientnet.preprocess_input(float_image)
    return np.expand_dims(preprocessed_image, axis=0)