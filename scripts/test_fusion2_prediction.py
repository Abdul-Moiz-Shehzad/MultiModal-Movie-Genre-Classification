import cv2
import numpy as np
from src.models.fusion_model.classifier import FusionGenreClassifierWithLIME
import os

def test_fusion_classifier_with_lime():
    print("\n🎬 Movie Poster Fusion Genre Classifier Tester (LIME-enabled)")
    print("Running automated test case...\n")

    try:
        classifier = FusionGenreClassifierWithLIME(dynamic_weights=True)
    except Exception as e:
        print(f"❌ Failed to initialize classifier: {str(e)}")
        return

    image_path = "temp/pic.jpg"
    text_description = "John Wick is a former hitman grieving the loss of his true love. When his home is broken into, robbed, and his dog killed, he is forced to return to action to exact revenge."

    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not read image from {image_path}")
            return

        genre = classifier.predict(text_description, image)
        probabilities = classifier.predict_proba(text_description, image)

        print(f"\nPredicted Genre: {genre}")
        print("Class Probabilities:")
        for cls, prob in probabilities.items():
            print(f"  {cls}: {prob:.2%}")

        print("\n🔍 Generating LIME explanation for text input...")
        lime_exp = classifier.explain_text(text_description)
        lime_exp.save_to_file('lime_text_explanation.html')

    except Exception as e:
        print(f"❌ Prediction failed: {str(e)}")

if __name__ == "__main__":
    test_fusion_classifier_with_lime()