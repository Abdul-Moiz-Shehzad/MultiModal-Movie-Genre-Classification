import cv2
import numpy as np
from src.models.fusion_model.classifier import FusionGenreClassifier
import os

def test_fusion_classifier():
    print("\n🎬 Movie Poster Fusion Genre Classifier Tester")
    print("Enter an image path and text description to test (or 'exit' to quit)\n")

    classifier = FusionGenreClassifier()

    while True:
        image_path = input("Enter movie poster image path:\n> ").strip()
        if image_path.lower() in ['exit', 'quit']:
            print("\n👋 Exiting...")
            break

        text_description = input("Enter movie description (or 'skip' to use only image):\n> ").strip()

        if not image_path and not text_description:
            print("⚠️ Please provide an image or a description!")
            continue

        try:
            image = None
            if image_path:
                image = cv2.imread(image_path)
                if image is None:
                    print(f"❌ Could not read image from {image_path}")
                    continue

            # Get prediction
            genre = classifier.predict(text_description if text_description else "", image)
            probabilities = classifier.predict_proba(text_description if text_description else "", image)

            print(f"\nPredicted Genre: {genre}")
            print("Class Probabilities:")
            for cls, prob in probabilities.items():
                print(f"  {cls}: {prob:.2%}")
            print(f"{'━'*30}\n")

        except Exception as e:
            print(f"❌ Prediction failed: {str(e)}")
            if "CUDA" in str(e):
                print("Note: You might need to configure your GPU setup")

if __name__ == "__main__":
    test_fusion_classifier()
