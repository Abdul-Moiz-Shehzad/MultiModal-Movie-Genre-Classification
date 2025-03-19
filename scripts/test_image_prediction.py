import cv2
import sys
from src.models.image_model.classifier import ImageGenreClassifier

def test_image_classifier():
    print("\n🎬 Movie Poster Genre Classifier Tester")
    print("Enter image paths to test (or 'exit' to quit)\n")
    
    classifier = ImageGenreClassifier()
    
    while True:
        image_path = input("Enter movie poster image path:\n> ").strip()
        
        if image_path.lower() in ['exit', 'quit']:
            print("\n👋 Exiting...")
            break
            
        if not image_path:
            print("⚠️ Please enter an image path")
            continue
            
        try:
            # Read image using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Could not read image from {image_path}")
                continue
                
            # Get prediction
            genre = classifier.predict(image)
            probabilities = classifier.predict_proba(image)
            
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
    test_image_classifier()