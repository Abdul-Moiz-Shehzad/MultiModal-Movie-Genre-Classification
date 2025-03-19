# test_predictions.py
from src.models.text_model.classifier import TextGenreClassifier

def test_classifier():
    print("\nMovie Genre Classifier Tester")
    print("Type 'exit' to quit\n")
    
    classifier = TextGenreClassifier()
    
    while True:
        text = input("Enter movie plot description:\n> ")
        
        if text.lower() in ['exit', 'quit']:
            print("\n👋 Exiting...")
            break
            
        if not text.strip():
            print("⚠️ Please enter some text")
            continue
            
        try:
            genre = classifier.predict(text)
            print(f"\nPredicted Genre: {genre}\n{'━'*30}\n")
        except Exception as e:
            print(f"❌ Prediction failed: {str(e)}")

if __name__ == "__main__":
    test_classifier()