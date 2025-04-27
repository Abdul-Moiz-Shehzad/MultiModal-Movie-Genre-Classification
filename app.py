from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import cv2
from src.models.text_model.classifier import TextGenreClassifier
from src.models.image_model.classifier import ImageGenreClassifier
from src.models.fusion_model.classifier import FusionGenreClassifier

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Instantiate models once
text_classifier = TextGenreClassifier()
image_classifier = ImageGenreClassifier()
fusion_classifier = FusionGenreClassifier()

@app.route('/reset', methods=['GET'])
def reset():
    # Redirect to the homepage to clear all state
    return redirect(url_for('index'))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probabilities = None
    image_path = None  # Initialize image_path

    if request.method == 'POST':
        description = request.form.get('description', '').strip()
        poster = request.files.get('poster')

        poster_uploaded = poster and poster.filename != ''

        if poster_uploaded:
            filename = secure_filename(poster.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            poster.save(save_path)
            image_path = filename  # Store the filename
            image = cv2.imread(save_path)
        else:
            image = None

        # Decide prediction logic
        if description and image is not None:
            try:
                prediction = fusion_classifier.predict(description, image)
                probabilities = fusion_classifier.predict_proba(description, image)
            except Exception as e:
                prediction = f"❌ Prediction failed: {str(e)}"
        elif description:
            try:
                prediction = text_classifier.predict(description)
                probabilities = text_classifier.predict_proba(description)
            except Exception as e:
                prediction = f"❌ Prediction failed: {str(e)}"
        elif image is not None:
            try:
                prediction = image_classifier.predict(image)
                probabilities = image_classifier.predict_proba(image)
            except Exception as e:
                prediction = f"❌ Prediction failed: {str(e)}"
        else:
            prediction = "⚠️ Please provide a description and/or upload a poster."

    return render_template('index.html', prediction=prediction, probabilities=probabilities, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
