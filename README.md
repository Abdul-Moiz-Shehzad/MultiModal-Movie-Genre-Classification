# Multimodal Movie Genre Classification

A deep learning system for movie genre classification utilizing late fusion of textual plot summaries and visual poster features. The architecture combines Sentence-Transformer text embeddings and EfficientNet visual representations with both static and confidence-weighted dynamic decision fusion.

---

## Overview

Predicting movie genres from metadata is inherently multimodal. While plot synopses provide explicit semantic indicators of narrative themes, movie posters convey mood, color palettes, and visual composition. This project provides an end-to-end multimodal classification pipeline supporting four target genres:

- Action
- Comedy
- Horror
- Romance

The system supports independent single-modality inference (text-only or image-only) as well as joint multimodal late fusion, complete with Local Interpretable Model-agnostic Explanations (LIME) for model interpretability and a Flask-based web interface.

---

## System Architecture

```
                                  +---------------------------------------+
                                  |         Raw Inputs (Multimodal)       |
                                  |  - Text: Plot Synopsis / Description   |
                                  |  - Image: Movie Poster (RGB / BGR)    |
                                  +-------------------+-------------------+
                                                      |
                         +----------------------------+----------------------------+
                         |                                                         |
                         v                                                         v
             [ Text Preprocessing ]                                    [ Image Preprocessing ]
             - Non-alphabetic regex filter                             - BGR to RGB conversion
             - Lowercase normalization                                 - Resize to (224, 224)
             - WordNet lemmatization & stopword removal                - EfficientNet input scaling
                         |                                                         |
                         v                                                         v
             [ Feature Extraction ]                                    [ Feature Extraction ]
             - SentenceTransformer (all-mpnet-base-v2)                 - Pretrained EfficientNetB3
             - 768-dimensional dense vector                            - Transfer learning / Fine-tuning
                         |                                                         |
                         v                                                         v
             [ Text Classifier Head ]                                  [ Vision Classifier Head ]
             - Dense / MLP classification layers                       - Global pooling + Dense classifier
             - Softmax output: P(Text)                                 - Softmax output: P(Image)
                         |                                                         |
                         +----------------------------+----------------------------+
                                                      |
                                                      v
                                        [ Multimodal Late Fusion ]
                         +---------------------------------------------------------+
                         |  Static Weighted Combination:                           |
                         |    P(Class) = w_text * P(Text) + w_image * P(Image)     |
                         |                                                         |
                         |  Dynamic Confidence-Based Combination:                  |
                         |    w_text  = max(P(Text))  / (max(P(T)) + max(P(I)))    |
                         |    w_image = max(P(Image)) / (max(P(T)) + max(P(I)))    |
                         +----------------------------+----------------------------+
                                                      |
                                                      v
                                      +-------------------------------+
                                      |   Final Predicted Genre       |
                                      |   Class Probability Vector    |
                                      |   LIME Feature Attribution    |
                                      +-------------------------------+
```

### Fusion Mechanisms

1. **Static Late Fusion**: Combines probability distributions using fixed, configurable hyperparameter weights (default: `w_text = 0.7`, `w_image = 0.3`):
   ```
   P_final(c) = w_text * P_text(c) + w_image * P_image(c)
   ```

2. **Dynamic Confidence-Weighted Fusion**: Computes adaptive weights at inference time based on the maximum softmax confidence of each modality:
   ```
   c_text = max_{c} P_text(c)
   c_image = max_{c} P_image(c)
   w_text = c_text / (c_text + c_image)
   w_image = c_image / (c_text + c_image)
   P_final(c) = w_text * P_text(c) + w_image * P_image(c)
   ```

3. **Single-Modality Fallback**: If only text or only an image is supplied, the fusion layer automatically routes inference to the active modality without requiring code modification.

---

## Directory Layout

```
MultiModal-Movie-Genre-Classification/
|-- app.py                                  # Flask application entry point
|-- setup.py                                # Package installation script
|-- requirements.txt                        # Python dependencies
|-- configs/
|   `-- base_config.yaml                    # Global paths and model hyperparameters
|-- data/
|   |-- raw/
|   |   `-- IMDB_four_genre_larger_plot_description.csv  # 1,000-sample balanced dataset
|   `-- external/
|       |-- action.csv                      # External scraped action movie dataset
|       |-- horror.csv                      # External scraped horror movie dataset
|       `-- romance.csv                     # External scraped romance movie dataset
|-- notebooks/
|   |-- eda_text_analysis.ipynb             # Text modeling experiments & embeddings
|   |-- eda_images_analysis.ipynb           # CNN, ResNet, EfficientNet poster experiments
|   |-- image_data.ipynb                    # Poster image extraction & verification
|   `-- research.ipynb                      # Advanced text augmentation, SMOTE, attention models
|-- scripts/
|   |-- test_text_prediction.py             # CLI interactive tester for text classifier
|   |-- test_image_prediction.py            # CLI interactive tester for image classifier
|   |-- test_fusion_prediction.py           # CLI interactive tester for multimodal fusion
|   `-- test_fusion2_prediction.py          # Automated test with LIME explanation export
|-- src/
|   |-- __init__.py
|   |-- data/
|   |   |-- Images/
|   |   |   |-- __init__.py
|   |   |   `-- preprocessing.py            # OpenCV and EfficientNet image preprocessing
|   |   `-- text/
|   |       |-- __init__.py
|   |       `-- preprocessing.py            # NLTK and regex text preprocessing
|   |-- models/
|   |   |-- fusion_model/
|   |   |   |-- __init__.py
|   |   |   `-- classifier.py               # Fusion classifier and LIME explainer
|   |   |-- image_model/
|   |   |   |-- __init__.py
|   |   |   `-- classifier.py               # Image inference wrapper
|   |   `-- text_model/
|   |       |-- __init__.py
|   |       `-- classifier.py               # Text inference wrapper (MPNet + Keras)
|   `-- utils/
|       |-- __init__.py
|       `-- config_loader.py                # YAML loader with project root resolution
|-- static/
|   |-- style.css                           # Web UI stylesheets
|   `-- uploads/                            # Temporary upload directory for poster images
|-- templates/
|   `-- index.html                          # Bootstrap 5 multimodal inference web template
`-- tests/
    |-- __init__.py
    `-- test_data.py                        # Preprocessing unit tests
```

---

## Data and Preprocessing Pipelines

### Text Pipeline

Located in `src/data/text/preprocessing.py`:
- Strips non-alphabetic characters using regular expressions (`[^A-Za-z\s]`).
- Converts characters to lowercase and strips leading/trailing whitespace.
- Tokenizes strings into word tokens using NLTK `word_tokenize`.
- Filters English stopwords via NLTK corpus.
- Lemmatizes remaining tokens with NLTK `WordNetLemmatizer`.
- Generates 768-dimensional contextual sentence embeddings via `sentence-transformers/all-mpnet-base-v2`.

### Vision Pipeline

Located in `src/data/Images/preprocessing.py`:
- Converts BGR image matrices (standard OpenCV format) to RGB.
- Resizes posters to standard input dimensions of `(224, 224, 3)`.
- Casts to `tf.float32`.
- Applies EfficientNet preprocessing normalization (`tf.keras.applications.efficientnet.preprocess_input`).
- Expands batch dimensions for downstream model inference.

### Research and Data Augmentation

During research (`notebooks/research.ipynb` and `notebooks/eda_images_analysis.ipynb`), several strategies were developed:
- **Synonym Augmentation**: Replaces random non-stopword tokens with WordNet synonyms and performs controlled random word deletion.
- **Class Balancing**: Balanced training sets using class-weighted loss computed via `numpy.bincount` and SMOTE over-sampling.
- **Mixup Augmentation**: Linear interpolation between poster pairs and one-hot label mixtures for image regularization.

---

## Configuration Reference

Project parameters and paths are defined in `configs/base_config.yaml`:

```yaml
text_model:
  preprocess_params:
    return_lst: False
  model_path: "data/processed/text/final_model_v3.keras"
  encoder_path: "data/processed/text/mpnet_encoder"
  label_encoder_path: "data/processed/text/label_encoder.pkl"
  class_weights: {0: 1, 1: 2.5, 2: 1, 3: 1}

image_model:
  label_encoder_path: "data/processed/text/label_encoder.pkl"
  model_path: "data/processed/images/image_model.keras"

fusion_model:
  text_weight: 0.7
  image_weight: 0.3
  label_encoder_path: "data/processed/text/label_encoder.pkl"
```

The configuration loader (`src/utils/config_loader.py`) resolves paths containing `"path"` relative to the repository root directory.

---

## Installation and Setup

### Prerequisites

- Python 3.10 or higher
- Git
- Virtual environment tool (`venv` or `conda`)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Abdul-Moiz-Shehzad/MultiModal-Movie-Genre-Classification.git
cd MultiModal-Movie-Genre-Classification
```

### Step 2: Create and Activate Virtual Environment

**Windows (Command Prompt / PowerShell):**
```powershell
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

Install all required packages and register the local package in editable mode:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Download NLTK Corpora

If running for the first time, download the required NLTK datasets:

```bash
python -c "import nltk; nltk.download(['wordnet', 'stopwords', 'punkt', 'punkt_tab'])"
```

### Step 5: Required Directory Structure for Artifacts

Ensure the expected directories exist for saved model weights and image uploads:

```bash
mkdir -p data/processed/text
mkdir -p data/processed/images
mkdir -p static/uploads
mkdir -p temp
```

> **Note**: Pretrained model artifacts (`final_model_v3.keras`, `mpnet_encoder/`, `image_model.keras`, `label_encoder.pkl`) must be placed in their respective `data/processed/` directories as configured in `configs/base_config.yaml`.

---

## Usage

### 1. Web Application

Launch the Flask server:

```bash
python app.py
```

Open a browser and navigate to:
```
http://127.0.0.1:5000/
```

Features:
- Submit text description, movie poster image, or both simultaneously.
- Live client-side poster image preview.
- Spinner loader for asynchronous inference feedback.
- Detailed probability distribution per genre with dynamic percentage breakdown.
- Form reset endpoint to clear state.

---

### 2. Interactive CLI Scripts

#### Text Classifier CLI
```bash
python scripts/test_text_prediction.py
```
Prompts for raw plot synopses and returns predicted genre labels.

#### Image Classifier CLI
```bash
python scripts/test_image_prediction.py
```
Prompts for local image file paths and outputs predicted genres with full class probability breakdowns.

#### Multimodal Fusion Classifier CLI
```bash
python scripts/test_fusion_prediction.py
```
Accepts an image path and plot description, returning combined predictions.

#### Automated Test with LIME Explanation
```bash
python scripts/test_fusion2_prediction.py
```
Executes dynamic confidence-weighted fusion on a sample input and generates `lime_text_explanation.html`.

---

### 3. Programmatic Python API

#### Text Classification
```python
from src.models.text_model.classifier import TextGenreClassifier

text_clf = TextGenreClassifier()
plot = "A secret agent embarks on a dangerous mission across Europe to stop an international rogue syndicate."

prediction = text_clf.predict(plot)
probabilities = text_clf.predict_proba(plot)

print(f"Prediction: {prediction}")
print(f"Probabilities: {probabilities}")
```

#### Image Classification
```python
import cv2
from src.models.image_model.classifier import ImageGenreClassifier

image_clf = ImageGenreClassifier()
poster_bgr = cv2.imread("path/to/poster.jpg")

prediction = image_clf.predict(poster_bgr)
probabilities = image_clf.predict_proba(poster_bgr)

print(f"Prediction: {prediction}")
print(f"Probabilities: {probabilities}")
```

#### Multimodal Late Fusion
```python
import cv2
from src.models.fusion_model.classifier import FusionGenreClassifier

# Static fusion
fusion_static = FusionGenreClassifier(dynamic_weights=False)

# Dynamic confidence-weighted fusion
fusion_dynamic = FusionGenreClassifier(dynamic_weights=True)

poster_bgr = cv2.imread("path/to/poster.jpg")
plot = "Two strangers meet on a train across Europe and fall deeply in love before sunrise."

# Joint prediction
pred = fusion_dynamic.predict(raw_text=plot, raw_image=poster_bgr)
probs = fusion_dynamic.predict_proba(raw_text=plot, raw_image=poster_bgr)

print(f"Predicted Genre: {pred}")
print(f"Probabilities: {probs}")
```

#### Interpretability with LIME
```python
from src.models.fusion_model.classifier import FusionGenreClassifierWithLIME

lime_clf = FusionGenreClassifierWithLIME(dynamic_weights=True)
plot = "A haunted house traps a family during a violent thunderstorm with paranormal occurrences."

explanation = lime_clf.explain_text(plot)
explanation.save_to_file("explanation_output.html")
```

---

## Experimental Notebooks and Research

The `notebooks/` directory contains complete exploratory data analysis, architectural experiments, and evaluation benchmarks:

| Notebook | Focus Area | Key Methodologies & Architectures Evaluated |
| :--- | :--- | :--- |
| `eda_text_analysis.ipynb` | Text Modeling & Baselines | TF-IDF, Word2Vec, SentenceTransformers (`all-mpnet-base-v2`), Logistic Regression, Random Forest, Gaussian Naive Bayes. |
| `eda_images_analysis.ipynb` | Visual Modeling | Basic CNN, ResNet50, MobileNetV2, Attention Layer, EfficientNetB0, EfficientNetB3 (Two-Phase Fine-Tuning + Mixup), EfficientNetB5. |
| `research.ipynb` | Augmentation & Hybrid Models | NLTK Synonym Replacement & Deletion, SMOTE, Class Weighting via `bincount`, Penultimate Feature Stacking, Bi-LSTM + MultiHeadAttention. |
| `image_data.ipynb` | Dataset Verification | Scraping, dataset parsing, and poster integrity checks. |

---

## Testing

Run unit tests using `pytest`:

```bash
pytest tests/
```

Test coverage includes verification of text tokenization, lowercasing, non-alphabetic filtering, and WordNet lemmatization workflows.

---

## Technical Stack

- **Machine Learning & Deep Learning**: TensorFlow 2.19, Keras, Sentence-Transformers, Scikit-Learn, Imbalanced-Learn
- **Computer Vision**: OpenCV (`cv2`)
- **Natural Language Processing**: NLTK, Gensim
- **Explainability**: LIME (`lime_text`)
- **Web Application**: Flask, Werkzeug, Bootstrap 5, HTML5/CSS3
- **Configuration & Utilities**: PyYAML, Joblib, Pandas, Matplotlib

---

## License

This project is open source and available under the standard MIT License.
