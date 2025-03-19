from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re

nltk.download(['wordnet', 'stopwords', 'punkt_tab'], quiet=True)
WNL = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text: str, return_lst: bool = True) -> list:
    """
    Preprocess text by cleaning, tokenizing, and lemmatizing
    
    Args:
        text: Input text string
        return_lst: Whether to return tokens as list or joined string
    
    Returns:
        Processed tokens (list or string)
    """
    if not isinstance(text, str) or not text.strip():
        return [] if return_lst else ""

    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower().strip()
    tokens = word_tokenize(text)
    processed_tokens = [
        WNL.lemmatize(token) 
        for token in tokens
        if token not in stop_words
    ]
    
    return processed_tokens if return_lst else ' '.join(processed_tokens)