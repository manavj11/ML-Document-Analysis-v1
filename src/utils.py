# src/utils.py

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    """
    Tokenizes, removes punctuation/numbers, removes stop words, and lemmatizes 
    the input text for cleaner feature engineering.
    
    Args:
        text (str): A single medical abstract or document.
        
    Returns:
        str: The cleaned and tokenized text string.
    """
    # Remove punctuation/numbers and convert to lower case
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    
    tokens = text.split()
    
    # Remove common English stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization (reducing words to their base form)
    # Note: Requires 'wordnet' to be downloaded (see notebook setup)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(tokens)

def display_topics(model, feature_names, no_top_words):
    """
    Prints the top words for each discovered topic and attempts to assign a label.
    
    Args:
        model (sklearn.decomposition.LatentDirichletAllocation): The trained LDA model.
        feature_names (np.array): Array of unique words (features) from the vectorizer.
        no_top_words (int): The number of top words to display for each topic.
    """
    print(f"\n✨ Discovered Topics (Top {no_top_words} Words):")
    print("---------------------------------------------")
    for topic_idx, topic in enumerate(model.components_):
        # Get the words with the highest weights for this topic
        top_words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        
        # --- Manual Topic Labeling Heuristic ---
        label = "Unknown"
        # Based on the expected sample data topics:
        if any(word in top_words for word in ["cancer", "tumor", "malignant", "oncology"]):
            label = "Oncology"
        elif any(word in top_words for word in ["heart", "cardiac", "statin", "aspirin"]):
            label = "Cardiology"
        elif any(word in top_words for word in ["drug", "trial", "vaccine", "fda"]):
            label = "Pharmacology/Trials"
        elif any(word in top_words for word in ["joint", "knee", "hip", "bone", "ligament"]):
            label = "Orthopedics"
        
        print(f"Topic {topic_idx + 1} ({label}): {' '.join(top_words)}")