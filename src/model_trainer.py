import pandas as pd
import numpy as np
import re

import nltk
# Download required NLTK data if they don't exist
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    # Avoid hard failure; warn so user can manually install if needed.
    print(f"Warning: Could not automatically download NLTK data. Error: {e}")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# --- Configuration ---
N_TOPICS = 8  # We expect 8 main topics: Cardio, Oncology, Diseases, Treatment
# Increased for better topic separation
N_TOP_WORDS = 4 # Number of words to display per topic
FILE_PATH = 'data/medical_corpus.txt'

# --- 1. Data Cleaning and Preprocessing ---
def preprocess_text(text):
    """Tokenizes, removes stop words, and lemmatizes the input text."""
    # Remove punctuation/numbers and convert to lower case
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    
    tokens = text.split()
    
    # Remove common English stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization (reducing words to their base form)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return " ".join(tokens)

# --- 2. LDA Visualization/Interpretation Helper ---
def display_topics(model, feature_names, no_top_words):
    """Prints the top words for each discovered topic."""
    print(f"\n✨ Discovered Topics (Top {no_top_words} Words):")
    print("---------------------------------------------")
    for topic_idx, topic in enumerate(model.components_):
        # Get the words with the highest weights for this topic
        top_words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
        
        # Manually infer a potential label based on keywords
        label = "Unknown"
        if "system" in top_words:
            label = "Healthcare Process" 
        elif "cancer" in top_words:
            label = "Oncology/Cardiology"
        elif "therapeutic" in top_words:
            label = "Therapeutic Agents"
        elif "therapy" in top_words:
            label = "Treatment/Therapy"
        elif "screening" in top_words:
            label = "Screening/Diagnosis"
        elif "patient" in top_words:
            label = "Patient Care"
        elif "surgical" in top_words:
            label = "Surgical Procedures"
        elif "longterm" in top_words:
            label = "Strategy/Long-term"

        print(f"Topic {topic_idx + 1} ({label}): {' '.join(top_words)}")


# --- Main Pipeline Execution ---
if __name__ == "__main__":
    print("Starting Unsupervised Medical Topic Discovery...")
    
    # 1. Load Data
    try:
        with open(FILE_PATH, 'r') as f:
            corpus = f.read().splitlines()
        print(f"✅ Loaded {len(corpus)} documents from {FILE_PATH}.")
    except FileNotFoundError:
        # Fail fast with a clear message if the dataset path is incorrect.      
        print(f"❌ Error: Data file not found at {FILE_PATH}. Please check the path.")
        exit()

    # Apply Preprocessing
    cleaned_corpus = [preprocess_text(doc) for doc in corpus]
    
    # 2. Feature Engineering (Count Vectorization)
    print("\nPreparing features with Count Vectorizer...")
    # CountVectorizer converts text to a matrix of token counts
    vectorizer = CountVectorizer(max_df=0.95, min_df=2) # Ignore terms that are too frequent/infrequent
    data_vectorized = vectorizer.fit_transform(cleaned_corpus)
    feature_names = vectorizer.get_feature_names_out()
    
    print(f"✅ Created a feature matrix with {data_vectorized.shape[1]} unique words (features).")
    
    # 3. Topic Modeling (LDA Algorithm)
    print(f"\nTraining Latent Dirichlet Allocation (LDA) model with K={N_TOPICS} topics...")
    # LDA hyperparameters:
    # - n_components: number of latent topics
    # - max_iter: number of EM iterations (increase for better convergence)
    # - learning_method='online' is preferable for larger corpora / streaming
    # - random_state for reproducible results
    lda = LatentDirichletAllocation(
        n_components=N_TOPICS,
        max_iter=30, # Increase for better convergence
        learning_method='online',
        random_state=42
    )
    lda.fit(data_vectorized)
    
    print("✅ Model training complete.")
    
    # 4. Interpretation
    display_topics(lda, feature_names, N_TOP_WORDS)
    
    # Show document-topic distribution for a user selectable document number
    doc_topic_distribution = lda.transform(data_vectorized)

    print("\n--- Document-Topic Distribution (Select Document by Index) ---")
    # Prompt the user for an index (default to 10 if they hit enter)
    while True:
        try:
            user_input = input(f"Enter document index between 0 and {len(corpus)-1} (default 10): ").strip()
            doc_idx = 10 if user_input == "" else int(user_input)
            if 0 <= doc_idx < len(corpus):
                break
            print(f"Index out of range. Please enter a number between 0 and {len(corpus)-1}.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    print(f"Document {doc_idx}: '{corpus[doc_idx]}'")
    # Find the index of the topic with the highest probability
    main_topic_idx = np.argmax(doc_topic_distribution[doc_idx])
    main_topic_prob = doc_topic_distribution[doc_idx, main_topic_idx]
    
    print(f"The model assigns this document primarily to Topic {main_topic_idx + 1} with a weight of {main_topic_prob:.2f}.")
    
     # NEW: show top words + normalized probabilities for the selected main topic
    topic_weights = lda.components_[main_topic_idx]
    topic_probs = topic_weights / topic_weights.sum()
    top_indices = topic_weights.argsort()[:-N_TOP_WORDS - 1:-1]
    print(f"\nTop {N_TOP_WORDS} words for Topic {main_topic_idx + 1} - For interpreting weight (word : probability):")
    for i in top_indices:
        print(f"{feature_names[i]} : {topic_probs[i]:.4f}")
    
    print("---------------------------------------------")