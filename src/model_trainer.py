import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# --- Configuration ---
N_TOPICS = 4  # We expect 4 main topics: Cardio, Oncology, Pharma, Infectious Disease
N_TOP_WORDS = 8 # Number of words to display per topic
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
        if "cancer" in top_words or "tumor" in top_words or "therapy" in top_words:
            label = "Oncology"
        elif "heart" in top_words or "cardiac" in top_words or "artery" in top_words:
            label = "Cardiology"
        elif "drug" in top_words or "trial" in top_words or "patient" in top_words:
            label = "Pharmacology/Trials"
        elif "viral" in top_words or "vaccine" in top_words or "fever" in top_words:
            label = "Infectious Disease"

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
    # 
    lda = LatentDirichletAllocation(
        n_components=N_TOPICS,
        max_iter=5,
        learning_method='online',
        random_state=42
    )
    lda.fit(data_vectorized)
    
    print("✅ Model training complete.")
    
    # 4. Interpretation
    display_topics(lda, feature_names, N_TOP_WORDS)
    
    # Show document-topic distribution for the first document
    doc_topic_distribution = lda.transform(data_vectorized)
    print("\n--- Example Document-Topic Distribution ---")
    print(f"Document 1: '{corpus[0]}'")
    # Find the index of the topic with the highest probability
    main_topic_idx = np.argmax(doc_topic_distribution[0])
    main_topic_prob = doc_topic_distribution[0, main_topic_idx]
    
    print(f"The model assigns this document primarily to Topic {main_topic_idx + 1} with a weight of {main_topic_prob:.2f}.")
    print("---------------------------------------------")