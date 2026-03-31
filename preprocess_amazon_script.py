"""
This script downloads the Kaggle Amazon product reviews and cleans them up.
It removes neutral reviews so we only have the 'Good' or 'Bad' ones, picks 50,000 of them, 
and splits them into training, validation, and testing sets. 
It saves two versions: one with the original text and one that's cleaned for AI training.
They are saved in the Data folder.

"""
import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import re
import os
import kagglehub
from functools import partial

def map_score_to_label(score):
    """maps product ratings to binary sentiment: 4-5 is positive (1), 1-2 is negative (0)"""
    return 1 if score > 3 else 0

def load_and_prepare_data(sample_size=50000):
    print("Downloading dataset using kagglehub...")
    path = kagglehub.dataset_download("arhamrumi/amazon-product-reviews")
    csv_path = os.path.join(path, "Reviews.csv")
    print("Path to dataset files:", path)
    
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: '{csv_path}' not found in the downloaded dataset.")
        return None
    
    # ensure score and text columns exist
    if 'Score' not in df.columns or 'Text' not in df.columns:
        print("Required columns 'Score' and/or 'Text' not found in the dataset.")
        return None

    print(f"Total reviews loaded: {len(df)}")
    
    # filter out neutral reviews (score == 3)
    print("Filtering out neutral reviews (Score == 3)...")
    df = df[df['Score'] != 3].copy()
    
    # create binary labels: positive (4, 5) -> 1, negative (1, 2) -> 0
    df['Label'] = df['Score'].apply(map_score_to_label)
    
    # drop missing text rows
    df = df.dropna(subset=['Text'])
    
    print(f"Reviews after filtering neutral: {len(df)}")
    
    # limit dataset to sample_size, maintaining class distribution
    if len(df) > sample_size:
        print(f"Sampling {sample_size} reviews...")
        # stratified sampling based on 'label' to maintain pos/neg ratio
        df, _ = train_test_split(df, train_size=sample_size, stratify=df['Label'], random_state=42)
    
    # keep only text and label
    df = df[['Text', 'Label']]
    print(f"Class distribution in sample:\n{df['Label'].value_counts(normalize=True)}")
    return df

def preprocess_text(text, stop_words, lemmatizer):
    # lowercase
    text = text.lower()
    # remove html tags (often seen in amazon reviews like <br />)
    text = re.sub(r'<.*?>', ' ', text)
    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # remove numbers
    text = re.sub(r'\d+', '', text)
    
    # tokenize, remove stopwords, and lemmatize
    words = text.split()
    processed_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(processed_words)

def main():
    # setup nltk
    print("Downloading required NLTK data...")
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
        
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    except:
        pass # in case wordnet errors out on some environments, we can gracefully catch it, but usually it works.
        
    # load and prepare raw data
    df = load_and_prepare_data(50000)
    if df is None:
        return
    
    # shuffle and split the data (80/10/10)
    print("\nShuffling and splitting data into 80% train, 10% validation, 10% test...")
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)

    train_df = df.iloc[:40000]
    val_df = df.iloc[40000:45000]
    test_df = df.iloc[45000:50000]
    
    # create output directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/preprocessed', exist_ok=True)
    
    # save version a: raw
    print("Saving RAW text versions (for LLM)...")
    train_df.to_csv('data/raw/train.csv', index=False)
    val_df.to_csv('data/raw/val.csv', index=False)
    test_df.to_csv('data/raw/test.csv', index=False)
    
    # setup for preprocessing
    print("\nPreprocessing text data (lowercase, no punctuation, no stopwords, lemmatized)...")
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # create a fixed version of preprocess_text with rules already applied
    clean_text = partial(preprocess_text, stop_words=stop_words, lemmatizer=lemmatizer)
    
    # apply preprocessing
    print("Processing training data...")
    train_df_prep = train_df.copy()
    train_df_prep['Text'] = train_df_prep['Text'].astype(str).apply(clean_text)
    
    print("Processing validation data...")
    val_df_prep = val_df.copy()
    val_df_prep['Text'] = val_df_prep['Text'].astype(str).apply(clean_text)
    
    print("Processing test data...")
    test_df_prep = test_df.copy()
    test_df_prep['Text'] = test_df_prep['Text'].astype(str).apply(clean_text)
    
    # save version b: preprocessed
    print("Saving PREPROCESSED text versions (for ML)...")
    train_df_prep.to_csv('data/preprocessed/train.csv', index=False)
    val_df_prep.to_csv('data/preprocessed/val.csv', index=False)
    test_df_prep.to_csv('data/preprocessed/test.csv', index=False)
    
    print("\nDone! Dataset successfully processed and saved to:")
    print(" - RAW (LLM format): data/raw/ {train, val, test}.csv")
    print(" - PREPROCESSED (ML format): data/preprocessed/ {train, val, test}.csv")

if __name__ == '__main__':
    main()
