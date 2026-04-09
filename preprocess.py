import re
import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from config import TRAIN_PATH, TEST_PATH, TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, RANDOM_STATE, TEST_SIZE


def clean_prompt(text):
    text = re.sub(r'\[INST\].*?\[/INST\]', '', text, flags=re.DOTALL)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_features(df):
    df = df.copy()
    df['Answer'] = df['Answer'].fillna("")
    df['clean_prompt'] = df['Prompt'].apply(clean_prompt)
    df['combined'] = df['clean_prompt'] + " [SEP] " + df['Answer']
    df['answer_len'] = df['Answer'].str.len()
    df['prompt_len'] = df['clean_prompt'].str.len()
    df['answer_word_count'] = df['Answer'].apply(lambda x: len(x.split()))
    df['answer_sentence_count'] = df['Answer'].apply(lambda x: len(re.split(r'[.!?]+', x)))
    df['has_no_answer'] = df['Answer'].str.lower().str.contains('no answer|i don|i cannot|unknown', na=False).astype(int)
    df['starts_with_step'] = df['Answer'].str.lower().str.startswith('step').astype(int)
    return df


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    train.dropna(subset=['Answer'], inplace=True)
    train = extract_features(train)
    test = extract_features(test)
    return train, test


def build_features(train, test):
    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        stop_words='english',
        sublinear_tf=True
    )

    X_tfidf_train = tfidf.fit_transform(train['combined'])
    X_tfidf_test = tfidf.transform(test['combined'])

    numeric_cols = ['answer_len', 'prompt_len', 'answer_word_count',
                    'answer_sentence_count', 'has_no_answer', 'starts_with_step']

    X_num_train = csr_matrix(train[numeric_cols].values.astype(np.float32))
    X_num_test = csr_matrix(test[numeric_cols].values.astype(np.float32))

    X_train_full = hstack([X_tfidf_train, X_num_train])
    X_test_full = hstack([X_tfidf_test, X_num_test])

    y = train['Target'].values

    return X_train_full, X_test_full, y, tfidf


def split_and_resample(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    return X_train_res, X_val, y_train_res, y_val
