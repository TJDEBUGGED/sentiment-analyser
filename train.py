"""
train.py — Train a sentiment classifier and save it to model/.
Run once: python train.py

Data (both via NLTK, no HuggingFace needed):
  - movie_reviews:      2,000 full-length IMDB movie reviews (1,000 pos / 1,000 neg)
  - sentence_polarity:  10,662 short Rotten Tomatoes snippets (5,331 pos / 5,331 neg)
Short snippets look much more like news headlines than full reviews do,
which matters when we reuse this model on stock headlines.
"""

import os
import pickle

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

for corpus in ("movie_reviews", "sentence_polarity"):
    nltk.download(corpus, quiet=True)

from nltk.corpus import movie_reviews, sentence_polarity  # noqa: E402


def load_data():
    texts, labels = [], []  # label: 0 = negative, 1 = positive

    for fid in movie_reviews.fileids():
        texts.append(movie_reviews.raw(fid))
        labels.append(1 if fid.startswith("pos/") else 0)

    for cat, label in (("neg", 0), ("pos", 1)):
        for sent in sentence_polarity.sents(categories=cat):
            texts.append(" ".join(sent))
            labels.append(label)

    return texts, labels


print("Loading data...")
texts, labels = load_data()
print(f"   Total samples: {len(texts):,}")

# Hold out 20% the model never sees during training, so accuracy is honest.
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)
print(f"   Train: {len(train_texts):,} | Test: {len(test_texts):,}")

print("\nVectorizing text with TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=50_000,
    ngram_range=(1, 2),   # unigrams + bigrams
    sublinear_tf=True,    # log-scale term frequency
    min_df=2,
)
X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)

print("\nTraining Logistic Regression classifier...")
model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
model.fit(X_train, train_labels)

print("\nEvaluating on held-out test set...")
preds = model.predict(X_test)
acc = accuracy_score(test_labels, preds)
print(f"   Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
print("\n" + classification_report(test_labels, preds, target_names=["Negative", "Positive"]))

os.makedirs("model", exist_ok=True)
with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open("model/classifier.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved to model/")
