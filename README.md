# Sentiment Analyser

A machine learning web app that classifies text as **positive**, **negative**, or **neutral** with a confidence score.

---

## What It Does

- Trained on 12,662 labelled samples: 2,000 IMDB movie reviews + 10,662 Rotten Tomatoes snippets (via NLTK)
- TF-IDF vectorization (up to 50k features, unigrams + bigrams)
- Logistic Regression classifier — **78.5% accuracy** on a held-out 20% test split of that movie-review data
- The model is binary (positive / negative). "Neutral" is shown when its confidence is below 60%
- FastAPI REST API with `/analyze` endpoint
- Clean web UI — paste any text, get an instant result

> **Scope:** trained on movie reviews, so it is not reliable on other domains. The
> [stock-sentiment](https://github.com/TJDEBUGGED/stock-sentiment) project measures how badly this
> kind of model transfers to financial headlines, and fixes it with finance training data.

## Tech Stack

`Python` · `scikit-learn` · `NLTK` · `FastAPI` · `Uvicorn`

---

## Run Locally

```bash
# 1. Clone and install dependencies
git clone https://github.com/TJDEBUGGED/sentiment-analyser
cd sentiment-analyser
pip install -r requirements.txt

# 2. Train the model (one-time; downloads the NLTK corpora on first run)
python train.py

# 3. Start the API
uvicorn app.main:app --reload

# 4. Open http://localhost:8000
```

## API

```bash
POST /analyze
Content-Type: application/json

{ "text": "This movie was absolutely fantastic!" }
```

```json
{
  "sentiment": "positive",
  "confidence": 0.9732,
  "scores": { "negative": 0.0268, "positive": 0.9732 },
  "text_preview": "This movie was absolutely fantastic!"
}
```

---

## Project Structure

```
sentiment-analyser/
├── train.py          # Train and save the model
├── model/            # Saved vectorizer + classifier (generated)
├── app/
│   ├── main.py       # FastAPI app
│   └── predictor.py  # Model inference
├── frontend/
│   └── index.html    # Web UI
├── requirements.txt
└── Dockerfile
```

---

Built by [Harry Tijani](https://github.com/TJDEBUGGED)
