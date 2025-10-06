from typing import Iterable
import pandas as pd
import nltk

# Ensure VADER is available (first run downloads)
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
except LookupError:
    nltk.download("vader_lexicon")
    from nltk.sentiment import SentimentIntensityAnalyzer


def add_vader_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sia = SentimentIntensityAnalyzer()

    # Choose text field per row type
    def _text(row) -> str:
        if row.get("type") == "submission":
            return " ".join([str(row.get("title","")), str(row.get("selftext",""))]).strip()
        return str(row.get("body",""))

    texts: Iterable[str] = (_text(r) for _, r in df.iterrows())
    scores = [sia.polarity_scores(t or "") for t in texts]
    # scores are dicts: {'neg':..,'neu':..,'pos':..,'compound':..}
    sdf = pd.DataFrame(scores)
    out = pd.concat([df.reset_index(drop=True), sdf], axis=1)
    # simple label for convenience
    out["label"] = out["compound"].apply(lambda c: "pos" if c >= 0.05 else ("neg" if c <= -0.05 else "neu"))
    return out


def summarize_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    # group by subreddit and content type
    if df.empty:
        return df
    keys = ["subreddit","type","label"]
    summary = (df
               .groupby(keys, dropna=False)
               .size()
               .reset_index(name="count"))
    total = summary.groupby(["subreddit","type"])["count"].transform("sum")
    summary["share"] = (summary["count"] / total).round(3)
    return summary.sort_values(["subreddit","type","label"])
