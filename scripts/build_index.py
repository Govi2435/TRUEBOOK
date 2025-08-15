import os
import numpy as np
import pandas as pd

from recommender.content_based import ContentBasedRecommender
from search.ann_index import AnnIndex
import yaml


def load_config(path: str = "config/config.yaml") -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    config = load_config()
    books_csv = config.get("paths", {}).get("books_csv", "sample_data/books_sample.csv")
    books = pd.read_csv(books_csv)
    cbr = ContentBasedRecommender(config)
    cbr.fit(books)
    tfidf = cbr.tfidf_matrix
    if tfidf is None:
        print("No TF-IDF matrix; nothing to index")
        return
    X = tfidf.astype("float32").toarray()
    index = AnnIndex(dim=X.shape[1], engine=config.get("ann", {}).get("engine", "auto"))
    index.build(X)
    out_dir = config.get("paths", {}).get("ann_index_dir", "data/indices")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "tfidf_index_vectors.npy"), X)
    print(f"[ok] built ANN vectors: {X.shape}")


if __name__ == "__main__":
    main()