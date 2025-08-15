import os
import pandas as pd

BOOKS = [
    {"book_id": "1", "title": "Akata Witch", "author": "Nnedi Okorafor", "country": "Nigeria", "language": "en", "genres": "Fantasy|YA", "themes": "Afrofuturism|Coming-of-age", "year": 2011, "avg_rating": 4.1, "rating_count": 25000, "description": "A Nigerian-American girl discovers she is a Leopard Person with magical powers."},
    {"book_id": "2", "title": "Zahrah the Windseeker", "author": "Nnedi Okorafor", "country": "Nigeria", "language": "en", "genres": "Fantasy", "themes": "Adventure|Coming-of-age", "year": 2005, "avg_rating": 4.0, "rating_count": 8000, "description": "Zahrah navigates a fantastical jungle with unique flora and fauna."},
    {"book_id": "3", "title": "Rosewater", "author": "Tade Thompson", "country": "Nigeria", "language": "en", "genres": "Science Fiction", "themes": "Afrofuturism|Aliens", "year": 2016, "avg_rating": 4.0, "rating_count": 12000, "description": "A city in Nigeria thrives around an alien biodome with psychic phenomena."},
    {"book_id": "4", "title": "Kafka on the Shore", "author": "Haruki Murakami", "country": "Japan", "language": "ja", "genres": "Magical Realism|Literary", "themes": "Surrealism|Coming-of-age", "year": 2002, "avg_rating": 4.1, "rating_count": 400000, "description": "A metaphysical odyssey blending reality and dreamscapes in Japan."},
    {"book_id": "5", "title": "1Q84", "author": "Haruki Murakami", "country": "Japan", "language": "ja", "genres": "Dystopian|Literary", "themes": "Parallel worlds|Mystery", "year": 2009, "avg_rating": 3.9, "rating_count": 350000, "description": "An alternate 1984 Tokyo where reality subtly shifts."},
]

INTERACTIONS = [
    {"user_id": "u1", "book_id": "1", "event_strength": 3.0},
    {"user_id": "u1", "book_id": "2", "event_strength": 1.0},
    {"user_id": "u2", "book_id": "1", "event_strength": 4.0},
    {"user_id": "u2", "book_id": "3", "event_strength": 2.0},
    {"user_id": "u3", "book_id": "4", "event_strength": 5.0},
    {"user_id": "u3", "book_id": "5", "event_strength": 1.0},
]


def ensure_sample_data() -> None:
    os.makedirs("sample_data", exist_ok=True)
    books_path = os.path.join("sample_data", "books_sample.csv")
    inter_path = os.path.join("sample_data", "user_interactions_sample.csv")
    if not os.path.exists(books_path):
        pd.DataFrame(BOOKS).to_csv(books_path, index=False)
        print(f"[ok] wrote {books_path}")
    else:
        print(f"[skip] {books_path} exists")
    if not os.path.exists(inter_path):
        pd.DataFrame(INTERACTIONS).to_csv(inter_path, index=False)
        print(f"[ok] wrote {inter_path}")
    else:
        print(f"[skip] {inter_path} exists")


if __name__ == "__main__":
    ensure_sample_data()