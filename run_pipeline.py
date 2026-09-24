"""Reproducible books.toscrape.com -> clean CSV -> normalized SQLite workflow."""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = "https://books.toscrape.com/"
RATE_GBP_TO_INR = 105.50
ROOT = Path(__file__).parent
OUT = ROOT / "output"
RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def soup(url: str) -> BeautifulSoup:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def scrape_books(pages: int = 5) -> pd.DataFrame:
    """Collect 100 rows from the first five catalogue pages."""
    rows = []
    for page in range(1, pages + 1):
        listing = soup(f"{BASE}catalogue/page-{page}.html")
        for card in listing.select("article.product_pod"):
            detail_url = BASE + "catalogue/" + card.h3.a["href"].replace("../../../", "")
            detail = soup(detail_url)
            category = detail.select_one("ul.breadcrumb li:nth-of-type(3)").get_text(strip=True)
            rows.append({
                "title": card.h3.a["title"],
                "price_as_listed": card.select_one(".price_color").get_text(strip=True),
                "star_rating": card.select_one(".star-rating")["class"][1],
                "availability": card.select_one(".availability").get_text(" ", strip=True),
                "category": category,
            })
    return pd.DataFrame(rows)


def clean_books(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["price_gbp"] = pd.to_numeric(df["price_as_listed"].str.replace(r"[^0-9.]", "", regex=True), errors="coerce")
    df["rating"] = df["star_rating"].map(RATING_MAP)
    df["in_stock"] = df["availability"].str.contains("in stock", case=False, na=False)
    # Median is stable for the two numeric fields; unusable identifying fields are removed.
    for column in ["price_gbp", "rating"]:
        df[column] = df[column].fillna(df[column].median())
    df = df.dropna(subset=["title", "category"]).drop_duplicates(subset=["title", "category"])
    df["rating"] = df["rating"].astype(int)
    df["in_stock"] = df["in_stock"].astype(bool)
    df["price_inr"] = (df["price_gbp"] * RATE_GBP_TO_INR).round(2)
    return df[["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]]


def create_database(df: pd.DataFrame, db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.executescript("""
        PRAGMA foreign_keys = ON;
        DROP TABLE IF EXISTS books; DROP TABLE IF EXISTS categories;
        CREATE TABLE categories (category_id INTEGER PRIMARY KEY, category_name TEXT UNIQUE NOT NULL);
        CREATE TABLE books (
          book_id INTEGER PRIMARY KEY, title TEXT NOT NULL, price_gbp REAL NOT NULL,
          price_inr REAL NOT NULL, rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
          in_stock INTEGER NOT NULL, category_id INTEGER NOT NULL,
          FOREIGN KEY(category_id) REFERENCES categories(category_id));
        """)
        for name in sorted(df.category.unique()):
            conn.execute("INSERT INTO categories(category_name) VALUES (?)", (name,))
        ids = dict(conn.execute("SELECT category_name, category_id FROM categories"))
        records = [(r.title, r.price_gbp, r.price_inr, r.rating, int(r.in_stock), ids[r.category]) for r in df.itertuples()]
        conn.executemany("INSERT INTO books(title,price_gbp,price_inr,rating,in_stock,category_id) VALUES (?,?,?,?,?,?)", records)


QUERIES = {
    "where": "SELECT title, price_gbp FROM books WHERE rating >= 4;",
    "order_limit": "SELECT title, price_inr FROM books ORDER BY price_inr DESC LIMIT 10;",
    "distinct": "SELECT DISTINCT rating FROM books ORDER BY rating;",
    "between": "SELECT title, price_gbp FROM books WHERE price_gbp BETWEEN 20 AND 30;",
    "join": "SELECT c.category_name, b.title, b.rating, b.price_inr FROM books b JOIN categories c ON b.category_id=c.category_id ORDER BY b.rating DESC, c.category_name LIMIT 10;",
}


def run_queries(db_path: Path, clean: pd.DataFrame) -> None:
    with sqlite3.connect(db_path) as conn, (OUT / "query_results.txt").open("w", encoding="utf-8") as log:
        for name, sql in QUERIES.items():
            result = pd.read_sql(sql, conn)
            log.write(f"\n-- {name}\n{sql}\n{result.to_string(index=False)}\n")
        sql_join = pd.read_sql(QUERIES["join"], conn)
        categories = pd.read_sql("SELECT * FROM categories", conn)
    pandas_join = (clean.merge(categories, left_on="category", right_on="category_name")
                   [["category_name", "title", "rating", "price_inr"]]
                   .sort_values(["rating", "category_name"], ascending=[False, True]).head(10).reset_index(drop=True))
    sql_join = sql_join.reset_index(drop=True)
    with (OUT / "query_results.txt").open("a", encoding="utf-8") as log:
        log.write(f"\n-- merge equivalent\n{pandas_join.to_string(index=False)}\nMatches SQL: {sql_join.equals(pandas_join)}\n")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    cleaned = clean_books(scrape_books())
    if len(cleaned) < 60:
        raise RuntimeError(f"Expected at least 60 books, got {len(cleaned)}")
    cleaned.to_csv(OUT / "books_cleaned.csv", index=False)
    database = OUT / "zepto_catalog.db"
    create_database(cleaned, database)
    run_queries(database, cleaned)
    print(f"Created {database} from {len(cleaned)} clean rows at GBP/INR rate {RATE_GBP_TO_INR}.")
