import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Učitavanje podataka
df = pd.read_csv("products.csv")
df.columns = df.columns.str.strip()

# Ulaz i izlaz
X = df["Product Title"]
y = df["Category Label"]

# Podjela podataka
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Kreiranje modela
log_model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=10000)),
    ("clf", LogisticRegression(max_iter=1000))
])

# Treniranje
log_model.fit(X_train, y_train)

# Čuvanje modela
joblib.dump(log_model, "product_classifier.pkl")

print("Model je uspješno sačuvan.")
