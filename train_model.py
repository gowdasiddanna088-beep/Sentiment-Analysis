import os
import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

# 1. Load the dataset
print("Loading dataset...")

df = pd.read_csv("dataset/IMDB Dataset.csv")

# 2. Remove missing values and duplicates
df = df.dropna(subset=["review", "sentiment"])
df = df.drop_duplicates(subset=["review"])

# 3. Prepare the text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


df["review"] = df["review"].apply(clean_text)

# 4. Separate input and output
X = df["review"]
y = df["sentiment"].str.lower().str.strip()

# Keep only positive and negative labels
valid_labels = y.isin(["positive", "negative"])
X = X[valid_labels]
y = y[valid_labels]

# 5. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# 6. Convert text into numerical features
print("\nConverting text using TF-IDF...")

vectorizer = TfidfVectorizer(
    max_features=20000,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# 7. Train the machine learning model
print("\nTraining Logistic Regression model...")

model = LogisticRegression(max_iter=1000)

model.fit(X_train_vectorized, y_train)

# 8. Evaluate the model
print("\nEvaluating model...")

y_pred = model.predict(X_test_vectorized)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 9. Save the trained model and vectorizer
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/sentiment_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("\nModel saved successfully!")
print("Location: model/sentiment_model.pkl")
print("Location: model/tfidf_vectorizer.pkl")