import json, random, pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

with open("intents.json") as file:
    data = json.load(file)

X, y = [], []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        X.append(pattern)
        y.append(intent["tag"])

vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vectorized, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
