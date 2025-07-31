from flask import Flask, render_template, request
import pickle, json, random

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

with open("intents.json") as file:
    intents = json.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    user_text = request.args.get("msg")
    X = vectorizer.transform([user_text])
    tag = model.predict(X)[0]

    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "I don't understand."

if __name__ == "__main__":
    app.run(debug=True)
