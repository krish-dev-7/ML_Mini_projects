from flask import Flask, request, jsonify
import joblib
import re

# Load model & vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Init app
app = Flask(__name__)


# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuations
    text = re.sub(r'\d+', '', text)  # remove numbers
    return text


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    message = data.get("message", "")

    cleaned = clean_text(message)
    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]

    class_probs = dict(zip(model.classes_, probabilities))

    # Debug print
    print("Classes:", model.classes_)
    print("Probs:", class_probs)

    # Use dynamic keys
    response = {'prediction': str(prediction), 'confidence': {}}
    for cls, prob in class_probs.items():
        response['confidence'][str(cls)] = round(float(prob) * 100, 2)

    return jsonify(response)


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
