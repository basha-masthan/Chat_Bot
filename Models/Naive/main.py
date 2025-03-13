import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
with open("data.json", "r") as file:
    data = json.load(file)

# Prepare training data
X_train, y_train = [], []
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        X_train.append(pattern)
        y_train.append(intent["tag"])

# Convert text to feature vectors
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)

# Train Naive Bayes model
clf = MultinomialNB()
clf.fit(X_train_counts, y_train)

# Function to predict user input

while True:
    def get_response(user_input):
        if user_input == "bye":
            print("Bye..")
            exit()
        user_input_vectorized = vectorizer.transform([user_input])
        predicted_tag = clf.predict(user_input_vectorized)[0]
        
        for intent in data["intents"]:
            if intent["tag"] == predicted_tag:
                return np.random.choice(intent["responses"])
        return "I don't understand."

    # Example usage
    val = input("Enter your Query: ")    
    print(get_response(val))  # Output: Future Bound Tech
