import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Load the datasets
train = pd.read_csv("data/preprocessed/train.csv")
val = pd.read_csv("data/preprocessed/val.csv")
test = pd.read_csv("data/preprocessed/test.csv")

#**Delete later, checks if loaded properly.
print(train.head())

# Split text and labels
X_train = train["Text"]
y_train = train["Label"]

X_val = val["Text"]
y_val = val["Label"]

X_test = test["Text"]
y_test = test["Label"]

# Convert text into numbers using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)

# Create the SVM model
model = LinearSVC()

# Train the model using the training data
model.fit(X_train_tfidf, y_train)

# Predict on validation data
val_predictions = model.predict(X_val_tfidf)

print("Validation Accuracy:")
print(accuracy_score(y_val, val_predictions))

# Predict on test data
test_predictions = model.predict(X_test_tfidf)

print("Test Accuracy:")
print(accuracy_score(y_test, test_predictions))

print("Classification Report:")
print(classification_report(y_test, test_predictions))