import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
import matplotlib.pyplot as plt

# Load the datasets and convert to pandas dataframe to read csv files
#Data was pre-split and processed from preprocess_amazon_script.py
train = pd.read_csv("data/preprocessed/train.csv")
val = pd.read_csv("data/preprocessed/val.csv")
test = pd.read_csv("data/preprocessed/test.csv") 

#Sample to run code to check output; whole dataset takes much longer
    #train_1 = pd.read_csv("data/preprocessed/train.csv")
    #val_1 = pd.read_csv("data/preprocessed/val.csv")
    #test_1 = pd.read_csv("data/preprocessed/test.csv")

    #train = train_1.sample(2000, random_state = 42)
    #val = val_1.sample(2000, random_state = 42)
    #test = test_1.sample(2000, random_state = 42)

# Creating x and y for val, train, and test datasets
x_train = train["Text"]
y_train = train["Label"]

x_val = val["Text"]
y_val = val["Label"]

x_test = test["Text"]
y_test = test["Label"]

#TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

#Vectorizing the data
vectorized_x_train = vectorizer.fit_transform(x_train)
vectorized_x_test = vectorizer.transform(x_test)
vectorized_x_val = vectorizer.transform(x_val)


# Random Forest Model Initialization with 50 trees
model = RandomForestClassifier(n_estimators = 50, n_jobs = -1) #Using all cores to run the mode, can be modified

# Training the Random Forest Model
model.fit(vectorized_x_train, y_train)

# Predictions
val_predict = model.predict(vectorized_x_val)
test_predict = model.predict(vectorized_x_test)

# Tracking errors across 50 trees
train_errors = []
val_errors = []
test_errors = []

#for n in range(1, 50):
#    rf = RandomForestRegressor(n, n_jobs = -1) #Using all cores to run the mode, can be modified
#    rf.fit(vectorized_x_train, y_train)
#    
#    train_pred = rf.predict(vectorized_x_train)
#    val_pred = rf.predict(vectorized_x_val)
#    test_pred = rf.predict(vectorized_x_test)
#    
#    train_errors.append(mean_squared_error(y_train, train_pred))
#    val_errors.append(mean_squared_error(y_val, val_pred))
#    test_errors.append(mean_squared_error(y_test, test_pred))
    

#Printing val and test accuracies
print("Validation accuracy:", accuracy_score(y_val, val_predict))
print("Test accuracy:", accuracy_score(y_test, test_predict))

#Precision, Recall, F1-score, support, overall accuracy, macro average, and weighted average.
print("Classification report:", classification_report(y_test, test_predict))

#Plotting the errors to show loss across the number of trees
plt.plot(range(1, len(train_errors) + 1), train_errors, 'Training Error')
plt.plot(range(1, len(test_errors) + 1), test_errors, 'Testing Error')
plt.plot(range(1, len(val_errors) + 1), val_errors, 'Validation Error')
plt.xlabel('Number of Trees')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()
