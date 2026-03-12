import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier

model_name = input("Enter model name: ")

def saveModel(model_name, clf):
    #save the model to .sav file
    with open(model_name+'.sav', 'wb') as model_file:
        pickle.dump(clf, model_file)

def saveAccuracy(model_name,accuracy):
    #save to text file
    text_file = open(model_name+"_accuracy.txt", "w")
    text_file.write(str(accuracy))
    text_file.close()

def saveClassification(model_name,classification_report_result):
    #save to text file
    text_file = open(model_name+"_classification_report.txt", "w")
    text_file.write(str(classification_report_result))
    text_file.close()

# Create a decision tree classifier
#clf = DecisionTreeClassifier()
#clf = RandomForestClassifier()
#clf = BaggingClassifier()
clf = SGDClassifier()


# Load X_test.sav
with open('X_test.sav', 'rb') as file:
    X_test = pickle.load(file)
X_test = pd.DataFrame(X_test)

#Load X_train.sav
with open('X_train.sav', 'rb') as file:
    X_train = pickle.load(file)
X_train = pd.DataFrame(X_train)

#Load y_train.sav
with open('y_train.sav', 'rb') as file:
    y_train = pickle.load(file)
y_train = pd.DataFrame(y_train)

# Load y_test.sav
with open('y_test.sav', 'rb') as file:
    y_test = pickle.load(file)
y_test = pd.DataFrame(y_test)


# Train the classifier on the training data
clf.fit(X_train, y_train)
# Calls the save model function
saveModel(model_name,clf)

# Make predictions on the test data
y_pred = clf.predict(X_test)

#code that creates a confusion matrix based on a specific model
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
display_labels=clf.classes_)
disp.plot(values_format=".3f")
plt.title(f'Confusion matrix of the {model_name} classifier')
plt.xlabel('Predicted')
plt.xticks(rotation=45)
plt.ylabel('True')
#PNG saved with model name
plt.savefig(f"ConfusionMatrix_{model_name}.png")
plt.close()


print(confusion_matrix(y_test, y_pred))

# Generate and print a classification report
classification_report_result = metrics.classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report_result)
saveClassification(model_name,classification_report_result)

# Evaluate the classifier's performance
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
saveAccuracy(model_name,accuracy)

# Print X_train.sav
print("x_test dataframe:")
print(X_test.head())