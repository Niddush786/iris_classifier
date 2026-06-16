
import os
from sklearn.datasets import load_iris # type: ignore

iris = load_iris()
X = iris.data
y = iris.target


os.makedirs("outputs", exist_ok=True)


from sklearn.model_selection import train_test_split # type: ignore

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.tree import DecisionTreeClassifier # type: ignore
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Predictions:", y_pred[:5])
print("Actual:", y_test[:5])

from sklearn.metrics import accuracy_score # type: ignore
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)    

from sklearn.metrics import confusion_matrix # type: ignore
matrix = confusion_matrix(y_test, y_pred)

import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
plt.figure(figsize=(8, 6))
sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title("Confusion Matrix")
plt.savefig('/Users/kashlogisticsltd/Desktop/Iris-Flower/iris_classifier/outputs/confusion_matrix.png')

import joblib # type: ignore
model_path = "/Users/kashlogisticsltd/Desktop/Iris-Flower/iris_classifier/outputs.joblib"
joblib.dump(model, model_path)





