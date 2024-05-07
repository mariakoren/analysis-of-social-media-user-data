import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("data.csv")
train_set, test_set = train_test_split(df, train_size=0.7, random_state=14)
X_train = train_set.drop(columns=["Interests"])
y_train = train_set["Interests"]
X_test = test_set.drop(columns=["Interests"])
y_test = test_set["Interests"]

# 1 - positive; 0 - negative
        


# mlp = MLPClassifier(hidden_layer_sizes=(6, 3), activation='logistic', max_iter=500)
# mlp.fit(train_data, train_labels)
# przy logistic 
# [[148   0]
#  [ 83   0]]

mlp = MLPClassifier(hidden_layer_sizes=(9, 10, 4), activation='identity', max_iter=500)
mlp.fit(X_train,  y_train)
        
predictions_test = mlp.predict(X_test)
print(accuracy_score(y_test, predictions_test))
print(confusion_matrix(y_test, predictions_test))