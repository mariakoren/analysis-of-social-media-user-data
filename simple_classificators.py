import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
# df = pd.read_csv("data.csv")
df = pd.read_csv("data_scaled.csv")


# a
train_set, test_set = train_test_split(df, train_size=0.7, random_state=14)
X_train = train_set.drop(columns=["Interests"])
y_train = train_set["Interests"]
X_test = test_set.drop(columns=["Interests"])
y_test = test_set["Interests"]


# 3-NN
knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(X_train, y_train)
y_pred_knn3 = knn3.predict(X_test)

# 5-NN
knn5 = KNeighborsClassifier(n_neighbors=5)
knn5.fit(X_train, y_train)
y_pred_knn5 = knn5.predict(X_test)

# 11-NN
knn11 = KNeighborsClassifier(n_neighbors=11)
knn11.fit(X_train, y_train)
y_pred_knn11 = knn11.predict(X_test)


# 110-NN
knn100 = KNeighborsClassifier(n_neighbors=100)
knn100.fit(X_train, y_train)
y_pred_knn100 = knn11.predict(X_test)

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Desicion Tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred_dt = clf.predict(X_test)


tree_text = export_text(clf, feature_names=X_train.columns.tolist())
plt.figure(figsize=(10, 7))
plot_tree(clf, feature_names=X_train.columns.tolist(), filled=True)
plt.savefig('tree.png')

# Dokładność klasyfikatorów
accuracy_knn3 = accuracy_score(y_test, y_pred_knn3)
accuracy_knn5 = accuracy_score(y_test, y_pred_knn5)
accuracy_knn11 = accuracy_score(y_test, y_pred_knn11)
accuracy_knn100 = accuracy_score(y_test, y_pred_knn100)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_dt = clf.score(X_test, y_test)

confusion_matrix_knn3 = confusion_matrix(y_test, y_pred_knn3)
confusion_matrix_knn5 = confusion_matrix(y_test, y_pred_knn5)
confusion_matrix_knn11 = confusion_matrix(y_test, y_pred_knn11)
confusion_matrix_knn100 = confusion_matrix(y_test, y_pred_knn100)
confusion_matrix_nb = confusion_matrix(y_test, y_pred_nb)
confusion_matrix_rf = confusion_matrix(y_test, y_pred_rf)
confusion_matrix_dt = confusion_matrix(y_test, y_pred_dt)

print("Dokładność klasyfikatora 3-NN:", accuracy_knn3)
print("Dokładność klasyfikatora 5-NN:", accuracy_knn5)
print("Dokładność klasyfikatora 11-NN:", accuracy_knn11)
print("Dokładność klasyfikatora 100-NN:", accuracy_knn100)
print("Dokładność klasyfikatora Naive Bayes:", accuracy_nb)
print("Dokładność klasyfikatora Random Forest:", accuracy_rf)
print("Dokładność klasyfikatora Desicion Tree:", accuracy_dt)




# print("\nMacierz błędu dla klasyfikatora 3-NN:\n", confusion_matrix_knn3)
# print("\nMacierz błędu dla klasyfikatora 5-NN:\n", confusion_matrix_knn5)
# print("\nMacierz błędu dla klasyfikatora 11-NN:\n", confusion_matrix_knn11)
# print("\nMacierz błędu dla klasyfikatora 100-NN:\n", confusion_matrix_knn100)
# print("\nMacierz błędu dla klasyfikatora Naive Bayes:\n", confusion_matrix_nb)

def plot_confusion_matrix(cm, title='Macierz błędu', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.xlabel('Przewidziane etykiety')
    plt.ylabel('Prawdziwe etykiety')
    plt.tight_layout()

# Tworzymy obrazy dla każdej macierzy błędu.
plt.figure(figsize=(15, 12))

plt.subplot(231)
plot_confusion_matrix(confusion_matrix_knn3)
plt.title("Macierz błędu dla klasyfikatora 3-NN")

plt.subplot(232)
plot_confusion_matrix(confusion_matrix_knn5)
plt.title("Macierz błędu dla klasyfikatora 5-NN")

plt.subplot(233)
plot_confusion_matrix(confusion_matrix_knn11)
plt.title("Macierz błędu dla klasyfikatora 11-NN")

plt.subplot(234)
plot_confusion_matrix(confusion_matrix_nb)
plt.title("Macierz błędu dla klasyfikatora Naive Bayes")

plt.subplot(235)
plot_confusion_matrix(confusion_matrix_rf)
plt.title("Macierz błędu dla klasyfikatora Random Forest")

plt.subplot(236)
plot_confusion_matrix(confusion_matrix_dt)
plt.title("Macierz błędu dla klasyfikatora Desicion Tree")


# plt.savefig('macierze_bledow.png')
plt.savefig('macierze_bledow_scaled.png')


