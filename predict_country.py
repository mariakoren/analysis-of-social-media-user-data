import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data.csv")

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


# 11-NN
knn100 = KNeighborsClassifier(n_neighbors=100)
knn100.fit(X_train, y_train)
y_pred_knn100 = knn11.predict(X_test)

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)

# Dokładność klasyfikatorów
accuracy_knn3 = accuracy_score(y_test, y_pred_knn3)
accuracy_knn5 = accuracy_score(y_test, y_pred_knn5)
accuracy_knn11 = accuracy_score(y_test, y_pred_knn11)
accuracy_knn100 = accuracy_score(y_test, y_pred_knn100)
accuracy_nb = accuracy_score(y_test, y_pred_nb)

confusion_matrix_knn3 = confusion_matrix(y_test, y_pred_knn3)
confusion_matrix_knn5 = confusion_matrix(y_test, y_pred_knn5)
confusion_matrix_knn11 = confusion_matrix(y_test, y_pred_knn11)
confusion_matrix_knn100 = confusion_matrix(y_test, y_pred_knn100)
confusion_matrix_nb = confusion_matrix(y_test, y_pred_nb)

print("Dokładność klasyfikatora 3-NN:", accuracy_knn3)
print("Dokładność klasyfikatora 5-NN:", accuracy_knn5)
print("Dokładność klasyfikatora 11-NN:", accuracy_knn11)
print("Dokładność klasyfikatora 100-NN:", accuracy_knn100)
print("Dokładność klasyfikatora Naive Bayes:", accuracy_nb)

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
plt.figure(figsize=(10, 8))

plt.subplot(221)
plot_confusion_matrix(confusion_matrix_knn3)
plt.title("Macierz błędu dla klasyfikatora 3-NN")

plt.subplot(222)
plot_confusion_matrix(confusion_matrix_knn5)
plt.title("Macierz błędu dla klasyfikatora 5-NN")

plt.subplot(223)
plot_confusion_matrix(confusion_matrix_knn11)
plt.title("Macierz błędu dla klasyfikatora 11-NN")

plt.subplot(224)
plot_confusion_matrix(confusion_matrix_nb)
plt.title("Macierz błędu dla klasyfikatora Naive Bayes")

# Zapisujemy wykres do pliku obrazkowego.
plt.savefig('macierze_bledow.png')

