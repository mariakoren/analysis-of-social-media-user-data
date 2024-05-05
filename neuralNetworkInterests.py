import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("DataCorrect.csv")
data.drop(columns=['City', 'Country'])

interests = data['Interests'].unique()
interests_dict = {interest: idx for idx, interest in enumerate(interests)}

data['Interests'] = data['Interests'].map(interests_dict)

gender_mapping = {'Male': 0, 'Female': 1}
data['Gender'] = data['Gender'].map(gender_mapping)

current_year = 2024
data['DOB'] = pd.to_datetime(data['DOB'])
data['Age'] = current_year - data['DOB'].dt.year

data.drop(columns=['DOB'], inplace=True)

city_country_iterable = data[['City', 'Country']].astype(str).apply(lambda x: [x['City'], x['Country']], axis=1)

hasher = FeatureHasher(n_features=10, input_type='string')

hashed_features = hasher.fit_transform(city_country_iterable)

hashed_df = pd.DataFrame(hashed_features.toarray())

data = pd.concat([data.drop(columns=['City', 'Country']), hashed_df], axis=1)

X = data.drop(columns=['Interests'])
y = data['Interests']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y = tf.keras.utils.to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.5),  
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5),  
    tf.keras.layers.Dense(len(interests), activation='softmax') 
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)


y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

plt.figure(figsize=(100, 80))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=interests, yticklabels=interests)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
