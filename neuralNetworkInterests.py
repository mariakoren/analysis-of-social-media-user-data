import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import FeatureHasher

data = pd.read_csv("DataCorrect.csv")

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.5),  
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5),  
    tf.keras.layers.Dense(len(interests), activation='softmax') 
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Zmiana funkcji straty
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)