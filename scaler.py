from sklearn.preprocessing import StandardScaler
import pandas as pd

df = pd.read_csv("data.csv")

selected_columns = ['InterestCount', 'Age']
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[selected_columns])
df[selected_columns] = scaled_data
df.to_csv('data_scaled.csv', index=False)