import pandas as pd
import numpy as np
import ast
from datetime import datetime

df = pd.read_csv("c:/Users/maria/Desktop/analysis-of-social-media-user-data/SocialMediaUsersDataset.csv")


# print(df.values)

df.drop(['Name', 'UserID'], axis=1, inplace=True)
df.to_csv('DataWithNoNameNoId.csv', index=False)

# df = df.assign(**{'Interests': df['Interests'].str.split(',')}).explode('Interests')
# df.drop_duplicates(inplace=True)
# df.to_csv('DataCorrect.csv', index=False)

# df['Interests'] = df['Interests'].str.split(',').str[0]
# df.drop_duplicates(inplace=True)
# df.to_csv('DataCorrect.csv', index=False)

df['Interests'] = df['Interests'].apply(lambda x: len(ast.literal_eval(x)))


df['DOB'] = pd.to_datetime(df['DOB'])
df['Age'] = ((datetime.now() - df['DOB']).dt.days / 365.25).astype(int)

df.drop(['DOB', 'City'], axis=1, inplace=True)

gender_mapping = {'Male': 0, 'Female': 1}
df['Gender'] = df['Gender'].map(gender_mapping)


# from collections import defaultdict
# mapa_wystapien = defaultdict(int)

# for el in df['Country']:
#     mapa_wystapien[el] += 1

# lista_krotek = list(mapa_wystapien.items())
# lista_krotek_posortowana = sorted(lista_krotek, key=lambda x: x[1], reverse=True)
# pierwsze_10 = lista_krotek_posortowana[:10]
# print(pierwsze_10)
# [('United States', 12311), 
#  ('India', 9399), 
#  ('China', 7381), 
#  ('Brazil', 4643),
#  ('Russia', 4254), 
#  ('Germany', 4174), 
#  ('Japan', 3148), 
#  ('United Kingdom', 3086), 
#  ('France', 2458), 
#  ('Mexico', 2445)]

df = df.drop(df[df['Interests'] > 10].index)

selected_countries = ['United States', 'India', 'China', 'Brazil', 'Russia', 'Germany', 'Japan', 'United Kingdom', 'France', 'Mexico']

dfs_by_country = {}
for country in selected_countries:
    if country in df['Country'].unique():
        country_data = df[df['Country'] == country].head(2000)
        dfs_by_country[country] = country_data

final_df = pd.concat(dfs_by_country.values(), ignore_index=True)


for country, df_country in dfs_by_country.items():
    print(f"\nDataFrame for {country}:")
    print(df_country)

df.to_csv('data.csv', index=False)