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




def count_words(text):
    words = text.split(", ")
    return len(words)
df['InterestCount'] = df['Interests'].apply(count_words)

df['Interests'] = df['Interests'].str.split(', ').str[0]
df.drop_duplicates(inplace=True)

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

selected_countries = ['United States', 'India', 'China', 'Brazil', 'Russia', 'Germany', 'Japan', 'United Kingdom', 'France', 'Mexico']
selected_interests = ["'Cooking'", "'Pets'" ,"'Movies'" ,"'Gaming'", "'Fitness'" ,"'Outdoor activities'", "'Travel'", "'Business and entrepreneurship'" , "'Social causes and activism'"]

country_to_index = {country: index for index, country in enumerate(selected_countries)}
interest_to_index = {interest: index for index, interest in enumerate(selected_interests)}

df['Country'] = df['Country'].apply(lambda x: country_to_index[x] if x in selected_countries else None)
df = df.dropna(subset=['Country']).astype({'Country': 'int'})
df = df[df['Interests'].isin(selected_interests)]
df['Interests'] = df['Interests'].apply(lambda x: interest_to_index[x] if x in selected_interests else None)
df = df.dropna(subset=['Interests']).astype({'Interests': 'int'})

final_df = pd.DataFrame(columns=df.columns)
grouped = df.groupby(['Country', 'Interests'])
for (country, interest), data in grouped:
    if len(data) < 70:
        final_df = pd.concat([final_df, data], ignore_index=True)
    else:
        sampled_data = data.sample(n=70, random_state=42)
        final_df = pd.concat([final_df, sampled_data], ignore_index=True)

final_df = final_df.groupby(['Country', 'Interests']).head(70).reset_index(drop=True)

final_df.to_csv('data.csv', index=False)