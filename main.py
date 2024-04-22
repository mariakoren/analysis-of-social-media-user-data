import pandas as pd
import numpy as np
df = pd.read_csv("c:/Users/maria/Desktop/analysis-of-social-media-user-data/SocialMediaUsersDataset.csv")


# print(df.values)

df.drop(['Name', 'UserID'], axis=1, inplace=True)
df.to_csv('DataWithNoNameNoId.csv', index=False)

df = df.assign(**{'Interests': df['Interests'].str.split(',')}).explode('Interests')
df.drop_duplicates(inplace=True)
df.to_csv('DataCorrect.csv', index=False)

