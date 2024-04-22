import pandas as pd
import numpy as np
input_file = "c:/Users/maria/Desktop/analysis-of-social-media-user-data/SocialMediaUsersDataset.csv"


# print(df.values)

output_file = "interests.csv"
input_data = pd.read_csv(input_file)


unique_fields = []
for index, row in input_data.iterrows():
    fields = row['Interests'].split(', ')
    for field in fields:
        if not (field in unique_fields):
            unique_fields.append(field)

output_data = pd.DataFrame(columns=['UserID'] + list(unique_fields))
new_rows = []
for index, row in input_data.iterrows():
    fields = row['Interests'].split(', ')
    new_row = {'UserID': row['UserID']}
    for field in unique_fields:
        new_row[field] = 1 if field in fields else 0
    new_rows.append(new_row)
output_data = pd.DataFrame(new_rows)
output_data.to_csv(output_file, index=False)