import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

input_file = "c:/Users/maria/Desktop/analysis-of-social-media-user-data/SocialMediaUsersDataset.csv"
output_file = "c:/Users/maria/Desktop/analysis-of-social-media-user-data/interests.csv"

input_data = pd.read_csv(input_file)

unique_fields = set()
for interests in input_data['Interests'].str.split(', '):
    unique_fields.update(interests)

data_dict = {'UserID': input_data['UserID']}
for field in unique_fields:
    data_dict[field] = input_data['Interests'].str.contains(field).astype(int)

output_data = pd.DataFrame(data_dict)

output_data.to_csv(output_file, index=False)

df = pd.read_csv(output_file)

freq_items = apriori(df.drop(columns=['UserID']), min_support=0.001, use_colnames=True, verbose=1, max_len=None)

rules = association_rules(freq_items, metric="confidence", min_threshold=0.1, support_only=False)

selected_rules = rules.sort_values(by='confidence', ascending=False)
print(selected_rules.head(10))

