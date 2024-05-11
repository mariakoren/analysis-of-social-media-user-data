import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv("data.csv")
df.drop(columns=['Interests', 'Country'], inplace=True)


average_age = df['Age'].mean()
df['Age'] = df['Age'].apply(lambda x: 1 if x >= average_age else 0)

average_count = df['InterestCount'].mean()
df['InterestCount'] = df['InterestCount'].apply(lambda x: 1 if x >= average_count else 0)

freq_items = apriori(df, min_support=0.2, use_colnames=True, verbose=1, max_len=None)

rules = association_rules(freq_items, metric="confidence", min_threshold=0.1, support_only=False)

selected_rules = rules.sort_values(by='confidence', ascending=False)
print(selected_rules)
# print(average_age) # 44.33248650365195
# print(average_count) # 3.005080978088282