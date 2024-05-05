import pandas as pd

data = pd.read_csv('DataWithNoNameNoId.csv', header=None, names=['Gender', 'DOB', 'Interests', 'City', 'Country'])
data['Interests'] = data['Interests'].apply(lambda x: list(set([interest.strip().strip("'") for interest in x.split(',')])))
interest_counts = data.explode('Interests')['Interests'].value_counts()
print("\nZainteresowania (posortowane malejąco):")
print(interest_counts)
