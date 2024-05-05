import csv

with open('SocialMediaUsersDataset.csv', 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    rows = list(reader)

filtered_rows = [row for row in rows if 'Fashion' not in row[4]]

with open('data_filtered.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(filtered_rows)
