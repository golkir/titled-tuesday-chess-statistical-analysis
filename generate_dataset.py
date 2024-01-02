import pandas
from mongo import *

documents = findAll(players)

# Extract relevant data and flatten the structure
data = []
for doc in documents:
    username = doc["username"]
    for game in doc["games"]:
        # Use get to handle cases where the username is not present
        data.append({"username": username, **game})

# Create a Pandas DataFrame
df = pandas.DataFrame(data)

# Specify the file path where you want to save the CSV file
file_path = 'data/titled-tuesday.csv'

# Save the DataFrame to a CSV file
df.to_csv(file_path, index=False)

print(f'DataFrame has been saved to {file_path}')
