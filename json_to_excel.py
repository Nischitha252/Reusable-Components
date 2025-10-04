import pandas as pd
import json

# Load JSON data
with open('abc.json') as file:
    data = json.load(file)

# Create DataFrame
customers_df = pd.DataFrame(data['result'])

# Write DataFrame to Excel
with pd.ExcelWriter('xyz.xlsx') as writer:
    customers_df.to_excel(writer, sheet_name='xyz', index=False)

# Create and write new DataFrame to Excel
with pd.ExcelWriter('xyz.xlsx') as writer:
    new_df = customers_df.T.reset_index()
    new_df.to_excel(writer, sheet_name='Sheet1', index=False)