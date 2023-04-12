# we just delete rows with missing values in this one

import pandas as pd

df = pd.read_csv('framingham_1.csv')

# see which features have missing values
print(df.info())

df1 = df.dropna(axis=0)

print(df1.info())

# new dataframe
result = df1
# create new csv file with new dataframe
result.to_csv(r'fr_nan_free_rows.csv', index = False, header=True)
