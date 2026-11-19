import pandas as pd
df = {
    'mpg': [18, 15, 18, 16, 17],
    'cylinders': [4, 8, 8, 4, 6],
    'displacement': [300, 325, 308, 304, 318],
    'HP': [150, 160, 130, 150, 156],
    'weight': [3506.3457, 3089, 3560, 3524, 3400],
    'acceleration': [12.0, 12.5, 11.0, 10.5, 11.5],
    'model year': [70, 71, 70, 80, 75],
    'origin': [1, 1, 1, 1, 1],
    'carname': ['honda', 'nexon', 'ford', 'indico', 'swift']
}
df1 = pd.DataFrame(df)
print(df1[df1['cylinders'] == 8])
print(df1.groupby('model year')['carname'].count())
print(df1['HP'].mean())
print(df1['acceleration'].std())
print(df1['model year'].value_counts().sort_index())
print(df1[df1['model year'] == 72])
print(df1.sort_values(by='cylinders'))
print(df1.max(numeric_only=True))
print(df1.min(numeric_only=True))