#   <<<-------Pandas Library--------->>>

import pandas as pd


#       <<.........EXP-1: Creating a DataFrame...........>>

# Creating a DataFrame from a dictionary
data = {
    'Name': ['Noor', 'Iqra', 'Ali'],
    'Age': [21, 18, 35],
    'City': ['New York', 'America', 'Chicago']
}

df = pd.DataFrame(data)
print(df)



#       <<.........EXP-2: Reading a CSV file...........>>

df = pd.read_csv(r'E:\Data-Science-BWF-Noor Ul Ain (Task8)\pythonlibraries\Iris.csv')
print(df.head())



#       <<.........EXP-3: Basic Data Manipulation...........>>

# Creating a DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']
}

df = pd.DataFrame(data)

# Filtering rows where Age is greater than 30
filtered_df = df[df['Age'] > 30]
print(filtered_df)

# Adding a new column
df['Salary'] = [70000, 80000, 90000, 100000]
print(df)

# Calculating the mean age
mean_age = df['Age'].mean()
print(f'Mean Age: {mean_age}')


#       <<<<<<...............................END...........................................>>>>>>