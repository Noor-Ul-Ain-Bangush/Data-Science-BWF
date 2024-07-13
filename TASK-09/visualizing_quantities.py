#   <<<--------Build a line plot about bird wingspan values-------->>>

import pandas as pd
import matplotlib.pyplot as plt

birds = pd.read_csv(r'E:\Data-Science-BWF-Noor Ul Ain (Task9)\pythonProject\minnesota_birds.csv')
print(birds.head())

#plotting some of the numeric data using a basic line plot
wingspan = birds['Wingspan Max (cm)']
wingspan.plot()

plt.title('Max Wingspan in CM')
plt.ylabel('Wingspan (CM)')
plt.xlabel('Birds')
plt.xticks(rotation=45)
x = birds['Name']
y = birds['Wingspan Max (cm)']
plt.plot(x,y)
plt.show()

#use a scatter chart
plt.title('Max Wingspan in CM')
plt.ylabel('Wingspan (CM)')
plt.tick_params(axis='both', which='both', labelbottom=False, bottom=False)

for i in range(len(birds)):
    x = birds['Name'][i]
    y = birds['Wingspan Max (cm)'][i]
    plt.plot(x, y, 'bo')
    if birds['Wingspan Max (cm)'][i] > 500:
        plt.text(x, y * (1 - 0.05), birds['Name'][i], fontsize=12)

plt.show()

#Finding your data
plt.title('Max Wingspan in CM')
plt.ylabel('Wingspan (CM)')
plt.xlabel('Birds')
plt.tick_params(axis='both',which='both',labelbottom=False,bottom=False)
for i in range(len(birds)):
    x = birds['Name'][i]
    y = birds['Wingspan Max (cm)'][i]
    if birds['Name'][i] not in ['Bald eagle', 'Prairie falcon']:
        plt.plot(x, y, 'bo')
plt.show()


#Explore bar charts
birds.plot(x='Category',
        kind='bar',
        stacked=True,
        title='Birds of Minnesota')

category_count = birds.value_counts(birds['Category'].values, sort=True)
plt.rcParams['figure.figsize'] = [6, 12]
category_count.plot.barh()


#Comparing data
maxlength = birds['Length Max (cm)']
plt.barh(y=birds['Category'], width=maxlength)
plt.rcParams['figure.figsize'] = [6, 12]
plt.show()

#superimpose Minimum and Maximum Length
minLength = birds['Length Min (cm)']
maxLength = birds['Length Max (cm)']
category = birds['Category']

plt.barh(category, maxLength)
plt.barh(category, minLength)

plt.show()