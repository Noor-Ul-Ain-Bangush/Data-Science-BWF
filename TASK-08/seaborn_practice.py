#   <<<-------seaborn Library--------->>>

import seaborn as sns
import matplotlib.pyplot as plt

'''
#       <<.........EXP-1: Scatter Plot...........>>

# Load example dataset
data = sns.load_dataset("iris")

# Create a scatter plot
sns.scatterplot(x='sepal_length', y='sepal_width', data=data)
plt.title('Scatter Plot of Sepal Length vs Sepal Width')
plt.show()

#       <<.........EXP-2: Histogram...........>>

# Load example dataset
data = sns.load_dataset("tips")

# Create a histogram
sns.histplot(data['total_bill'], bins=20, kde=True)
plt.title('Histogram of Total Bill')
plt.xlabel('Total Bill')
plt.ylabel('Frequency')
plt.show()


#       <<.........EXP-3: Box Plot...........>>

# Load example dataset
data = sns.load_dataset("tips")

# Create a box plot
sns.boxplot(x='day', y='total_bill', data=data)
plt.title('Box Plot of Total Bill by Day')
plt.xlabel('Day')
plt.ylabel('Total Bill')
plt.show()

'''

#       <<.....................EXP-4...................>>

# Apply the default theme
sns.set_theme()

# Load an example dataset
tips = sns.load_dataset("tips")

# Create a visualization
sns.relplot(
    data=tips,
    x="total_bill", y="tip", col="time",
    hue="smoker", style="smoker", size="size",
)

plt.show()

#                 <<.................EXP-4.1....................>>

#A high-level API for statistical graphics
dots = sns.load_dataset("dots")
sns.relplot(
    data=dots, kind="line",
    x="time", y="firing_rate", col="align",
    hue="choice", size="coherence", style="choice",
    facet_kws=dict(sharex=False),
)

plt.show()

#    ****.......................Statistical estimation..............................*****
#                 <<.................EXP-4.2....................>>

fmri = sns.load_dataset("fmri")
sns.relplot(
    data=fmri, kind="line",
    x="timepoint", y="signal", col="region",
    hue="event", style="event",
)
plt.show()


#                 <<.................EXP-4.3....................>>

sns.lmplot(data=tips, x="total_bill", y="tip", col="time", hue="smoker")
plt.show()


#      ****.........................Distributional representations...........................****
#                 <<.................EXP-4.4....................>>

sns.displot(data=tips, x="total_bill", col="time", kde=True)
plt.show()


#                 <<.................EXP-4.5....................>>

sns.displot(data=tips, kind="ecdf", x="total_bill", col="time", hue="smoker", rug=True)
plt.show()


#           ****.....................Plots for categorical data.........................****
#                 <<.................EXP-4.6....................>>

sns.catplot(data=tips, kind="swarm", x="day", y="total_bill", hue="smoker")
plt.show()


#                 <<.................EXP-4.7....................>>

sns.catplot(data=tips, kind="violin", x="day", y="total_bill", hue="smoker", split=True)
plt.show()


#                 <<.................EXP-4.8....................>>

sns.catplot(data=tips, kind="bar", x="day", y="total_bill", hue="smoker")
plt.show()

#       ****..................Multivariate views on complex datasets..................****
#                 <<.................EXP-4.9....................>>

penguins = sns.load_dataset("penguins")
sns.jointplot(data=penguins, x="flipper_length_mm", y="bill_length_mm", hue="species")
plt.show()

#                 <<.................EXP-4.10....................>>

sns.pairplot(data=penguins, hue="species")
plt.show()


#        ****..................Lower-level tools for building figures..................****
#                 <<.................EXP-4.11....................>>

g = sns.PairGrid(penguins, hue="species", corner=True)
g.map_lower(sns.kdeplot, hue=None, levels=5, color=".2")
g.map_lower(sns.scatterplot, marker="+")
g.map_diag(sns.histplot, element="step", linewidth=0, kde=True)
g.add_legend(frameon=True)
g.legend.set_bbox_to_anchor((.61, .6))
plt.show()

#       ****..............Opinionated defaults and flexible customization..................****
#                 <<.................EXP-4.12....................>>

sns.relplot(
    data=penguins,
    x="bill_length_mm", y="bill_depth_mm", hue="body_mass_g"
)
plt.show()

#                 <<.................EXP-4.13....................>>

sns.set_theme(style="ticks", font_scale=1.25)
g = sns.relplot(
    data=penguins,
    x="bill_length_mm", y="bill_depth_mm", hue="body_mass_g",
    palette="crest", marker="x", s=100,
)
g.set_axis_labels("Bill length (mm)", "Bill depth (mm)", labelpad=10)
g.legend.set_title("Body mass (g)")
g.figure.set_size_inches(6.5, 4.5)
g.ax.margins(.15)
g.despine(trim=True)

plt.show()


#       <<<<<<...............................END...........................................>>>>>>