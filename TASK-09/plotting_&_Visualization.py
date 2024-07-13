#           <<<<---------CHAPTER:09_Plotting & Visualization------------>>>>

import numpy as np
import matplotlib.pyplot as plt

#example-01
data = np.arange(10)
print(data)
plt.plot(data)
plt.show()


#example-02
fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
plt.show()

ax3.plot(np.random.standard_normal(50).cumsum(), color="black", linestyle="dashed")
ax1.hist(np.random.standard_normal(100), bins=20, color="black", alpha=0.3)
ax2.scatter(np.arange(30), np.arange(30) + 3 * np.random.standard_normal(30))
plt.show()


#example-03
fig , axes = plt.subplots(2,3)
print(axes)


#example-04: Adjusting the spacing around subplots
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
for i in range(2):
    for j in range(2):
        axes[i, j].hist(np.random.standard_normal(500), bins=50, color="black", alpha=0.5)
fig.subplots_adjust(wspace=0, hspace=0)
plt.show()


#example-05: Colors, Markers & Linestyles
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(np.random.standard_normal(30).cumsum(), color="red", linestyle="dashed", marker="o");
plt.show()


#example-06: Ticks, Labels & Legends
fig, ax = plt.subplots()
ax.plot(np.random.standard_normal(1000).cumsum())
plt.show()


#example-07
ticks = ax.set_xticks([0, 250, 500, 750, 1000])
labels = ax.set_xticklabels(["one", "two", "three", "four", "five"], rotation=30, fontsize=8)
plt.show()


#example-08
ax.text(x, y, "Hello world!",
        family="monospace", fontsize=10)
plt.show()


#example-09
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
rect = plt.Rectangle((0.2, 0.75), 0.4, 0.15, color="black", alpha=0.3)
circ = plt.Circle((0.7, 0.2), 0.15, color="blue", alpha=0.3)
pgon = plt.Polygon([[0.15, 0.15], [0.35, 0.4], [0.2, 0.6]], color="green", alpha=0.5)
ax.add_patch(rect)
ax.add_patch(circ)
ax.add_patch(pgon)
plt.show()

# Use raw string to specify the file path
fig.savefig(r"E:\Data-Science-BWF-Noor Ul Ain (Task9)\pythonProject.svg")
fig.savefig(r"E:\Data-Science-BWF-Noor Ul Ain (Task9)\pythonProject.pdf")
fig.savefig(r"E:\Data-Science-BWF-Noor Ul Ain (Task9)\pythonProject.png", dpi=400)



#           <<<<--------Using pandas & seaborn Library------------>>>>

#example-01: Line Plots
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

s = pd.Series(np.random.standard_normal(10).cumsum(), index=np.arange(0, 100, 10))
s.plot()
plt.show()


#example-02: Bar Plots
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1)
data = pd.Series(np.random.uniform(size=16), index=list("abcdefghijklmnop"))
data.plot.bar(ax=axes[0], color="black", alpha=0.7)
data.plot.barh(ax=axes[1], color="black", alpha=0.7)
plt.show()


#example-03
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Create a DataFrame with random uniform data
df = pd.DataFrame(np.random.uniform(size=(6, 4)),
                  index=["one", "two", "three", "four", "five", "six"],
                  columns=pd.Index(["A", "B", "C", "D"], name="Genus"))

# Print the DataFrame
print(df)
df.plot.bar()
df.plot.barh(stacked=True, alpha=0.5)
plt.show()


#example-04: Histograms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

comp1 = np.random.standard_normal(200)
comp2 = 10 + 2 * np.random.standard_normal(200)
values = pd.Series(np.concatenate([comp1, comp2]))
sns.histplot(values, bins=100, color="black")
plt.show()


#example-05: Scatter Plots
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming trans_data is already defined and loaded correctly
# Example DataFrame creation for demonstration purposes
trans_data = pd.DataFrame({
    "m1": np.random.rand(100),
    "unemp": np.random.rand(100)
})

ax = sns.regplot(x="m1", y="unemp", data=trans_data)
ax.set_title("Changes in log(m1) versus log(unemp)")
sns.pairplot(trans_data, diag_kind="kde", plot_kws={"alpha": 0.2})
plt.show()


#<<<................................................END.................................................>>>