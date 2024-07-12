#   <<<-------matplotlib Library--------->>>

import matplotlib.pyplot as plt


#       <<.........EXP-1: Line Plot...........>>

# Example data
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

# Create a line plot
plt.plot(x, y, marker='*')
plt.title('Line Plot Example')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.grid(True)

# Display the plot
plt.show()


#       <<.........EXP-2: Bar Chart...........>>

# Example data
categories = ['A', 'B', 'C', 'D']
values = [4, 7, 1, 8]

# Create a bar chart
plt.bar(categories, values, color='green')
plt.title('Bar Chart Example')
plt.xlabel('Category')
plt.ylabel('Values')

# Display the plot
plt.show()


#       <<.........EXP-3: Scatter Plot...........>>

# Example data
x = [1, 2, 3, 4, 5]
y = [5, 7, 2, 4, 6]

# Create a scatter plot
plt.scatter(x, y, color='red', marker='x')
plt.title('Scatter Plot Example')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

# Display the plot
plt.show()


#       <<<<<<...............................END...........................................>>>>>>