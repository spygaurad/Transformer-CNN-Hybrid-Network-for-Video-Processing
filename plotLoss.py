import matplotlib.pyplot as plt

# Sample data
data = []

with open(f'trainLogs/Training_Full_Model.txt', 'r') as f:
    for line in f:
        num = int(line.strip())
        data.append(num)

# Create a figure and axis
fig, ax = plt.subplots()

# Plot the data
ax.plot(data)

# Show the plot
plt.show()