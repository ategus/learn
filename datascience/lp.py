import pulp
import matplotlib.pyplot as plt
import numpy as np

# Create the LP problem
problem = pulp.LpProblem("Profit_Maximization", pulp.LpMaximize)

# Define the variables
chairs = pulp.LpVariable("Chairs", lowBound=0, cat='Integer')
tables = pulp.LpVariable("Tables", lowBound=0, cat='Integer')

# Define the objective function
problem += 40 * chairs + 50 * tables, "Profit"

# Define the constraints
problem += 2 * chairs + 1 * tables <= 100, "Labor_Constraint"
problem += 1 * chairs + 3 * tables <= 90, "Wood_Constraint"

# Solve the problem
problem.solve()

# Print the results
print("Status:", pulp.LpStatus[problem.status])
print("Optimal number of chairs to produce:", chairs.varValue)
print("Optimal number of tables to produce:", tables.varValue)
print("Maximum profit:", pulp.value(problem.objective))

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))

# Plot constraints
x = np.linspace(0, 100, 1000)
labor_constraint = (100 - 2*x) / 1  # y = (100 - 2x) / 1 for the labor constraint
wood_constraint = (90 - x) / 3  # y = (90 - x) / 3 for the wood constraint

ax.plot(x, labor_constraint, label='Labor Constraint')
ax.plot(x, wood_constraint, label='Wood Constraint')

# Shade the feasible region
feasible_x = np.minimum(50, 90)  # x coordinate where constraints intersect
feasible_y = (100 - 2*feasible_x) / 1  # y coordinate where constraints intersect
ax.fill_between(x, 0, np.minimum(labor_constraint, wood_constraint), 
                where=(x <= feasible_x), alpha=0.2, color='green', label='Feasible Region')

# Plot optimal solution
ax.plot(chairs.varValue, tables.varValue, 'ro', markersize=10, label='Optimal Solution')

ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_xlabel('Number of Chairs')
ax.set_ylabel('Number of Tables')
ax.set_title('Production Optimization')
ax.legend()
ax.grid(True)

plt.show()
