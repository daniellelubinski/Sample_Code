'''Finding a farmers minimum cost of feed for pigs with 
how amount of each feed while maintaining necessary nutrients a pig needs''' 
# libraries
from pyomo.environ import *
# Problem data
feed = ['Corn','Tankage','Alfalfa']
nutrient = ['Carbohydrates','Protein','Vitamins']
feed_costs_list = [2.10,1.80,1.50]
min_nutrient_req_list = [200, 180, 150]
nutrient_per_feed_list = [[90,20,40], [30,80,60], [10,20,60]]
# Parse lists into dictionaries
feed_costs = dict(zip(feed, feed_costs_list))
min_nutrient_req = dict(zip(nutrient, min_nutrient_req_list))
nutrient_per_feed = {pl: {pr: nutrient_per_feed_list[i][j] for j, pr in enumerate(feed)} for i, pl in enumerate(nutrient)}

# Declaration
model = ConcreteModel()
# Decision Variables
model.weekly_feed = Var(feed, domain=NonNegativeReals)
# Objective
model.cost = Objective(expr=sum(feed_costs[pr] * model.weekly_feed[pr] for pr in feed), sense=minimize)
# Constraints
model.requirement = ConstraintList()
for pl in nutrient:
    model.requirement.add(sum(nutrient_per_feed[pl][pr] * model.weekly_feed[pr]
                              for pr in feed) >= min_nutrient_req[pl])
model.pprint()

# Solve
solver = SolverFactory('glpk')
solver.solve(model)
# Output
print(f"Minimum Costs = ${model.cost():,.2f}")
for j in feed:
    print(f"Amount of {j} = {model.weekly_feed[j]():.2f} kg")
