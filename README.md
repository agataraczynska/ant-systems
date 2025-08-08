# ant-systems
Implementation of Ant Colony Optimisation algorithms inspired by real ant foraging behaviour. Includes simulation of colony path selection on asymmetric bridges and an Ant System approach to solve the Travelling Salesman Problem, with pheromone-based stochastic decision rules.




# Ant Colony Optimisation Systems

This repository contains Python implementations of Ant Colony Optimisation (ACO) systems for two classic problems:

1. **Shortest Path Problem** – ants choosing paths across a double asymmetric bridge.
2. **Travelling Salesman Problem (TSP)** – ants finding the shortest route visiting all cities.

---

## Task 1 – Shortest Path Across a Double Asymmetric Bridge

A map of the environment with a separated nest and food source was created. The nest and food were connected by a double asymmetric bridge with the following coordinates:

x = [0, 1, 2, 2, 3, 4, 5, 5, 6, 7]
y = [2, 2, 0, 3, 2, 2, 4, 1, 2, 2]

In the simulation:
- **Number of ants**: 1000
- **Path choice**: The probability of choosing the upper branch (*pR*) or lower branch (*pL*) was determined by a pheromone-based equation.
- **Decision rule**: A random number `r ∈ [0,1]` determined which branch the ant would choose.

**Results**:
- 80.1% of ants chose the shorter path through the **first bridge**.
- 93.2% of ants chose the shorter path through the **second bridge**.

**Conclusion**:
The simulation showed that ants tend to reinforce and select the shortest path due to pheromone accumulation, effectively modelling real-world ant colony behaviour.

---

## Task 2 – Travelling Salesman Problem (TSP)

The set of cities used (N = 10) was **Set 1** from Laboratory 1 (*Genetic Algorithm for TSP*), allowing performance comparison.

x = [0, 3, 6, 7, 15, 12, 14, 9, 7, 0]
y = [1, 4, 5, 3, 0, 4, 10, 6, 9, 10]

**Results**:
- Minimal total distance travelled: **55.044**
- Optimal city visit sequence:
`[0, 1, 2, 3, 7, 5, 4, 6, 8, 9, 0]` (and its permutations with equal cost)
- Pheromone concentration was highest along optimal routes, with near-zero values elsewhere.

**Conclusion**:
The ACO algorithm efficiently found the optimal route for the given set of cities. The result matched the optimal solution previously found using a Genetic Algorithm.

---

## How to Run the Code

### Requirements
- Python 3.x
- NumPy
- Matplotlib

You can install dependencies with:
pip install numpy matplotlib

### Running
Clone this repository:
git clone https://github.com/agataraczynska/ant-systems.git
cd ant-systems

Run the simulation scripts:
python ant_systems.py

Plots and outputs will be generated showing the paths chosen by ants and TSP solutions.

---

## References
- Dorigo, M., & Gambardella, L. M. (1997). *Ant colonies for the travelling salesman problem*. BioSystems, 43(2), 73–81.
- Bonabeau, E., Dorigo, M., & Theraulaz, G. (1999). *Swarm Intelligence: From Natural to Artificial Systems*. Oxford University Press.

