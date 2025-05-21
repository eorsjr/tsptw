

# Traveling Salesman Problem with Time Windows (TSP-TW) using Genetic Algorithm

## Overview
This project solves a variant of the classic Traveling Salesman Problem (TSP), where each city must be visited within a specific time window. The objective is to find a minimal-cost route that starts and ends at city A, visits all cities exactly once, and minimizes a combined cost function of total travel distance and penalties for arriving outside specified time windows.

## Problem Description
- **Start and End:** The route begins and ends at city A (index 0).
- **Constraints:** Each city must be visited once. Cities B through H have predefined time windows.
- **Penalty:** If the agent arrives at a city outside its time window, a penalty of 1 unit is added per violation.
- **Total Cost:** Computed as the total distance traveled plus the total penalty.

## Metaheuristic Method

### Genetic Algorithm (GA)
This problem is solved using a Genetic Algorithm (GA), a population-based metaheuristic method that mimics natural selection. The algorithm was specifically tailored for the TSP with Time Windows as follows:

- **Encoding:** Each individual in the population is a permutation of city indices, always starting and ending with 0 (representing city A).
- **Population Initialization:** Each route is a shuffled sequence of cities (excluding A), wrapped with `[0]` at the beginning and end.
- **Crossover Operator:** Order Crossover (OX) is used to preserve partial orderings from parents while producing valid permutations.
- **Mutation Operator:** Swap Mutation randomly selects two non-depot cities in a route and swaps them.
- **Selection Strategy:** Tournament selection with `k=3` selects the best individual among three randomly sampled candidates.
- **Fitness Function:** The sum of the total travel distance and the number of time window violations (each contributing a penalty of 1).

## Source Files
- `genetic_algorithm.py`: Contains the `GeneticAlgorithmTSPTW` class with all core components of the GA.
- `TSPTW.ipynb`: Demonstrates how to define the TSP-TW instance, execute the GA, and analyze the results.

## Evaluation

- **Feasibility Check (B–E):** No feasible solution was found that satisfies all time windows for cities B through E.
- **Best Approximate Route Found:**
```
A → F → C → G → D → H → B → E → A
```
- **Performance Metrics:**
  - Total Distance: `16`
  - Time Window Violations: `6`
  - Total Cost (Distance + Penalty): `22`

Although no fully feasible route was found, the genetic algorithm effectively minimized the total cost under the given constraints, demonstrating robustness in handling constrained optimization problems like the TSP-TW.