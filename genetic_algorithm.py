import numpy as np
import random

np.random.seed(42)
random.seed(42)

class GeneticAlgorithmTSPTW:
    """
    A class to solve the Traveling Salesman Problem with Time Windows (TSPTW) using a genetic algorithm."""
    
    def __init__(self, distance_matrix, time_windows, population_size=100, mutation_rate=0.1, generations=300):
        self.distance_matrix = distance_matrix
        self.time_windows = time_windows
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.num_cities = len(distance_matrix)
        self.city_names = list(time_windows.keys())
        self.best_route = None
        self.best_cost = float('inf')
        self.cost_history = []


    def init_population(self):
        """
        Initialize the population with random routes.
        Each route starts and ends at the depot (city 0)."""
        
        population = []
        
        for _ in range(self.population_size): # Create a random route
            route = list(range(1, self.num_cities)) # Exclude depot
            np.random.shuffle(route) # Shuffle the cities
            route = [0] + route + [0] # Add depot at start and end
            population.append(route) # Add to population
        
        return population # Return the initialized population

    def crossover(self, parent1, parent2):
        """
        Perform order crossover (OX) between two parents to create a child.
        The child inherits a segment from one parent and fills the rest with the other parent's genes."""
        
        size = len(parent1)
        start, end = sorted(np.random.choice(range(1, size - 1), 2, replace=False)) # Randomly select crossover points

        child = [None] * size # Create empty child
        child[start:end] = parent1[start:end] # Fill segment from parent1
        
        fill_values = [city for city in parent2 if city not in child and city != 0] # Get remaining cities from parent2
        
        fill_positions = [i for i in range(1, size - 1) if child[i] is None] # Get positions to fill in child

        for i, city in zip(fill_positions, fill_values): # Fill remaining positions in child
            child[i] = city

        child[0] = child[-1] = 0 # Set depot at start and end
        return child # Return the child route

    def mutate(self, route):
        """
        Perform swap mutation on the route.
        Two cities are selected randomly and swapped to create a new route."""
        
        mutated = route[:] # Create a copy of the route
        idx1, idx2 = np.random.choice(range(1, len(route) - 1), 2, replace=False) # Randomly select two cities to swap
        mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1] # Swap the cities
        return mutated # Return the mutated route

    def select(self, population, k=3):
        """
        Select a route from the population using tournament selection.
        A subset of the population is randomly selected, and the best route among them is chosen."""
        
        selected = random.sample(population, k) # Randomly select k routes from the population
        selected = sorted(selected, key=lambda r: self.fitness(r)) # Sort the selected routes by fitness
        return selected[0] # Return the best route among the selected

    def calc_cost(self, route):
        """
        Calculate the total cost of the route.
        The cost is the sum of distances between consecutive cities in the route."""
        
        cost = 0
        
        for i in range(len(route) - 1): # Iterate through the route
            from_city = route[i] # Current city
            to_city = route[i + 1] # Next city
            cost += self.distance_matrix[from_city][to_city] # Add distance to cost
        
        return cost # Return the total cost of the route

    def calc_penalty(self, route):
        """
        Calculate the penalty for violating time windows.
        The penalty is the number of time window violations in the route."""
        
        current_time = 0 # Initialize current time
        penalty = 0 # Initialize penalty
        
        for i in range(1, len(route)): # Iterate through the route
            from_city = route[i - 1] # Previous city
            to_city = route[i] # Current city
            current_time += self.distance_matrix[from_city][to_city] # Update current time
            min_time, max_time = self.time_windows[self.city_names[to_city]] # Get time window for current city
            if current_time < min_time or current_time > max_time: # Check if current time is within time window
                penalty += 1 # Increment penalty if time window is violated
        
        return penalty # Return the total penalty for time window violations

    def fitness(self, route):
        """
        Calculate the fitness of the route.
        The fitness is the total cost plus a penalty for time window violations."""
        
        cost = self.calc_cost(route) # Calculate the cost of the route
        penalty = self.calc_penalty(route) # Calculate the penalty for time window violations
        return cost + penalty # Return the total cost and penalty
    
    def eval_route(self, route):
        """
        Evaluate the route and return the total distance, penalty, and fitness.
        The fitness is the total distance plus a penalty for time window violations."""
        
        distance = self.calc_cost(route) # Calculate the cost of the route
        penalty = self.calc_penalty(route) # Calculate the penalty for time window violations
        return distance, penalty, distance + penalty # Return the total cost and penalty

    def run(self):
        """
        Run the genetic algorithm to find the best route.
        The algorithm iteratively selects parents, performs crossover and mutation, and updates the population."""
        population = self.init_population() # Initialize the population
        
        for gen in range(self.generations): # Iterate through generations
            new_population = [] # Create a new population
            while len(new_population) < self.population_size: # While the new population is not full
                parent1 = self.select(population) # Select first parent
                parent2 = self.select(population) # Select second parent
                child = self.crossover(parent1, parent2) # Perform crossover to create child
                if np.random.rand() < self.mutation_rate: # Mutate with probability mutation_rate
                    child = self.mutate(child) # Perform mutation
                new_population.append(child) # Add child to new population

            population = new_population # Update population
            best_in_pop = min(population, key=self.fitness) # Find the best route in the population
            best_cost = self.fitness(best_in_pop) # Calculate the cost of the best route
            self.cost_history.append(best_cost) # Store the cost history
            if best_cost < self.best_cost: # Update best route and cost if found a better one
                self.best_cost = best_cost # Update best cost
                self.best_route = best_in_pop # Update best route
        
        return self.best_route, self.best_cost # Return the best route and cost