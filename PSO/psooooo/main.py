import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from pso import PSO, distance_matrix
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('large.csv', header=None)
df=df[:200]
# Extract the first and second columns as numpy arrays
x = df.iloc[:, 0].to_numpy()
y = df.iloc[:, 1].to_numpy()

# Create a numpy array of shape (n, 2) where n is the number of cities
coords = np.stack((x, y), axis=1)

# Create a list of City objects from the coordinates
cities = [(x, y) for x, y in coords]

# Create a distance matrix using the City objects
distance_matrix = distance_matrix(cities)


class PSOGUI:
    def __init__(self, master):
        master.title("PSO Configuration")
        master.geometry("300x300")

        # Initialize PSO parameters
        self.iterations = tk.IntVar(value=1500)
        self.population_size = tk.IntVar(value=300)
        self.best_solution_probability = tk.DoubleVar(value=0.9)
        self.gbest_probability = tk.DoubleVar(value=0.01)

        # Create labels and entry fields for PSO parameters
        self.create_pso_parameters_widgets(master)

        # Create labels and entry fields for entering city coordinates
        self.create_city_coordinates_widgets(master)

        # Create submit button for running PSO
        submit_button = ttk.Button(master, text="Run PSO", command=self.create_pso)
        submit_button.grid(row=4, column=0, columnspan=2, pady=10)

        # Create submit button for reusing PSO model
        submit_button = ttk.Button(master, text="Reuse PSO Model", command=self.run_user_pso)
        submit_button.grid(row=7, column=0, columnspan=2, pady=10)

    def create_pso_parameters_widgets(self, master):
        # Labels and entry fields for PSO parameters
        iter_label = ttk.Label(master, text="Iterations:")
        iter_label.grid(row=0, column=0, padx=10, pady=5, sticky="e")

        self.iter_entry = ttk.Entry(master, textvariable=self.iterations)
        self.iter_entry.grid(row=0, column=1, padx=10, pady=5, sticky="nswe")

        pop_size_label = ttk.Label(master, text="Population Size:")
        pop_size_label.grid(row=1, column=0, padx=10, pady=5, sticky="e")

        self.pop_size_entry = ttk.Entry(master, textvariable=self.population_size)
        self.pop_size_entry.grid(row=1, column=1, padx=10, pady=5, sticky="nswe")

        best_solution_probability_label = ttk.Label(master, text="Best Solution Probability:")
        best_solution_probability_label.grid(row=2, column=0, padx=10, pady=5, sticky="e")

        self.best_solution_probability_entry = ttk.Entry(master, textvariable=self.best_solution_probability)
        self.best_solution_probability_entry.grid(row=2, column=1, padx=10, pady=5, sticky="nswe")

        gbest_prob_label = ttk.Label(master, text="Gbest Probability:")
        gbest_prob_label.grid(row=3, column=0, padx=10, pady=5, sticky="e")

        self.gbest_prob_entry = ttk.Entry(master, textvariable=self.gbest_probability)
        self.gbest_prob_entry.grid(row=3, column=1, padx=10, pady=5, sticky="nswe")

    def create_city_coordinates_widgets(self, master):
        # Labels and entry fields for entering city coordinates
        coords_label = ttk.Label(master, text="Enter City Coordinates:")
        coords_label.grid(row=5, column=0, columnspan=2, padx=10, pady=5, sticky="w")

        self.coords_entry = ttk.Entry(master)
        self.coords_entry.grid(row=6, column=0, columnspan=2, padx=10, pady=5, sticky="nswe")

    def run_user_pso(self):
        iterations = self.iterations.get()
        population_size = self.population_size.get()
        best_solution_probability = self.best_solution_probability.get()
        gbest_probability = self.gbest_probability.get()

        # Extract coordinates from entry and create Node objects
        coordinates = self.coords_entry.get()
        points = [tuple(map(float, point.split(','))) for point in coordinates.split(';')]
        cities2 = self.create_cities_from_points(points)

        # Assuming the PSO class is imported and instantiated as pso
        pso = PSO(iterations=iterations, pop_size=population_size, best_solution_probability=best_solution_probability,
                  gbest_probability=gbest_probability, nodes=cities2)
        pso.run()

        # Display results
        self.display_results(pso)

    def display_results(self, pso):
        print(f'cost: {pso.gbest.broute_cost}\t| gbest: {pso.gbest.broute}')
        initial_cost = pso.initial_cost
        best_cost = pso.gbest.broute_cost
        improvement = (initial_cost - best_cost) / initial_cost * 100
        print(f"Improvement: {improvement:.2f}%")
        x_list, y_list = [], []
        for node in pso.gbest.broute:
            x_list.append(node[0])
            y_list.append(node[1])
        x_list.append(pso.gbest.broute[0][0])
        y_list.append(pso.gbest.broute[0][1])
        fig = plt.figure(1)
        fig.suptitle(f"pso TSP \nImprovement: {improvement:.2f}%")

        plt.plot(x_list, y_list, 'ro')
        plt.plot(x_list, y_list)
        plt.show(block=True)

    def create_cities_from_points(self, points):
        # Create Node objects from the given points
        nodes = [(x, y) for x, y in points]
        return nodes

    def create_pso(self):
        iterations = self.iterations.get()
        population_size = self.population_size.get()
        best_solution_probability = self.best_solution_probability.get()
        gbest_probability = self.gbest_probability.get()

        # Assuming the PSO class is imported and instantiated as pso
        pso = PSO(iterations=iterations, pop_size=population_size, best_solution_probability=best_solution_probability,
                  gbest_probability=gbest_probability, cities=cities)
        pso.main_function()

        # Display results
        self.display_results(pso)

def main():
    root = tk.Tk()
    app = PSOGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
