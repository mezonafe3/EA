import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random

def path_cost(route):
    return sum([distance(city,route[index - 1]) for index, city in enumerate(route)])

def distance(city1, city2):
    return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)
def distance_matrix(cities):
    n = len(cities)
    dmatrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            dmatrix[i, j] = dmatrix[j, i] = distance(cities[i],cities[j])
    return dmatrix



class Particle:
    def __init__(self, route):
        self.current_route = route
        self.broute = route
        self.current_cost = path_cost(self.current_route)
        self.broute_cost = path_cost(self.current_route)
        self.velocity = []

    def update_cost_broute(self):
        self.current_cost = path_cost(self.current_route)
        if self.current_cost < self.broute_cost:
            self.broute = self.current_route
            self.broute_cost = self.current_cost

class PSO:

    def __init__(self, iterations, pop_size, gbest_probability, best_solution_probability, cities):
        self.cities = cities
        self.gbest = None
        self.gcost_iter = []
        self.initial_cost = 0
        self.iterations = iterations
        self.pop_size = pop_size
        self.particles = []
        self.gbest_probability = gbest_probability
        self.best_solution_probability = best_solution_probability
        solutions = self.initial_population()
        self.particles = [Particle(route=solution) for solution in solutions]

    def random_route(self):
        return random.sample(self.cities, len(self.cities))

    def initial_population(self):
        random_population = [self.random_route() for _ in range(self.pop_size )]
        return [*random_population]

    def main_function(self):
        self.gbest = min(self.particles, key=lambda p: p.broute_cost)
        self.initial_cost = self.gbest.broute_cost
        print(f"initial cost is {self.gbest.broute_cost}")
        plt.ion()
        plt.draw()
        for t in range(self.iterations):
            self.gbest = min(self.particles, key=lambda p: p.broute_cost)
            if t % 20 == 0:
                plt.figure(0)
                plt.plot(self.gcost_iter, 'g')
                plt.ylabel('Distance')
                plt.xlabel('Generation')
                fig = plt.figure(0)
                fig.suptitle('pso iter')
                x_list, y_list = [], []
                for city in self.gbest.broute:
                    x_list.append(city[0])
                    y_list.append(city[1])
                x_list.append(self.gbest.broute[0][0])
                y_list.append(self.gbest.broute[0][1])
                fig = plt.figure(1)
                fig.clear()
                fig.suptitle(f'pso TSP iter {t}')

                plt.plot(x_list, y_list, 'ro')
                plt.plot(x_list, y_list, 'g')
                plt.draw()
                plt.pause(.001)
            self.gcost_iter.append(self.gbest.broute_cost)

            for particle in self.particles:
                particle.velocity=[]
                temp_velocity = []
                gbest = self.gbest.broute[:]
                new_route = particle.current_route[:]

                for i in range(len(self.cities)):
                    if new_route[i] != particle.broute[i]:
                        swap = (i, particle.broute.index(new_route[i]), self.best_solution_probability)
                        temp_velocity.append(swap)
                        # new_route[swap[0]], new_route[swap[1]] = new_route[swap[1]], new_route[swap[0]]

                for i in range(len(self.cities)):
                    if new_route[i] != gbest[i]:
                        swap = (i, gbest.index(new_route[i]), self.gbest_probability)
                        temp_velocity.append(swap)
                        gbest[swap[0]], gbest[swap[1]] = gbest[swap[1]], gbest[swap[0]]

                particle.velocity = temp_velocity

                for swap in temp_velocity:
                    if random.random() <= swap[2]:
                        new_route[swap[0]], new_route[swap[1]] = new_route[swap[1]], new_route[swap[0]]
                particle.current_route = new_route
                particle.update_cost_broute()