"""
Population evolution for the system described in the file '2-MasterEuqation.ipynb'
"""


import numpy as np
from tqdm import tqdm
import random

class PopulationEvolution:
    def __init__(self, N, Delta_t, n_0, u, alpha, nu, mu) -> None:
        self.n_0 = n_0
        self.N = N
        self.delta_t = Delta_t/N

        self.palpha = alpha*self.delta_t/u      # Probability of migration event
        self.pnu = nu*self.delta_t/u            # Probability of birth event
        self.pmu = mu*self.Delta_t/(1-u)        # Probability of death event

        self.population = []
        self.population[0] = n_0
        self.t = 0

    def specie_choiced(self, place, n) -> str:
        """ Function that returns the specie of the place 'place' 
        Parameters:
        -----------
        place: int
            Place of the population
        n: int
            Number of specie A
        
        Returns:
        --------
        str: "A" or "E"
        """
        if place < n:
            return "A"
        else:
            return "E"


    def make_evolution(self, t_max) -> None:
        self.t = 0
        n = self.n_0
        self.population = [n]

        while True:
            self.t += self.delta_t
            
            # Choice the type of interaction (u: single, 1-u: double)
            if random.random() < self.u:
                # Single interaction
                place = random.randint(0, self.N-1)
                specie = self.specie_choiced(place, n)
                if specie == "A":   # The place choiced is occupied by a specie A
                    if random.random() < self.pmu:
                        n -= 1  # Death event (A -> E)
                else:               # The place choiced is occupied by a specie E
                    if random.random() < self.palpha:
                        n += 1  # Expontanical migration event (E -> A)
            else: 
                # Double interaction
                place1, place2 = random.sample(range(self.N), 2)
                specie1 = self.specie_choiced(place1, n)
                specie2 = self.specie_choiced(place2, n)
                if specie1 == "A" and specie2 == "E":
                    # First place is occupied by A and the second by E
                    if random.random() < self.pnu:
                        n += 1  # Birth event (A + E -> A + A)
                # Any other combination, nothing happens

            self.population.append(n)       # Add the new value of n to the population list 

            if self.t >= t_max:
                break

    def get_population(self) -> list[int]:
        """ Function that returns the population evolution of the system
        Returns:
        --------
        list: Population evolution        
        """
        return self.population
