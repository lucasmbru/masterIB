"""
Population evolution for the system described in the file '2-MasterEquation.ipynb'
"""

import numpy as np
import matplotlib.pyplot as plt
import random

class PopulationEvolution:
    def __init__(self, N, n_0, u, alpha, nu, mu, pmu) -> None:
        self.n_0 = n_0
        self.N = N
        Delta_t = u*pmu/mu
        self.delta_t = Delta_t/N
        self.u = u

        self.palpha = alpha*Delta_t/u      # Probability of migration event
        self.pnu = nu*Delta_t/u            # Probability of birth event
        self.pmu = mu*Delta_t/(1-u)        # Probability of death event

        self.T_ext = 0                     # Time of extintion
        self.N_ext = 0                     # Number of extintions

        self.population = [n_0]
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
        """ Function that makes the evolution of the population
        
        Parameters:
        -----------
        t_max: float
            Time of the evolution
        
        Returns:
        --------
        None
        """
        self.t = 0
        self.T_ext = 0
        self.N_ext = 1
        n = self.n_0
        self.population = [n]

        while True:
            self.t += self.delta_t
            if n == 0:
                self.T_ext += self.delta_t
            
            # Choice the type of interaction (u: single, 1-u: double)
            if random.random() < self.u:
                # Single interaction
                place = random.randint(0, self.N-1)
                specie = self.specie_choiced(place, n)
                if specie == "A":   # The place choiced is occupied by a specie A
                    if random.random() < self.pmu:
                        n -= 1  # Death event (A -> E)
                        if n == 0:  # The population is extinted with n = 0
                            self.N_ext += 1
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

    
    def make_single_evolution(self):
        n = self.n_0
        self.population = [n]
        self.t = 0
        
        while True:

            if n == 0:
                break

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


        mean = np.mean(self.population)
        std = np.std(self.population)
        medium = np.median(self.population)
        max_value = max(self.population)
        length = self.t

        return mean, medium, std, max_value, length

    def get_time(self):
        """ Function that returns the time evolution of the system
        
        Returns:
        --------
        np.array: Time evolution
        """
        return np.array([self.delta_t*i for i in range(len(self.population))])
        

    def get_population(self) -> list[int]:
        """ Function that returns the population evolution of the system
        Returns:
        --------
        list: Population evolution        
        """
        return self.population
    
    def get_distribution_of_n_neq_zero(self) -> np.array:
        """ Function that returns the distribution of n values after the evolution is performed without zeros
        
        Returns:
        --------
        np.array: Distribution of n values without zeros
        """
        return np.array([n for n in self.population if n != 0])
    
    def get_zeros(self) -> int:
        """ Function that returns the number of zeros in the population evolution
        
        Returns:
        --------
        int: Number of zeros in the population evolution
        """
        return self.population.count(0)

    def get_distribution_of_length_extinted(self) -> np.array:
        """ Function that returns the distribution of the duration of the extinted periods

        Returns:
        --------
        np.array: Distribution of the duration of the extinted periods
        """
        
        length_extinted_periods = []
        cuurrent_length = 0
        for n in self.population:
            if n == 0:
                cuurrent_length += 1
            else:
                if cuurrent_length != 0:
                    length_extinted_periods.append(cuurrent_length)
                    cuurrent_length = 0
        
        # This is for the case when the last value is zero. Should we add it?
        if cuurrent_length != 0:
            length_extinted_periods.append(cuurrent_length)
        
        return np.array(length_extinted_periods)
    
    def deletePopulationClass(self):
        # delete memory
        del self.population
    
class PlotPopulationEvolution(PopulationEvolution):
    """Subclass of PopulationEvolution for plotting an evolution of population"""
    def plot_evolution(self, ax) -> None:
        """ Function that plots the population evolution
        
        Parameters:
        -----------
        ax: matplotlib.axes.Axes
            Axes to plot the population evolution
        """
        ax.plot(self.get_time(), self.get_population(), lw = 0.5, color='blue')
        ax.set_xlabel(r"$t/\mu$")
        ax.set_yticks(range(0, self.N + 1, 2))
        ax.set_yticklabels(range(0, self.N + 1, 2))
        ax.set_ylabel(r"$n$")
        ax.set_ylim([0, self.N])
        ax.set_title(r"Population evolution, $\nu/\mu = $ " + str(self.pnu/self.pmu) + r", $\alpha/\mu = $" + str(self.palpha/self.pmu) + r", $N = $" + str(self.N)) 

    def plot_distribution_of_n(self, ax) -> None:
        """
        Function that plots the distribution of `n` values after evolution. The bin corresponding to `n = 0` is plotted with a different color compared to the rest of the bins.

        Parameters:
        -----------
        ax: matplotlib.axes.Axes
            Axes object on which the distribution of `n` values is plotted. The bin corresponding to `n = 0` is colored differently from the other bins.
        """

        counts, bins, patches = ax.hist(self.get_population(), bins=np.linspace(0, self.N, self.N+1), orientation='horizontal', align = "left", rwidth = 0.75, color='blue', density = True)
        patches[0].set_facecolor('red')
        for patch in patches[1:]:
            patch.set_facecolor('blue')
        ax.set_xlabel(r"$P(n)$")
        ax.set_title(r"Histogram of $n$")
        ax.set_yticks(range(0, self.N+1))
        ax.set_yticklabels(range(0, self.N+1))
        ax.set_ylim([-0.5, self.N+0.5])

    def plot_distribution_of_lengths_extinted(self, ax) -> None:
        """ Function that plots the distribution of the duration of the extinted periods
        
        Parameters:
        -----------
        ax: matplotlib.axes.Axes
            Axes to plot the distribution of the duration of the extinted periods
        """

        ax.hist(self.get_distribution_of_lenght_extinted(), color='red', orientation='horizontal', edgecolor='black', density = False, alpha = 0.75)
        ax.set_ylabel(r"Time of extinted periods/\mu")
        ax.set_xlabel(r"Number of extinted periods")
        ax.set_title(r"Histogram of the time of extinted periods")
        
    def plot_all(self, ax) -> None:
        """ Function that plots all the plots of the population evolution
        
        Parameters:
        -----------
        ax: matplotlib.axes.Axes
            Axes to plot the population evolution
        """
        self.plot_evolution(ax[0])
        self.plot_distribution_of_n(ax[1])
        self.plot_distribution_of_lenghts_extinted(ax[2])