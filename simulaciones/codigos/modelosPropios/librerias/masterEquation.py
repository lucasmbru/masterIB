" Solver of the master equation for the population evolution descripted in the file '2-MasterEquation.ipynb' "

import numpy as np
from scipy.linalg import null_space

class MasterEquation:
    def __init__(self, N) -> None:
        self.N = N
        self.mu = 0
        self.nu = 0
        self.alpha = 0

        self.nullSpace = 0
        self.P0 = 0
        self.mean = 0
        self.real_mean = 0  # Real mean of the distribution considering P(0)
        self.std_deviation = 0
        self.real_std_deviation = 0  # Real standard deviation of the distribution considering P(0)

        self.Matrix = np.zeros((N+1, N+1))

    def set_parameters(self, mu, nu, alpha) -> None:
        """ Function that sets the parameters of the model
        Parameters:
        -----------
        mu: float
            Death rate
        nu: float
            Birth rate
        alpha: float
            Migration rate
        Returns:
        --------
        None
        """
        self.mu = mu
        self.nu = nu
        self.alpha = alpha

    def set_matrix(self) -> None:
        """ Function that sets the matrix of the model with the parameters given

        Returns:
        --------
        None
        """
        for i in range(self.N+1):
            # Terms of the diagonal
            self.Matrix[i, i] = -(self.alpha*(self.N - i) + self.nu*i*(self.N - i)/(self.N - 1) + self.mu*i)
            # Terms of the upper diagonal
            if i < self.N:
                self.Matrix[i, i+1] = self.mu*(i+1)
            # Terms of the lower diagonal
            if i > 0:
                self.Matrix[i, i-1] = self.alpha * (self.N - i + 1) + self.nu * (i-1) * (self.N - i + 1) / (self.N - 1)
    
    def get_nullSpace(self) -> None:
        """ Function that gets the null space normalized of the matrix. This null space is the stationary distribution of the system

        Returns:
        --------
        None
        """
        aux_vector = null_space(self.Matrix)
        self.nullSpace = aux_vector/aux_vector.sum()
        self.P0 = self.nullSpace[0][0]
        aux_mean = np.sum([i*self.nullSpace[i][0] for i in range(self.N+1)])
        self.mean = aux_mean/(self.N*(1-self.P0))
        self.real_mean = aux_mean/self.N
        aux_std_deviation = np.sum([i**2*self.nullSpace[i][0] for i in range(self.N+1)])
        self.std_deviation = np.sqrt((aux_std_deviation)/(self.N*self.N*(1-self.P0)) - self.mean**2)
        self.real_std_deviation = np.sqrt((aux_std_deviation)/(self.N*self.N) - self.real_mean**2)
    
    # delete memory
    def __del__(self):
        del self.N
        del self.mu
        del self.nu
        del self.alpha
        del self.nullSpace
        del self.P0
        del self.Matrix

class P0vsNu(MasterEquation):
    def __init__(self, N, nu_min, nu_max, N_nus, mu, alpha) -> None:
        super().__init__(N)
        self.mu = mu
        self.alpha = alpha
        self.nus = np.linspace(nu_min, nu_max, N_nus)
        self.P0s = np.zeros(N_nus)
        self.means = np.zeros(N_nus)
        self.real_means = np.zeros(N_nus)
        self.std_deviations = np.zeros(N_nus)
        self.real_std_deviations = np.zeros(N_nus)

    def makeP0vsNu(self) -> None:
        """
        Function that calculates the stationary distribution for different values of nu

        Returns:
        --------
        None
        """
        for i, nu in enumerate(self.nus):
            self.set_parameters(self.mu, nu, self.alpha)
            self.set_matrix()
            self.get_nullSpace()
            self.P0s[i] = self.P0
            self.means[i] = self.mean
            self.real_means[i] = self.real_mean
            self.std_deviations[i] = self.std_deviation
            self.real_std_deviations[i] = self.real_std_deviation

    def getnuVector(self) -> np.array:
        """
        Function that returns the vector of nu values

        Returns:
        --------
        nus: np.array
            Vector of nu values
        """
        return self.nus
    
    def getP0Vector(self) -> np.array:
        """
        Function that returns the vector of P0 values

        Returns:
        --------
        P0s: np.array
            Vector of P0 values
        """
        return self.P0s
    
    def getMeanVector(self) -> float:
        """
        Function that returns the mean value of the distribution

        Returns:
        --------
        mean: float
            Mean of the distribution
        """
        return self.means

    def getCaracteristicNu(self, r) -> float:
        """
        Function that return the nu value which is closest to the value P0 = r. This will be the characteristic value of nu to quantify the behavior of the system

        Returns:
        --------
        nu: float
            Value of nu

        Inputs:
        -------
        r: float
            Value of P0
        """
        return self.nus[np.argmin(np.abs(self.P0s-r))]
    
    def getRealMeanVector(self) -> float:
        """
        Function that returns the real mean value of the distribution considering P(0)

        Returns:
        --------
        real_mean: float
            Real mean of the distribution
        """
        return self.real_means
    
    def getStdDeviationVector(self) -> float:
        """
        Function that returns the standard deviation of the distribution considering P(0)

        Returns:
        --------
        std_deviation: float
            Standard deviation of the distribution
        """
        return self.std_deviations
    
    def getRealStdDeviationVector(self) -> float:
        """
        Function that returns the real standard deviation of the distribution considering P(0)

        Returns:
        --------
        real_std_deviation: float
            Real standard deviation of the distribution
        """
        return self.real_std_deviations