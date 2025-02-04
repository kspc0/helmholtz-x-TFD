import numpy as np
from ufl import exp

# n-tau model definition
class nTau:
    def __init__(self, n, tau):
        self.n = n
        self.tau = tau

    def __call__(self, omega):
        # n-tau function
        return self.n * exp(1j * omega * self.tau)
    
    # calculate the derivative of FTF -> needed for perturbation theory
    def derivative(self, omega):
        # first derivative of n-tau function
        return self.n * (1j * self.tau) * exp(1j * omega * self.tau) 

# if FTF from experimental data wants to be used:
class stateSpace:
    def __init__(self, S1, s2, s3, s4):
        self.A = S1
        self.b = s2
        self.c = s3
        self.d = s4
        print("- FTF initialized")
        self.Id = np.eye(*S1.shape)

    def __call__(self, omega):
        k = 0
        omega = np.conj(omega) # why should omega be conjugated before used in FTF?
        print("- FTF uses omega:",round(omega))
        Mat = (- 1j) ** k * np.math.factorial(k) * \
            np.linalg.matrix_power(1j * omega * self.Id - self.A, - (k + 1))
        row = np.dot(self.c, Mat) # *s3
        H = np.dot(row, self.b) # *s2
        H += self.d # +s4 (1 dimensional)
        #print("FTF matrix state space:",H)
        print("- matrix state space: FTF=",round(np.conj(H[0][0]),7)) # print to 7 digits after comma
        return np.conj(H[0][0]) # changed manually here to "(H[0])" because scalar element H doesnt have two indices

    def derivative(self, omega):
        #print("- calculating derivative of FTF..")
        k = 1
        omega = np.conj(omega)
        Mat = (- 1j) ** k * np.math.factorial(k) * \
            np.linalg.matrix_power(1j * omega * self.Id - self.A, - (k + 1))
        row = np.dot(self.c, Mat)
        H = np.dot(row, self.b)
        return np.conj(H) # [0][0] two indeces for H needed?