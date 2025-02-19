import numpy as np
from ufl import exp

# FTF as n-tau model
class nTau:
    def __init__(self, n, tau):
        self.n = n # interaction index
        self.tau = tau # time delay

    def __call__(self, omega):
        # classic n-tau function
        return self.n * exp(1j * omega * self.tau)
    
    # calculate the derivative of FTF
    def derivative(self, omega):
        # first derivative of n-tau function after omega
        return self.n * (1j * self.tau) * exp(1j * omega * self.tau)

# FTF from experimental data in state-space form
class stateSpace:
    def __init__(self, S1, s2, s3, s4):
        # read matrices from experimental data
        self.A = S1
        self.b = s2
        self.c = s3
        self.d = s4
        print("- FTF initialized")
        self.Id = np.eye(*S1.shape)

    def __call__(self, omega):
        k = 0
        omega = np.conj(omega) # why should omega be conjugated before used in FTF?
        print("- FTF uses omega:", round(omega))
        # for k=0 this is standard state-space format
        # FTF = s3^transposed * (i \omega I - S1)^inverse * s2 + s4
        Mat = (- 1j) ** k * np.math.factorial(k) * \
            np.linalg.matrix_power(1j * omega * self.Id - self.A, - (k + 1))
        row = np.dot(self.c, Mat) # *s3
        H = np.dot(row, self.b) # *s2
        H += self.d # +s4 (1 dimensional)
        print("- matrix state space: FTF = ", round(np.conj(H[0][0]),3))
        return np.conj(H[0][0])
    
    # calculate the derivative of FTF
    def derivative(self, omega):
        k = 1
        omega = np.conj(omega)
        # for k=1 this is first derivative of state-space format
        Mat = (- 1j) ** k * np.math.factorial(k) * \
            np.linalg.matrix_power(1j * omega * self.Id - self.A, - (k + 1))
        row = np.dot(self.c, Mat)
        H = np.dot(row, self.b)
        return np.conj(H)