import numpy as np
class ThermalSimulator:
    def __init__(self):
        self.T = 25
        self.ambient = 25
        self.tau = 40
        self.K = 20
        self.dt = 0.7
    def step(self, heater_power):
        heater_power = max(0, min(1, heater_power))
        dT = (-(self.T - self.ambient) + self.K * heater_power) / self.tau
        self.T = self.T + dT * self.dt
        # sensor noise
        sensor = self.T + np.random.normal(0,0.02)
        print("Data sent by Simulator")
        return sensor
# simulate = ThermalSimulator()
# print(simulate.step(4))