import simulate_chamber
temps = []
powers = []
sim = simulate_chamber.ThermalSimulator()
heater_power = 0.8
for i in range(200):
    temp = sim.step(heater_power)
    temps.append(temp)
    powers.append(heater_power)
print(temps)
print(powers)