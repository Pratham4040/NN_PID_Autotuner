from simulate_chamber import ThermalSimulator
from pid import PID
from neural_model import NeuralPlantModel
from autotuner import estimate_parameters, compute_tau_K, imc_pid
from visualize import plot_all

sim = ThermalSimulator()
pid = PID(69.69, 68.69, 69, sim.dt)

nn_model = NeuralPlantModel()

setpoint = 37

temps = []
powers = []
losses = []          # NN MSE loss each step
errors = []          # control error each step

# track PID gain updates: list of (step, Kp, Ki, Kd)
gain_history = [(0, pid.Kp, pid.Ki, pid.Kd)]

# track plant parameter estimates: list of (step, tau, K)
params_history = []

heater = 0

for step in range(1000):

    temp = sim.step(heater)
    heater = pid.Calculate_heater(setpoint, temp)

    temps.append(temp)
    powers.append(heater)
    errors.append(setpoint - temp)

    if len(temps) >= 3:
        nn_model.add_sample(
            temps[-3],
            temps[-2],
            powers[-2],
            powers[-3],
            temps[-1]
        )

    loss_val = nn_model.train_step()
    losses.append(loss_val if loss_val is not None else float('nan'))

    if step % 50 == 0 and step > 100:

        a, b = estimate_parameters(nn_model, temp, heater)
        tau, K = compute_tau_K(a, b, sim.dt)

        if tau and K:
            params_history.append((step, tau, K))
            gains = imc_pid(K, tau)
            if gains:
                pid.update_gains(*gains)
                gain_history.append((step, pid.Kp, pid.Ki, pid.Kd))

    print(f"Step {step} | Temp {temp:.2f} | Heater {heater:.2f}")

print("\nSimulation complete. Generating plots...")
plot_all(
    temps=temps,
    powers=powers,
    errors=errors,
    losses=losses,
    gain_history=gain_history,
    params_history=params_history,
    setpoint=setpoint
)