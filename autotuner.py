import numpy as np
def estimate_parameters(nn_model, T, U):
    eps = 0.01
    base = nn_model.predict(T, T, U, U)
    T_pert = nn_model.predict(T + eps, T, U, U)
    U_pert = nn_model.predict(T, T, U + eps, U)
    a = (T_pert - base) / eps
    b = (U_pert - base) / eps
    print("got a and b")
    print(a,b)
    return a, b

def compute_tau_K(a, b, dt):
    if a <= 0 or a >= 1:
        return None, None
    tau = -dt / np.log(a)
    K = b / (1 - a)
    print("got tau and K")
    print(tau,K)
    return tau, K
def imc_pid(K, tau, L=0.7, lam=3):
    if K == 0 or tau is None:
        return None
    kp = tau / (K * (lam + L))
    ki = kp / tau
    kd = kp * L
    print("calculated kp ki kd")
    print(kp,ki,kd)
    return kp, ki, kd