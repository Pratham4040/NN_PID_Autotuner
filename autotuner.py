import numpy as np
from collections import deque


def _simulate_step_response_fnn(nn_model, T_current, U_current, num_steps, pre_steps, u_step, verbose):
    """
    Step response simulation for the feed-forward 2-step model.
    """
    T_sim = [T_current, T_current]  # Need two initial points
    U_sim = [U_current, U_current]

    sim_min = T_current - 30.0
    sim_max = T_current + 120.0

    for i in range(num_steps):
        T_prev = T_sim[-1]
        T_prev2 = T_sim[-2]
        U_prev = U_sim[-1]
        U_prev2 = U_sim[-2]

        T_next = nn_model.predict(T_prev, T_prev2, U_prev, U_prev2)
        if not np.isfinite(T_next):
            if verbose:
                print("    ❌ Simulation diverged (non-finite prediction)")
                print("=" * 70 + "\n")
            return None, None
        if T_next < sim_min or T_next > sim_max:
            if verbose:
                print(f"    ❌ Simulation diverged (T_next out of bounds: {T_next:.3f}°C)")
                print("=" * 70 + "\n")
            return None, None

        T_sim.append(T_next)
        u_next = U_current if i < pre_steps else u_step
        U_sim.append(u_next)

    # Remove initial conditions
    T_sim = np.array(T_sim[2:])
    U_sim = np.array(U_sim[2:])

    if verbose:
        print(f"    Simulation complete: {len(T_sim)} points")
        print(f"    Temperature range: {T_sim.min():.3f} to {T_sim.max():.3f}°C")
        print(f"    Final temp: {T_sim[-1]:.3f}°C")

    return T_sim, U_sim


def _simulate_step_response_lstm(nn_model, T_current, U_current, num_steps, pre_steps, u_step, verbose):
    """
    Step response simulation for the LSTM sequence model.

    The LSTM expects a full sequence of (T, U) pairs where U is the
    command applied after measuring T. To simulate, we keep a rolling
    buffer, inject the step input on the latest pair, predict T_next,
    and then append (T_next, U) for the next iteration.
    """
    seq_len = nn_model.sequence_length
    history = deque([(T_current, U_current)] * seq_len, maxlen=seq_len)

    T_sim = []
    U_sim = []

    sim_min = T_current - 30.0
    sim_max = T_current + 120.0

    for i in range(num_steps):
        # Apply step after pre_steps; otherwise hold current input.
        u_next = U_current if i < pre_steps else u_step

        # Build the sequence with the current input placed on the last pair.
        # This mirrors online operation where U_k is computed after T_k.
        seq = list(history)
        seq[-1] = (seq[-1][0], u_next)

        T_next = nn_model.predict_from_sequence(seq)
        if not np.isfinite(T_next):
            if verbose:
                print("    ❌ Simulation diverged (non-finite prediction)")
                print("=" * 70 + "\n")
            return None, None
        if T_next < sim_min or T_next > sim_max:
            if verbose:
                print(f"    ❌ Simulation diverged (T_next out of bounds: {T_next:.3f}°C)")
                print("=" * 70 + "\n")
            return None, None

        T_sim.append(T_next)
        U_sim.append(u_next)

        # Update the rolling buffer with the newly predicted state.
        history = deque(seq, maxlen=seq_len)
        history.append((T_next, u_next))

    T_sim = np.array(T_sim)
    U_sim = np.array(U_sim)

    if verbose:
        print(f"    Simulation complete: {len(T_sim)} points")
        print(f"    Temperature range: {T_sim.min():.3f} to {T_sim.max():.3f}°C")
        print(f"    Final temp: {T_sim[-1]:.3f}°C")

    return T_sim, U_sim


def estimate_parameters(nn_model, T_current, U_current, dt, verbose=True):
    """
    Estimate first-order plant parameters using NN-simulated step response
    
    METHOD:
    Instead of taking finite differences on the NN (which gives NN derivatives, 
    not plant parameters), we:
    1. Use the trained NN to simulate a step response
    2. Fit a discrete first-order model: T[k+1] = a*T[k] + b*U[k] + c
    3. Extract physical parameters from the fitted model
    
    Args:
        nn_model: Trained neural plant model
        T_current: Current temperature
        U_current: Current heater power
        dt: Time step
        verbose: Print detailed diagnostics
        
    Returns:
        (a, b): Discrete model parameters, or (None, None) if estimation fails
    """
    
    if verbose:
        print("\n" + "="*70)
        print("[Parameter Estimation] STARTING")
        print("="*70)
    
    # Step 1: Check if model is trained enough
    is_ready, loss, status = nn_model.get_training_quality(loss_threshold=0.015)
    
    if verbose:
        print(f"  Model status: {status}")
        print(f"  Training loss: {loss:.6f}")
    
    if not is_ready:
        if verbose:
            print(f"  ❌ Model not ready for parameter extraction")
            print("="*70 + "\n")
        return None, None
    
    # Step 2: Simulate step response using trained NN
    num_steps = 80  # Total simulated steps
    pre_steps = 10  # Steps to hold current input before stepping
    u_step = 0.6    # Default step input (60% power)
    if abs(U_current - u_step) < 0.1:
        u_step = 0.2 if U_current > 0.5 else 0.8

    if verbose:
        print(f"\n  Simulating step response from current state:")
        print(f"    Initial temp: {T_current:.3f}°C")
        print(f"    Step input: {U_current:.2f} -> {u_step:.2f} (pre_steps={pre_steps})")
    
    # Run simulation (FFN or LSTM depending on model type)
    if getattr(nn_model, "is_sequence_model", False) or hasattr(nn_model, "sequence_length"):
        T_sim, U_sim = _simulate_step_response_lstm(
            nn_model,
            T_current,
            U_current,
            num_steps,
            pre_steps,
            u_step,
            verbose,
        )
    else:
        T_sim, U_sim = _simulate_step_response_fnn(
            nn_model,
            T_current,
            U_current,
            num_steps,
            pre_steps,
            u_step,
            verbose,
        )

    if T_sim is None or U_sim is None:
        return None, None
    
    # Step 3: Fit first-order discrete model
    # Model: T[k+1] = a*T[k] + b*U[k] + c
    if verbose:
        print(f"\n  Fitting first-order model: T[k+1] = a*T[k] + b*U[k] + c")
    
    if len(T_sim) < 10:
        if verbose:
            print(f"  ❌ Insufficient simulation data ({len(T_sim)} points)")
            print("="*70 + "\n")
        return None, None
    
    # Prepare least squares problem
    T_k = T_sim[:-1]    # T[k]
    T_k1 = T_sim[1:]    # T[k+1]
    U_k = U_sim[:-1]    # U[k]
    
    # Build design matrix: [T[k], U[k], 1] for each sample
    A_matrix = np.column_stack([T_k, U_k, np.ones_like(T_k)])
    
    try:
        # Solve: [a, b, c] = argmin ||A*[a,b,c] - T[k+1]||^2
        params, residuals, rank, s = np.linalg.lstsq(A_matrix, T_k1, rcond=None)
        a, b, c = params
        
        # Compute fit quality
        fit_error = np.sqrt(residuals[0] / len(T_k1)) if len(residuals) > 0 else 0
        
        if verbose:
            print(f"    ✓ Least squares fit complete")
            print(f"    Parameters: a={a:.6f}, b={b:.6f}, c={c:.6f}")
            print(f"    Fit RMSE: {fit_error:.6f}°C")
            print(f"    Matrix rank: {rank}/3")
        
    except np.linalg.LinAlgError as e:
        if verbose:
            print(f"  ❌ Least squares failed: {e}")
            print("="*70 + "\n")
        return None, None
    
    # Step 4: Validate parameters are physically reasonable
    if verbose:
        print(f"\n  Validating parameters:")

    # Soft clamp for near-unstable fits (LSTM rollouts can drift slightly > 1)
    a_tolerance = 0.02
    a_clamp = 0.999
    if 1.0 < a < 1.0 + a_tolerance:
        if verbose:
            print(f"    ⚠ Parameter 'a' slightly > 1: {a:.6f} -> clamped to {a_clamp}")
            print(f"       This keeps tau positive while preserving near-unity dynamics")
        a = a_clamp
    
    # Check 1: Stability (0 < a < 1 for stable discrete system)
    if not (0 < a < 1):
        if verbose:
            print(f"    ❌ Parameter 'a' out of range: {a:.6f} (must be in (0, 1))")
            print(f"       a ≥ 1 means unstable system (exponential growth)")
            print(f"       a ≤ 0 means non-physical dynamics")
            print("="*70 + "\n")
        return None, None
    
    # Check 2: System must respond to input
    if abs(b) < 1e-6:
        if verbose:
            print(f"    ❌ Parameter 'b' too small: {b:.6f}")
            print(f"       System doesn't respond to heater input")
            print("="*70 + "\n")
        return None, None
    
    # Check 3: Parameter 'a' should be close to 1 for thermal systems
    # Thermal systems have slow dynamics, so discrete a ≈ exp(-dt/tau) ≈ 0.95-0.995
    if a < 0.8:
        if verbose:
            print(f"    ⚠ WARNING: Parameter 'a' unusually low: {a:.6f}")
            print(f"       Expected ~0.9-0.99 for thermal systems")
            print(f"       This suggests very fast dynamics (unusual for heating)")
    
    if verbose:
        print(f"    ✓ Parameter 'a': {a:.6f} (valid stable range)")
        print(f"    ✓ Parameter 'b': {b:.6f} (valid response)")
        print(f"    ✓ Parameters validated successfully")
        print("="*70 + "\n")
    
    return a, b


def compute_tau_K(a, b, dt, verbose=True):
    """
    Compute continuous-time parameters (time constant τ, gain K) from discrete parameters
    
    THEORY:
    Continuous first-order system: τ*dT/dt = -T + K*U
    Discretized (zero-order hold): T[k+1] = exp(-dt/τ)*T[k] + K*(1-exp(-dt/τ))*U[k]
    
    Comparing with fitted model T[k+1] = a*T[k] + b*U[k]:
        a = exp(-dt/τ)  →  τ = -dt / ln(a)
        b = K*(1-a)     →  K = b / (1-a)
    
    Args:
        a, b: Discrete model parameters
        dt: Time step
        verbose: Print diagnostics
        
    Returns:
        (tau, K): Time constant and gain, or (None, None) if invalid
    """
    
    if verbose:
        print("\n" + "="*70)
        print("[Continuous Parameter Conversion]")
        print("="*70)
    
    if a is None or b is None:
        if verbose:
            print("  ❌ Cannot compute τ and K: discrete parameters are None")
            print("="*70 + "\n")
        return None, None
    
    if verbose:
        print(f"  Input: a={a:.6f}, b={b:.6f}, dt={dt:.3f}s")
    
    # Validate input constraints
    if a <= 0 or a >= 1:
        if verbose:
            print(f"  ❌ Invalid 'a' parameter: {a:.6f} (must be in (0, 1))")
            print("="*70 + "\n")
        return None, None
    
    # Compute time constant
    tau = -dt / np.log(a)
    
    # Compute steady-state gain
    K = b / (1 - a)
    
    if verbose:
        print(f"\n  Computed continuous parameters:")
        print(f"    Time constant τ: {tau:.3f} seconds")
        print(f"    Steady-state gain K: {K:.3f} °C per unit power")
    
    # Sanity checks
    checks_passed = True
    
    # Check 1: Time constant should be reasonable for thermal systems
    if tau < dt:
        if verbose:
            print(f"    ⚠ WARNING: τ < dt ({tau:.3f} < {dt:.3f})")
            print(f"       System responds faster than sampling - may need faster sampling")
        checks_passed = False
    
    if tau > 1000:
        if verbose:
            print(f"    ⚠ WARNING: τ very large ({tau:.1f}s)")
            print(f"       Heating chamber shouldn't have such slow dynamics")
        checks_passed = False
    
    # Check 2: Gain should be reasonable
    if abs(K) > 100:
        if verbose:
            print(f"    ⚠ WARNING: Gain very large (|K| = {abs(K):.1f})")
            print(f"       100% power shouldn't cause {abs(K):.1f}°C change")
        checks_passed = False
    
    if abs(K) < 0.1:
        if verbose:
            print(f"    ⚠ WARNING: Gain very small (|K| = {abs(K):.3f})")
            print(f"       Heater seems ineffective")
        checks_passed = False
    
    # Check 3: Sign check (for heating, K should be positive)
    if K < 0:
        if verbose:
            print(f"    ⚠ WARNING: Negative gain (K = {K:.3f})")
            print(f"       For heating systems, gain should be positive")
            print(f"       This might indicate inverse action or model error")
    
    # Reasonable ranges for heating chamber:
    # τ: 10-500 seconds (thermal lag)
    # K: 1-50 °C (temperature rise per unit power)
    tau_ok = dt < tau < 500
    K_ok = 0.5 < abs(K) < 50
    
    if tau_ok and K_ok:
        if verbose:
            print(f"    ✓ Parameters within expected ranges for thermal systems")
            print("="*70 + "\n")
        return tau, K
    else:
        if verbose:
            print(f"\n    {'✓' if tau_ok else '❌'} Time constant check: {tau:.1f}s (expect {dt:.1f}s < τ < 500s)")
            print(f"    {'✓' if K_ok else '❌'} Gain check: {abs(K):.1f}°C (expect 0.5 < |K| < 50)")
            if checks_passed:
                print(f"    ⚠ Parameters outside typical range but accepting anyway")
                print("="*70 + "\n")
                return tau, K
            else:
                print(f"    ❌ Parameters failed validation")
                print("="*70 + "\n")
                return None, None


def imc_pid(K, tau, L=0.7, lam=3.0, verbose=True):
    """
    Compute PID gains using Internal Model Control (IMC) tuning rules
    
    THEORY:
    For FOPDT model (First Order Plus Dead Time): K, τ, L
    IMC tuning gives:
        Kp = τ / (K * (λ + L))
        Ki = Kp / τ
        Kd = Kp * L
    
    Where λ is the closed-loop time constant (tuning parameter)
    
    Args:
        K: Process gain (°C per unit power)
        tau: Time constant (seconds)
        L: Dead time estimate (seconds)
        lam: Desired closed-loop time constant (seconds, larger = more conservative)
        verbose: Print diagnostics
        
    Returns:
        (Kp, Ki, Kd): PID gains, or None if computation fails
    """
    
    if verbose:
        print("\n" + "="*70)
        print("[IMC PID Tuning]")
        print("="*70)
    
    if K is None or tau is None:
        if verbose:
            print("  ❌ Cannot compute PID gains: K or τ is None")
            print("="*70 + "\n")
        return None
    
    if abs(K) < 1e-6:
        if verbose:
            print(f"  ❌ Cannot compute PID gains: K too small ({K:.6f})")
            print("="*70 + "\n")
        return None
    
    if verbose:
        print(f"  Process parameters:")
        print(f"    Gain K: {K:.3f} °C")
        print(f"    Time constant τ: {tau:.3f} s")
        print(f"    Dead time L: {L:.3f} s")
        print(f"  Tuning parameter:")
        print(f"    Lambda λ: {lam:.3f} s (desired closed-loop speed)")
    
    # Adaptive lambda: adjust based on tau for better robustness
    # Faster systems (small tau) need larger lambda for stability
    adjusted_lam = max(lam, tau * 0.3)
    
    if adjusted_lam != lam and verbose:
        print(f"    → Adjusted λ: {adjusted_lam:.3f} s (for stability)")
    
    # Compute IMC-PID gains
    kp = tau / (K * (adjusted_lam + L))
    ki = kp / tau
    kd = kp * L
    
    if verbose:
        print(f"\n  Computed PID gains (IMC method):")
        print(f"    Kp (proportional): {kp:.6f}")
        print(f"    Ki (integral):     {ki:.6f}")
        print(f"    Kd (derivative):   {kd:.6f}")
    
    # Validation and adjustments
    if kp < 0:
        if verbose:
            print(f"\n    ⚠ Negative Kp detected (inverse action)")
            print(f"    → Inverting all gains for direct action")
        kp, ki, kd = -kp, -ki, -kd
    
    # Limit derivative gain to prevent noise amplification
    # Rule of thumb: Kd should not exceed 2*Kp for noisy measurements
    max_kd = kp * 2.0
    if kd > max_kd:
        if verbose:
            print(f"\n    ⚠ Kd too large ({kd:.6f}), limiting to {max_kd:.6f}")
        kd = max_kd
    
    # Final validation ranges
    kp_ok = 0.01 < kp < 1000
    ki_ok = 0 <= ki < 200
    kd_ok = 0 <= kd < 200
    
    if verbose:
        print(f"\n  Gain validation:")
        print(f"    {'✓' if kp_ok else '❌'} Kp: {kp:.3f} (expect 0.01-1000)")
        print(f"    {'✓' if ki_ok else '❌'} Ki: {ki:.3f} (expect 0-200)")
        print(f"    {'✓' if kd_ok else '❌'} Kd: {kd:.3f} (expect 0-200)")
    
    if kp_ok and ki_ok and kd_ok:
        if verbose:
            print(f"    ✓ All gains validated")
            print("="*70 + "\n")
        return kp, ki, kd
    else:
        if verbose:
            print(f"    ❌ Gains outside acceptable ranges")
            print("="*70 + "\n")
        return None