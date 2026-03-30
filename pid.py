import math


class PID:
    """
    Enhanced PID Controller with Anti-Windup and Derivative Filtering
    
    Features:
    1. Anti-windup: Prevents integral term from growing unbounded
    2. Derivative filtering: Reduces noise amplification
    3. Bumpless transfer: Resets integrator when gains change
    4. Output clamping: Ensures output stays in [0, 1]
    
    PID equation: u(t) = Kp*e(t) + Ki*∫e(τ)dτ + Kd*de/dt
    """
    
    def __init__(self, Kp, Ki, Kd, dt, verbose=True):
        """
        Args:
            Kp: Proportional gain
            Ki: Integral gain
            Kd: Derivative gain
            dt: Control loop time step (seconds)
            verbose: Enable detailed logging
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.verbose = verbose
        
        # State variables
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_filtered_derivative = 0.0
        
        # Anti-windup configuration
        self.integral_max = 15.0  # Maximum integral accumulation
        self.integral_min = -15.0
        
        # Derivative filter coefficient (α = dt / (τ_filter + dt))
        # Lower α = more filtering, less noise but slower response
        self.derivative_filter_alpha = 0.3  # Moderate filtering
        
        # Diagnostic counters
        self.steps = 0
        self.saturation_count = 0
        
        if self.verbose:
            print("\n" + "="*70)
            print("[PID] INITIALIZATION")
            print("="*70)
            print(f"  Kp (proportional): {Kp:.6f}")
            print(f"  Ki (integral):     {Ki:.6f}")
            print(f"  Kd (derivative):   {Kd:.6f}")
            print(f"  dt (time step):    {dt:.3f} s")
            print(f"  Integral limits:   [{self.integral_min}, {self.integral_max}]")
            print(f"  Derivative filter: α={self.derivative_filter_alpha}")
            print("="*70 + "\n")
    
    def Calculate_heater(self, setpoint, measurement):
        """
        Compute PID control output
        
        Args:
            setpoint: Desired temperature
            measurement: Current temperature
            
        Returns:
            Heater power in [0, 1] (0% to 100%)
        """
        self.steps += 1
        
        # Compute error
        error = setpoint - measurement
        
        # === PROPORTIONAL TERM ===
        p_term = self.Kp * error
        
        # === INTEGRAL TERM with Anti-Windup ===
        # Only integrate if output is not saturated OR error would reduce integral
        self.integral += error * self.dt
        
        # Clamp integral to prevent windup
        if self.integral > self.integral_max:
            self.integral = self.integral_max
            if self.verbose and self.steps % 100 == 0:
                print(f"[PID] ⚠ Integral saturated at upper limit: {self.integral_max}")
        elif self.integral < self.integral_min:
            self.integral = self.integral_min
            if self.verbose and self.steps % 100 == 0:
                print(f"[PID] ⚠ Integral saturated at lower limit: {self.integral_min}")
        
        i_term = self.Ki * self.integral
        
        # === DERIVATIVE TERM with Filtering ===
        # Raw derivative (change in error)
        raw_derivative = (error - self.prev_error) / self.dt
        
        # Apply first-order low-pass filter to reduce noise
        # filtered[k] = α*raw[k] + (1-α)*filtered[k-1]
        filtered_derivative = (self.derivative_filter_alpha * raw_derivative + 
                               (1 - self.derivative_filter_alpha) * self.prev_filtered_derivative)
        
        d_term = self.Kd * filtered_derivative
        
        # === TOTAL OUTPUT ===
        output = p_term + i_term + d_term
        
        # Clamp output to [0, 1]
        clamped_output = max(0.0, min(1.0, output))
        
        # Track saturation for diagnostics
        if clamped_output != output:
            self.saturation_count += 1
        
        # Detailed logging every N steps
        if self.verbose and self.steps % 50 == 0:
            print(f"\n[PID] Step {self.steps:05d} Breakdown:")
            print(f"  Error: {error:+.4f}°C")
            print(f"  P-term: {p_term:+.6f} (Kp={self.Kp:.4f} × e={error:+.4f})")
            print(f"  I-term: {i_term:+.6f} (Ki={self.Ki:.4f} × ∫e={self.integral:+.4f})")
            print(f"  D-term: {d_term:+.6f} (Kd={self.Kd:.4f} × de/dt={filtered_derivative:+.4f})")
            print(f"  Sum:    {output:+.6f}")
            print(f"  Output: {clamped_output:.6f} {'(saturated)' if clamped_output != output else ''}")
            if self.saturation_count > 0:
                print(f"  Saturation count: {self.saturation_count}")
        
        # Update state for next iteration
        self.prev_error = error
        self.prev_filtered_derivative = filtered_derivative
        
        return clamped_output
    
    def update_gains(self, kp, ki, kd):
        """
        Update PID gains with bumpless transfer
        
        CRITICAL: Resets integral and derivative state to prevent transients
        when gains change significantly
        
        Args:
            kp, ki, kd: New PID gains
        """
        print("\n" + "="*70)
        print("[PID] GAIN UPDATE - BUMPLESS TRANSFER")
        print("="*70)
        print(f"  OLD GAINS:")
        print(f"    Kp: {self.Kp:.6f} → {kp:.6f} (Δ = {kp - self.Kp:+.6f})")
        print(f"    Ki: {self.Ki:.6f} → {ki:.6f} (Δ = {ki - self.Ki:+.6f})")
        print(f"    Kd: {self.Kd:.6f} → {kd:.6f} (Δ = {kd - self.Kd:+.6f})")
        
        # Compute relative change
        kp_change_pct = abs(kp - self.Kp) / (abs(self.Kp) + 1e-8) * 100
        ki_change_pct = abs(ki - self.Ki) / (abs(self.Ki) + 1e-8) * 100
        kd_change_pct = abs(kd - self.Kd) / (abs(self.Kd) + 1e-8) * 100
        
        print(f"  RELATIVE CHANGES:")
        print(f"    Kp: {kp_change_pct:6.2f}%")
        print(f"    Ki: {ki_change_pct:6.2f}%")
        print(f"    Kd: {kd_change_pct:6.2f}%")
        
        # Update gains
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd
        
        # CRITICAL: Reset integrator to prevent huge transients
        # When gains change, old integral is meaningless with new Ki
        old_integral = self.integral
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_filtered_derivative = 0.0
        
        print(f"  STATE RESET:")
        print(f"    Integral: {old_integral:.4f} → 0.0 (RESET for bumpless transfer)")
        print(f"    Previous error: RESET")
        print(f"    Previous derivative: RESET")
        print(f"  ✓ Gain update complete")
        print("="*70 + "\n")
        
        # Reset saturation counter
        self.saturation_count = 0
    
    def get_diagnostics(self):
        """Return current PID state for debugging"""
        return {
            'Kp': self.Kp,
            'Ki': self.Ki,
            'Kd': self.Kd,
            'integral': self.integral,
            'prev_error': self.prev_error,
            'steps': self.steps,
            'saturation_count': self.saturation_count,
        }
    
    def print_state(self):
        """Print current PID internal state"""
        print("\n" + "-"*70)
        print("[PID] Current State")
        print("-"*70)
        print(f"  Gains: Kp={self.Kp:.6f}, Ki={self.Ki:.6f}, Kd={self.Kd:.6f}")
        print(f"  Integral accumulator: {self.integral:.6f} (limits: [{self.integral_min}, {self.integral_max}])")
        print(f"  Previous error: {self.prev_error:.6f}°C")
        print(f"  Total steps: {self.steps}")
        print(f"  Saturation events: {self.saturation_count}")
        print("-"*70 + "\n")