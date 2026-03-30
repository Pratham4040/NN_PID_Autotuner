import argparse
import csv
import json
import math
import time
import urllib.error
import urllib.request
from datetime import datetime

from pid import PID


MAX_PWM_CAP = 125


def http_get_text(url, timeout_s):
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return response.read().decode("utf-8").strip()


def http_post_text(url, body_text, timeout_s):
    request = urllib.request.Request(
        url,
        data=body_text.encode("utf-8"),
        method="POST",
        headers={"Content-Type": "text/plain"},
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return response.read().decode("utf-8").strip()


def read_temp(esp_ip, timeout_s):
    text = http_get_text(f"http://{esp_ip}/temp", timeout_s)
    safety = False

    if "," in text:
        temp_text, tail = text.split(",", 1)
        safety = "SAFETY" in tail.upper()
    else:
        temp_text = text

    temp_c = float(temp_text)
    return temp_c, safety


def write_pwm(esp_ip, pwm, timeout_s):
    pwm = int(max(0, min(MAX_PWM_CAP, pwm)))
    reply = http_post_text(f"http://{esp_ip}/pwm", str(pwm), timeout_s)
    return reply


def read_status(esp_ip, timeout_s):
    text = http_get_text(f"http://{esp_ip}/status", timeout_s)
    return json.loads(text)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PID control loop on PC using ESP32 sensor/actuator endpoints."
    )
    parser.add_argument("--esp-ip", required=True, help="ESP32 IP address (example: 192.168.137.42)")
    parser.add_argument("--setpoint", type=float, default=35.0, help="Target chamber temperature in C")
    parser.add_argument("--dt", type=float, default=1.0, help="Control period in seconds (CHANGED: was 0.7, now 1.0 for thermal systems)")

    parser.add_argument("--kp", type=float, default=25.0, help="Initial Kp (CHANGED: was 69.69, now 25.0 - less aggressive)")
    parser.add_argument("--ki", type=float, default=8.0, help="Initial Ki (CHANGED: was 68.69, now 8.0 - less aggressive)")
    parser.add_argument("--kd", type=float, default=3.0, help="Initial Kd (CHANGED: was 69.0, now 3.0 - less aggressive)")

    parser.add_argument("--steps", type=int, default=0, help="Number of control steps, 0 means run forever")
    parser.add_argument("--duration", type=float, default=0.0, help="Run duration in seconds, 0 means no limit")

    parser.add_argument("--host-max-temp", type=float, default=38.0, help="PC-side safety cutoff in C")
    parser.add_argument("--request-timeout", type=float, default=1.5, help="HTTP timeout in seconds")
    parser.add_argument("--max-failures", type=int, default=8, help="Consecutive HTTP failures before stop")

    parser.add_argument("--csv", default="esp32_run_log.csv", help="Path for CSV run log")
    parser.add_argument("--status-every", type=int, default=20, help="Print /status every N steps, 0 disables")

    parser.add_argument("--autotune", action="store_true", help="Enable online NN-based autotuning")
    parser.add_argument("--retune-every", type=int, default=150, help="Retune period in steps (CHANGED: was 50, now 150 - less frequent)")
    parser.add_argument("--retune-start", type=int, default=250, help="Start retuning after this step (CHANGED: was 100, now 250 - wait for NN convergence)")
    parser.add_argument("--imc-L", type=float, default=1.2, help="IMC dead-time estimate (CHANGED: was 0.7, now 1.2)")
    parser.add_argument("--imc-lambda", type=float, default=6.0, help="IMC lambda (CHANGED: was 3.0, now 6.0 - more conservative)")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.dt <= 0:
        raise ValueError("--dt must be > 0")

    # Initialize PID controller with verbose logging
    pid = PID(args.kp, args.ki, args.kd, args.dt, verbose=True)
    
    # Initialize autotuning components
    nn_model = None
    estimate_parameters = None
    compute_tau_K = None
    imc_pid = None
    
    if args.autotune:
        try:
            from autotuner import compute_tau_K as _compute_tau_K
            from autotuner import estimate_parameters as _estimate_parameters
            from autotuner import imc_pid as _imc_pid
            from neural_model import NeuralPlantModel
        except ImportError as exc:
            raise ImportError(
                "--autotune requires neural_model/autotuner dependencies (torch, numpy)."
            ) from exc

        estimate_parameters = _estimate_parameters
        compute_tau_K = _compute_tau_K
        imc_pid = _imc_pid
        
        # Create neural model with verbose diagnostics and proper normalization
        nn_model = NeuralPlantModel(
            temp_ref=args.setpoint,  # Use setpoint as reference temperature
            temp_scale=10.0,          # Expected deviation range ±10°C
            verbose=True              # Enable detailed logging
        )
        
        print("\n" + "🤖"*35)
        print("AUTOTUNING ENABLED - Neural Network Will Learn Plant Dynamics")
        print("🤖"*35)
        print(f"  Normalization: (T - {args.setpoint}) / 10.0")
        print(f"  First retune at step: {args.retune_start}")
        print(f"  Retune frequency: every {args.retune_every} steps")
        print(f"  IMC parameters: L={args.imc_L}, λ={args.imc_lambda}")
        print("🤖"*35 + "\n")

    temps = []
    powers = []
    losses = []

    consecutive_failures = 0
    step = 0

    start_t = time.monotonic()
    next_tick = start_t

    print("\n" + "="*70)
    print("STARTING ESP32 PID CONTROL LOOP")
    print("="*70)
    print(f"  ESP32 IP:        {args.esp_ip}")
    print(f"  Setpoint:        {args.setpoint:.2f}°C")
    print(f"  Control period:  {args.dt:.3f}s")
    print(f"  Initial PID:     Kp={args.kp:.2f}, Ki={args.ki:.2f}, Kd={args.kd:.2f}")
    print(f"  CSV log:         {args.csv}")
    print(f"  Safety cutoff:   {args.host_max_temp:.2f}°C")
    print("="*70 + "\n")

    with open(args.csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "timestamp",
                "step",
                "elapsed_s",
                "temp_c",
                "error_c",
                "heater_norm",
                "pwm_cmd",
                "esp_safety",
                "host_safety",
                "kp",
                "ki",
                "kd",
                "nn_loss",
            ]
        )

        try:
            while True:
                now = time.monotonic()
                if now < next_tick:
                    time.sleep(next_tick - now)

                loop_t = time.monotonic()
                elapsed_s = loop_t - start_t

                if args.steps > 0 and step >= args.steps:
                    print("\n" + "="*70)
                    print("STEP LIMIT REACHED - STOPPING")
                    print("="*70)
                    break
                if args.duration > 0 and elapsed_s >= args.duration:
                    print("\n" + "="*70)
                    print("DURATION LIMIT REACHED - STOPPING")
                    print("="*70)
                    break

                # ============================================================
                # READ TEMPERATURE FROM ESP32
                # ============================================================
                try:
                    temp_c, esp_safety = read_temp(args.esp_ip, args.request_timeout)
                    consecutive_failures = 0
                except (ValueError, urllib.error.URLError, TimeoutError) as exc:
                    consecutive_failures += 1
                    print(f"\n❌ Step {step:05d} | SENSOR READ FAILED ({consecutive_failures}/{args.max_failures})")
                    print(f"   Error: {exc}")

                    try:
                        write_pwm(args.esp_ip, 0, args.request_timeout)
                        print(f"   → Set PWM=0 for safety")
                    except Exception:
                        pass

                    if consecutive_failures >= args.max_failures:
                        print("\n" + "="*70)
                        print("TOO MANY COMMUNICATION FAILURES - STOPPING")
                        print("="*70)
                        break

                    next_tick += args.dt
                    step += 1
                    continue

                # ============================================================
                # COMPUTE PID CONTROL OUTPUT
                # ============================================================
                error_c = args.setpoint - temp_c
                heater_norm = pid.Calculate_heater(args.setpoint, temp_c)
                host_safety = temp_c >= args.host_max_temp

                # Safety override
                if esp_safety or host_safety:
                    heater_norm = 0.0
                    if step % 10 == 0:  # Don't spam this message
                        if esp_safety:
                            print(f"⚠️  Step {step:05d} | ESP SAFETY TRIGGERED - Heater disabled")
                        if host_safety:
                            print(f"⚠️  Step {step:05d} | HOST SAFETY TRIGGERED (T={temp_c:.2f}°C ≥ {args.host_max_temp:.2f}°C) - Heater disabled")

                heater_norm = max(0.0, min(1.0, heater_norm))
                pwm_cmd = int(round(heater_norm * MAX_PWM_CAP))

                # ============================================================
                # SEND PWM COMMAND TO ESP32
                # ============================================================
                try:
                    write_pwm(args.esp_ip, pwm_cmd, args.request_timeout)
                except (urllib.error.URLError, TimeoutError) as exc:
                    print(f"⚠️  Step {step:05d} | PWM write failed: {exc}")

                # ============================================================
                # STORE DATA FOR NEURAL NETWORK TRAINING
                # ============================================================
                temps.append(temp_c)
                powers.append(heater_norm)

                # ============================================================
                # NEURAL NETWORK TRAINING & AUTOTUNING
                # ============================================================
                loss_val = None
                if args.autotune and nn_model is not None:
                    # Add training sample (need at least 3 historical points)
                    if len(temps) >= 3:
                        nn_model.add_sample(
                            temps[-3],   # T[k-2]
                            temps[-2],   # T[k-1]
                            powers[-2],  # U[k-1]
                            powers[-3],  # U[k-2]
                            temps[-1],   # T[k] (target to predict)
                        )
                    
                    # Train neural network with mini-batching
                    loss_val = nn_model.train_step(batch_size=32, num_epochs=1)
                    losses.append(loss_val if loss_val is not None else float("nan"))
                    
                    # Print comprehensive diagnostics periodically
                    if step > 0 and step % 200 == 0:
                        nn_model.print_diagnostics()
                    
                    # ========================================================
                    # ATTEMPT PID RETUNING
                    # ========================================================
                    if step >= args.retune_start and step % args.retune_every == 0:
                        print("\n" + "🔧"*35)
                        print(f"[AUTOTUNER] RETUNING ATTEMPT AT STEP {step}")
                        print("🔧"*35 + "\n")
                        
                        try:
                            # STEP 1: Estimate discrete model parameters (a, b)
                            print(f"[AUTOTUNER] Step 1/3: Estimating discrete parameters...")
                            a, b = estimate_parameters(
                                nn_model, 
                                temp_c, 
                                heater_norm, 
                                args.dt,
                                verbose=True  # Detailed diagnostics
                            )
                            
                            if a is not None and b is not None:
                                # STEP 2: Convert to continuous-time parameters (τ, K)
                                print(f"[AUTOTUNER] Step 2/3: Converting to continuous parameters...")
                                tau, K = compute_tau_K(a, b, args.dt, verbose=True)
                                
                                if tau is not None and K is not None:
                                    # STEP 3: Compute IMC-based PID gains
                                    print(f"[AUTOTUNER] Step 3/3: Computing IMC-PID gains...")
                                    gains = imc_pid(
                                        K, tau, 
                                        L=args.imc_L, 
                                        lam=args.imc_lambda,
                                        verbose=True
                                    )
                                    
                                    if gains is not None:
                                        kp, ki, kd = gains
                                        
                                        # Strict validation before applying
                                        finite = all(math.isfinite(g) for g in (kp, ki, kd))
                                        reasonable = (
                                            0.1 < kp < 500 and 
                                            0 <= ki < 100 and 
                                            0 <= kd < 100
                                        )
                                        
                                        if finite and reasonable:
                                            print(f"\n{'='*70}")
                                            print(f"[AUTOTUNER] ✅ NEW GAINS ACCEPTED AND APPLIED")
                                            print(f"{'='*70}\n")
                                            pid.update_gains(kp, ki, kd)
                                        else:
                                            print(f"\n{'='*70}")
                                            print(f"[AUTOTUNER] ❌ GAINS REJECTED (outside safe ranges)")
                                            print(f"  Kp={kp:.6f} (expect: 0.1-500, finite={math.isfinite(kp)})")
                                            print(f"  Ki={ki:.6f} (expect: 0-100, finite={math.isfinite(ki)})")
                                            print(f"  Kd={kd:.6f} (expect: 0-100, finite={math.isfinite(kd)})")
                                            print(f"{'='*70}\n")
                                    else:
                                        print(f"\n[AUTOTUNER] ❌ IMC-PID computation returned None\n")
                                else:
                                    print(f"\n[AUTOTUNER] ❌ τ/K extraction failed (returned None)\n")
                            else:
                                print(f"\n[AUTOTUNER] ❌ Parameter estimation failed (model not ready or estimation failed)\n")
                        
                        except Exception as exc:
                            print(f"\n{'='*70}")
                            print(f"[AUTOTUNER] ❌ EXCEPTION DURING RETUNING:")
                            print(f"  Exception type: {type(exc).__name__}")
                            print(f"  Message: {exc}")
                            print(f"{'='*70}\n")
                            import traceback
                            traceback.print_exc()

                # ============================================================
                # LOG DATA TO CSV
                # ============================================================
                writer.writerow(
                    [
                        datetime.now().isoformat(timespec="seconds"),
                        step,
                        round(elapsed_s, 3),
                        round(temp_c, 4),
                        round(error_c, 4),
                        round(heater_norm, 5),
                        pwm_cmd,
                        int(bool(esp_safety)),
                        int(bool(host_safety)),
                        round(pid.Kp, 6),
                        round(pid.Ki, 6),
                        round(pid.Kd, 6),
                        "" if loss_val is None else round(float(loss_val), 8),
                    ]
                )
                csv_file.flush()

                # ============================================================
                # PRINT STEP SUMMARY
                # ============================================================
                # Color coding for temperature error
                if abs(error_c) < 0.5:
                    error_indicator = "✓"
                elif abs(error_c) < 1.0:
                    error_indicator = "~"
                else:
                    error_indicator = "!"
                
                # Color coding for PWM (detect bang-bang)
                if pwm_cmd == 0 or pwm_cmd == MAX_PWM_CAP:
                    pwm_indicator = "⚠️ "
                else:
                    pwm_indicator = ""
                
                print(
                    f"Step {step:05d} | "
                    f"Temp {temp_c:7.3f}°C | "
                    f"Err {error_c:+7.3f}°C {error_indicator} | "
                    f"Out {heater_norm:5.3f} | "
                    f"{pwm_indicator}PWM {pwm_cmd:3d} | "
                    f"ESP_SAFE={int(esp_safety)} HOST_SAFE={int(host_safety)}"
                    + (f" | Loss={loss_val:.6f}" if loss_val is not None else "")
                )

                # ============================================================
                # READ ESP32 STATUS (OPTIONAL DIAGNOSTICS)
                # ============================================================
                if args.status_every > 0 and step > 0 and step % args.status_every == 0:
                    try:
                        status = read_status(args.esp_ip, args.request_timeout)
                        print(f"📊 ESP32 Status: {status}")
                    except Exception as e:
                        print(f"⚠️  Could not read ESP32 status: {e}")

                next_tick += args.dt
                step += 1

        except KeyboardInterrupt:
            print("\n\n" + "="*70)
            print("INTERRUPTED BY USER (Ctrl+C)")
            print("="*70)
        finally:
            # ============================================================
            # SAFE SHUTDOWN - SET PWM TO 0
            # ============================================================
            print(f"\nShutting down safely...")
            try:
                write_pwm(args.esp_ip, 0, args.request_timeout)
                print(f"✅ Sent final PWM=0 to ESP32")
            except Exception as exc:
                print(f"❌ Could not send final PWM=0: {exc}")
            
            print(f"\n" + "="*70)
            print(f"RUN SUMMARY")
            print(f"="*70)
            print(f"  Total steps:     {step}")
            print(f"  Total duration:  {elapsed_s:.1f}s")
            print(f"  CSV log saved:   {args.csv}")
            if args.autotune and nn_model is not None:
                print(f"  NN samples:      {nn_model.total_samples_seen}")
                print(f"  Training steps:  {nn_model.training_steps}")
                if nn_model.train_losses:
                    final_loss = list(nn_model.train_losses)[-1]
                    print(f"  Final NN loss:   {final_loss:.6f}")
            print(f"  Final PID gains: Kp={pid.Kp:.6f}, Ki={pid.Ki:.6f}, Kd={pid.Kd:.6f}")
            print(f"="*70 + "\n")


if __name__ == "__main__":
    main()