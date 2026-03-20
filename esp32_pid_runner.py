import argparse
import csv
import json
import math
import time
import urllib.error
import urllib.request
from datetime import datetime

from pid import PID


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
    pwm = int(max(0, min(255, pwm)))
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
    parser.add_argument("--setpoint", type=float, default=37.0, help="Target chamber temperature in C")
    parser.add_argument("--dt", type=float, default=0.7, help="Control period in seconds")

    parser.add_argument("--kp", type=float, default=69.69, help="Initial Kp")
    parser.add_argument("--ki", type=float, default=68.69, help="Initial Ki")
    parser.add_argument("--kd", type=float, default=69.0, help="Initial Kd")

    parser.add_argument("--steps", type=int, default=0, help="Number of control steps, 0 means run forever")
    parser.add_argument("--duration", type=float, default=0.0, help="Run duration in seconds, 0 means no limit")

    parser.add_argument("--host-max-temp", type=float, default=38.0, help="PC-side safety cutoff in C")
    parser.add_argument("--request-timeout", type=float, default=1.5, help="HTTP timeout in seconds")
    parser.add_argument("--max-failures", type=int, default=8, help="Consecutive HTTP failures before stop")

    parser.add_argument("--csv", default="esp32_run_log.csv", help="Path for CSV run log")
    parser.add_argument("--status-every", type=int, default=20, help="Print /status every N steps, 0 disables")

    parser.add_argument("--autotune", action="store_true", help="Enable online NN-based autotuning")
    parser.add_argument("--retune-every", type=int, default=50, help="Retune period in steps")
    parser.add_argument("--retune-start", type=int, default=100, help="Start retuning after this step")
    parser.add_argument("--imc-L", type=float, default=0.7, help="IMC dead-time estimate")
    parser.add_argument("--imc-lambda", type=float, default=3.0, help="IMC lambda")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.dt <= 0:
        raise ValueError("--dt must be > 0")

    pid = PID(args.kp, args.ki, args.kd, args.dt)
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
        nn_model = NeuralPlantModel()

    temps = []
    powers = []
    losses = []

    consecutive_failures = 0
    step = 0

    start_t = time.monotonic()
    next_tick = start_t

    print("Starting ESP32 PID loop")
    print(f"ESP32: {args.esp_ip} | setpoint={args.setpoint:.2f}C | dt={args.dt:.3f}s")
    print(f"Logging CSV to: {args.csv}")

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
                    print("Step limit reached, stopping.")
                    break
                if args.duration > 0 and elapsed_s >= args.duration:
                    print("Duration limit reached, stopping.")
                    break

                try:
                    temp_c, esp_safety = read_temp(args.esp_ip, args.request_timeout)
                    consecutive_failures = 0
                except (ValueError, urllib.error.URLError, TimeoutError) as exc:
                    consecutive_failures += 1
                    print(f"Step {step:05d} | sensor read failed ({consecutive_failures}/{args.max_failures}): {exc}")

                    try:
                        write_pwm(args.esp_ip, 0, args.request_timeout)
                    except Exception:
                        pass

                    if consecutive_failures >= args.max_failures:
                        print("Too many communication failures. Stopping run.")
                        break

                    next_tick += args.dt
                    step += 1
                    continue

                error_c = args.setpoint - temp_c
                heater_norm = pid.Calculate_heater(args.setpoint, temp_c)
                host_safety = temp_c >= args.host_max_temp

                if esp_safety or host_safety:
                    heater_norm = 0.0

                heater_norm = max(0.0, min(1.0, heater_norm))
                pwm_cmd = int(round(heater_norm * 255.0))

                try:
                    write_pwm(args.esp_ip, pwm_cmd, args.request_timeout)
                except (urllib.error.URLError, TimeoutError) as exc:
                    print(f"Step {step:05d} | pwm write failed: {exc}")

                temps.append(temp_c)
                powers.append(heater_norm)

                loss_val = None
                if args.autotune and nn_model is not None:
                    if len(temps) >= 3:
                        nn_model.add_sample(
                            temps[-3],
                            temps[-2],
                            powers[-2],
                            powers[-3],
                            temps[-1],
                        )

                    loss_val = nn_model.train_step()
                    losses.append(loss_val if loss_val is not None else float("nan"))

                    if step % args.retune_every == 0 and step > args.retune_start:
                        try:
                            a, b = estimate_parameters(nn_model, temp_c, heater_norm)
                            tau, K = compute_tau_K(a, b, args.dt)
                            gains = imc_pid(K, tau, L=args.imc_L, lam=args.imc_lambda) if tau and K else None
                        except Exception as exc:
                            gains = None
                            print(f"Step {step:05d} | autotune failed: {exc}")

                        if gains:
                            kp, ki, kd = gains
                            finite = all(math.isfinite(g) for g in (kp, ki, kd))
                            non_negative = kp > 0 and ki >= 0 and kd >= 0
                            if finite and non_negative:
                                pid.update_gains(kp, ki, kd)
                            else:
                                print(
                                    f"Step {step:05d} | skipped unsafe gains: "
                                    f"Kp={kp}, Ki={ki}, Kd={kd}"
                                )

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

                print(
                    f"Step {step:05d} | Temp {temp_c:7.3f} C | Err {error_c:7.3f} | "
                    f"Out {heater_norm:0.3f} | PWM {pwm_cmd:3d} | "
                    f"ESP_SAFE={int(esp_safety)} HOST_SAFE={int(host_safety)}"
                )

                if args.status_every > 0 and step > 0 and step % args.status_every == 0:
                    try:
                        status = read_status(args.esp_ip, args.request_timeout)
                        print(f"Status: {status}")
                    except Exception:
                        pass

                next_tick += args.dt
                step += 1

        except KeyboardInterrupt:
            print("Interrupted by user, stopping loop.")
        finally:
            try:
                write_pwm(args.esp_ip, 0, args.request_timeout)
                print("Sent final PWM=0 for safe stop.")
            except Exception as exc:
                print(f"Could not send final PWM=0: {exc}")


if __name__ == "__main__":
    main()