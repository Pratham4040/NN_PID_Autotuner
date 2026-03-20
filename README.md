# NN_PID_Autotuner

Neural-network-based PID autotuner for a thermal chamber.

## Real Hardware Run (ESP32)

The ESP32 should expose these endpoints:

- GET /temp -> current temperature as plain text
- POST /pwm -> accepts 0..255 PWM as plain text body
- GET /status -> JSON status

Use this script to run control from PC:

- esp32_pid_runner.py

### Run command

```bash
python esp32_pid_runner.py --esp-ip 192.168.137.42 --setpoint 37 --dt 0.7
```

### Useful options

```bash
# run for 20 minutes
python esp32_pid_runner.py --esp-ip 192.168.137.42 --duration 1200

# fixed number of control steps
python esp32_pid_runner.py --esp-ip 192.168.137.42 --steps 1000

# custom starting PID gains
python esp32_pid_runner.py --esp-ip 192.168.137.42 --kp 20 --ki 0.5 --kd 5

# enable online NN autotuning
python esp32_pid_runner.py --esp-ip 192.168.137.42 --autotune
```

### Safety and logging

- ESP32 safety is still active on-board (MAX_TEMP in firmware).
- PC runner also has host cutoff via --host-max-temp (default 38 C).
- On stop or Ctrl+C, the script sends PWM=0.
- CSV log is written to esp32_run_log.csv by default.
