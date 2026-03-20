class PID:
    def __init__(self,Kp,Ki,Kd,dt):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.integral = 0
        self.prev_error = 0
    def Calculate_heater(self,setpoint,measurement):
        error = setpoint - measurement
        self.integral += error*self.dt
        derivative = (error-self.prev_error)/self.dt
        out = (error*self.Kp)+(self.Ki*self.integral)+(self.Kd*derivative)
        self.prev_error = error
        print("calculated heater")
        print(max(0,min(1,out)))
        return max(0,min(1,out))
    def update_gains(self, kp, ki, kd):
        print(f"\nUpdated PID: Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f}\n")
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd