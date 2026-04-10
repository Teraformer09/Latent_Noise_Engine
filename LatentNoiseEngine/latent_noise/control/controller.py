class AdaptiveController:
    """
    PID Controller for adaptive noise suppression.
    - Proportional (kp): immediate response to current error
    - Integral (ki): eliminates steady-state error via accumulated correction
    - Derivative (kd): damps oscillations by reacting to rate of change
    Anti-windup clamping is applied to the integral term.
    """
    def __init__(self, base_alpha=1.0, kp=10.0, ki=5.0, kd=0.0, target=0.1, dt=0.1):
        self.base_alpha = base_alpha
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target = target
        self.dt = dt

        self.integral = 0.0
        self.prev_error = 0.0
        self.alpha = base_alpha

    def update(self, hazard):
        error = hazard - self.target

        # Anti-windup clamping for integral term
        self.integral = max(-5.0, min(self.integral + error * self.dt, 5.0))

        # Derivative term (finite difference)
        derivative = (error - self.prev_error) / (self.dt + 1e-12)
        self.prev_error = error

        # PID control signal
        control_signal = (
            self.base_alpha
            + self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )

        # Low-pass filter for stability
        self.alpha = 0.8 * self.alpha + 0.2 * control_signal

        # Clamping
        self.alpha = max(0.01, min(self.alpha, 50.0))
        return self.alpha
