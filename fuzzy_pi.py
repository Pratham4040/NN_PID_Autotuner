class FuzzyPIController:
    """
    Sugeno-style fuzzy PI gain scheduler.

    Inputs:
      e  = setpoint - predicted_temp (deg C)
      ec = (e - prev_e) / dt (deg C/s)

    Outputs:
      kp, ki plus incremental terms dkp, dki
    """

    def __init__(
        self,
        base_kp,
        base_ki,
        dt,
        e_small=0.3,
        e_large=2.0,
        ec_small=0.02,
        ec_large=0.15,
        dkp_boost=0.6,
        dkp_brake=0.4,
        dki_boost=0.02,
        kp_min=0.3,
        ki_min=0.3,
        kp_max=500.0,
        ki_max=100.0,
    ):
        self.base_kp = max(float(base_kp), kp_min)
        self.base_ki = max(float(base_ki), ki_min)
        self.dt = float(dt)

        self.e_small = float(e_small)
        self.e_large = float(e_large)
        self.ec_small = float(ec_small)
        self.ec_large = float(ec_large)

        self.dkp_boost = float(dkp_boost)
        self.dkp_brake = float(dkp_brake)
        self.dki_boost = float(dki_boost)

        self.kp_min = float(kp_min)
        self.ki_min = float(ki_min)
        self.kp_max = float(kp_max)
        self.ki_max = float(ki_max)

    def _mf_small(self, x, small, large):
        ax = abs(x)
        if ax <= small:
            return 1.0
        if ax >= large:
            return 0.0
        return (large - ax) / (large - small)

    def _mf_large(self, x, small, large):
        ax = abs(x)
        if ax <= small:
            return 0.0
        if ax >= large:
            return 1.0
        return (ax - small) / (large - small)

    def compute_gains(self, e, ec):
        # Memberships for error magnitude (small/large) and rate (small/fast-neg)
        mu_e_small = self._mf_small(e, self.e_small, self.e_large)
        mu_e_pos_large = self._mf_large(max(e, 0.0), self.e_small, self.e_large)
        mu_ec_small = self._mf_small(ec, self.ec_small, self.ec_large)
        mu_ec_fast_neg = self._mf_large(max(-ec, 0.0), self.ec_small, self.ec_large)

        # Rule weights
        w_heat = mu_e_pos_large
        w_brake = mu_e_small * mu_ec_fast_neg
        w_steady = mu_e_small * mu_ec_small

        w_sum = w_heat + w_brake + w_steady
        if w_sum > 0.0:
            dkp = (w_heat * self.dkp_boost + w_brake * (-self.dkp_brake)) / w_sum
            dki = (w_steady * self.dki_boost) / w_sum
        else:
            dkp = 0.0
            dki = 0.0

        kp = self.base_kp + dkp
        ki = self.base_ki + dki

        kp = max(self.kp_min, min(self.kp_max, kp))
        ki = max(self.ki_min, min(self.ki_max, ki))

        return kp, ki, dkp, dki
