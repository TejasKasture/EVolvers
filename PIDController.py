import math
from pythonfmu import Fmi2Slave, Fmi2Causality, Fmi2Variability, Real


class SupervisoryController2(Fmi2Slave):
    author = "Team EVolvers - Enhanced SOC Management"
    description = "Energy-optimized FMU with advanced SOC management for 60-70 km/h operation"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Inputs from CarMaker
        self.act_vel = 0.0
        self.brk_pedal = 0.0
        self.distance_covered = 0.0
        self.slope = 0.0
        self.soc = 1.0
        self.rolling_resistance = 0.008
        self.air_density = 1.225

        # Outputs
        self.motor_trq_target = 0.0

        # Constants
        self.batt_cap = 60.0  # kWh
        self.target_distance = 200000.0  # meters

        self.motor_temp = 30.0  # Celsius
        self.max_motor_temp = 90.0

        self.vehicle_mass = 1500.0
        self.wheel_radius = 0.3
        self.motor_efficiency = 0.88
        self.aero_drag_coeff = 0.28
        self.frontal_area = 2.3
        self.prev_torque = 0.0

        # Enhanced SOC management parameters
        self.energy_used = 0.0
        self.avg_kwh_per_km = 0.15  # Initial estimate
        self.soc_history = []
        self.energy_buffer_factor = 1.25  # 25% safety margin

        # SOC management thresholds
        self.critical_soc = 0.08
        self.low_soc = 0.15
        self.conservative_soc = 0.25
        self.comfort_soc = 0.40

        # PID gains - tuned for energy efficiency
        self.kp = 50.0  # Reduced for smoother control
        self.ki = 6.0  # Lower integral gain
        self.kd = 4.0  # Moderate derivative gain
        self.vel_err_int = 0.0
        self.vel_err_prev = 0.0

        # Speed management for SOC optimization
        self.speed_limits = {
            'max_highway': 19.44,  # 70 km/h
            'min_highway': 16.67,  # 60 km/h
            'eco_optimal': 17.78,  # 64 km/h - most efficient
            'emergency': 13.89  # 50 km/h - emergency SOC mode
        }

        # Torque limits
        self.max_brake_trq = -180.0
        self.max_regen_trq = -120.0
        self.base_max_torque = 100.0

        # SOC prediction and management
        self.soc_prediction_horizon = 10000.0  # meters ahead
        self.energy_consumption_window = []
        self.max_window_size = 100

        self.register_variables()

    def register_variables(self):
        # Inputs
        for name in ["act_vel", "brk_pedal", "distance_covered", "slope", "soc",
                     "rolling_resistance", "air_density"]:
            self.register_variable(
                Real(name, causality=Fmi2Causality.input, variability=Fmi2Variability.continuous)
            )

        # Output
        self.register_variable(
            Real("motor_trq_target", causality=Fmi2Causality.output, variability=Fmi2Variability.continuous)
        )

    def do_step(self, current_time: float, step_size: float) -> bool:
        # Update energy consumption tracking
        self.update_energy_tracking(step_size)

        # Enhanced SOC-based target velocity calculation
        target_vel = self.calc_soc_optimized_velocity()

        # Predict future SOC needs
        self.update_soc_prediction()

        # Calculate control action with SOC awareness
        accel = self.soc_aware_pid_control(target_vel, step_size)

        # Calculate torque with enhanced SOC management
        torque = self.calc_soc_optimized_torque(accel)

        # Enhanced regenerative braking with SOC prioritization
        if self.brk_pedal > 0.05:
            torque = self.calc_regenerative_braking(torque)

        # Passive regenerative braking on downhill when not braking
        if self.brk_pedal <= 0.05 and self.slope < -2.5 and self.act_vel > 5.0:
            torque = max(torque, -50.0)  # Light passive regen

        # Emergency energy-saving mode when SOC is nearly depleted
        if self.soc <= self.critical_soc + 0.01 and self.act_vel > 10.0:
            torque = min(torque, 0.0)  # Force coast or zero torque

        # Apply SOC-based torque limiting
        torque = self.apply_soc_torque_limits(torque)

        # Thermal management
        self.update_thermal_model(torque)

        # Final safety clamps
        torque = max(min(torque, self.get_dynamic_max_torque()), self.max_brake_trq)

        self.motor_trq_target = torque
        self.prev_torque = torque

        return True

    def update_energy_tracking(self, step_size):
        """Enhanced energy consumption tracking for better SOC prediction"""
        if self.act_vel > 0.1:
            # Calculate instantaneous power consumption
            power_kW = abs(self.prev_torque * self.act_vel / self.wheel_radius) / 1000.0
            energy_consumed = power_kW * step_size / 3600.0  # kWh

            self.energy_used += energy_consumed

            # Update consumption window for moving average
            consumption_rate = energy_consumed / max(step_size / 3600.0, 0.001)  # kW
            self.energy_consumption_window.append(consumption_rate)

            if len(self.energy_consumption_window) > self.max_window_size:
                self.energy_consumption_window.pop(0)

            # Update average consumption per km
            km_covered = self.distance_covered / 1000.0
            if km_covered > 0.1:
                self.avg_kwh_per_km = self.energy_used / km_covered

                # Smooth the average to avoid rapid changes
                if hasattr(self, 'prev_avg_kwh_per_km'):
                    self.avg_kwh_per_km = 0.8 * self.prev_avg_kwh_per_km + 0.2 * self.avg_kwh_per_km
                self.prev_avg_kwh_per_km = self.avg_kwh_per_km

    def update_soc_prediction(self):
        """Predict future SOC based on current consumption patterns"""
        remaining_distance_km = (self.target_distance - self.distance_covered) / 1000.0

        if remaining_distance_km > 0 and self.avg_kwh_per_km > 0:
            # Predict energy needed for remaining distance
            predicted_energy_need = remaining_distance_km * self.avg_kwh_per_km * self.energy_buffer_factor

            # Calculate if current SOC is sufficient
            available_energy = self.soc * self.batt_cap
            self.energy_deficit = max(0.0, predicted_energy_need - available_energy)
            self.energy_surplus = max(0.0, available_energy - predicted_energy_need)

            # Store SOC history for trend analysis
            self.soc_history.append(self.soc)
            if len(self.soc_history) > 50:
                self.soc_history.pop(0)

    def calc_soc_optimized_velocity(self):
        """Calculate target velocity optimized for SOC management in 60-70 km/h range"""
        remaining_distance_km = (self.target_distance - self.distance_covered) / 1000.0
        available_energy = self.soc * self.batt_cap
        predicted_energy_need = remaining_distance_km * self.avg_kwh_per_km * self.energy_buffer_factor

        # Base speed selection based on SOC status
        if self.soc <= self.critical_soc:
            # Critical SOC: minimum speed for maximum efficiency
            base_speed = self.speed_limits['emergency']
        elif self.soc <= self.low_soc:
            # Low SOC: conservative speed
            base_speed = self.speed_limits['min_highway']
        elif self.soc <= self.conservative_soc:
            # Conservative SOC: slightly below optimal
            base_speed = (self.speed_limits['min_highway'] + self.speed_limits['eco_optimal']) / 2
        elif available_energy < predicted_energy_need:
            # Energy deficit predicted: use eco-optimal speed
            base_speed = self.speed_limits['eco_optimal']
        else:
            # Sufficient energy: can use higher speeds
            energy_ratio = available_energy / predicted_energy_need
            if energy_ratio > 1.5:
                base_speed = self.speed_limits['max_highway']
            elif energy_ratio > 1.2:
                base_speed = (self.speed_limits['eco_optimal'] + self.speed_limits['max_highway']) / 2
            else:
                base_speed = self.speed_limits['eco_optimal']

        # Slope-based speed adjustment
        slope_factor = self.calc_slope_speed_factor()

        # Distance-based adjustment for final approach
        distance_factor = self.calc_distance_speed_factor(remaining_distance_km)

        # Environmental efficiency factor
        efficiency_factor = self.calc_efficiency_factor()

        # Calculate final target velocity
        target_velocity = base_speed * slope_factor * distance_factor * efficiency_factor

        # Ensure velocity stays within operational bounds
        target_velocity = max(self.speed_limits['min_highway'],
                              min(target_velocity, self.speed_limits['max_highway']))

        # Smooth velocity transitions to avoid energy waste
        if hasattr(self, 'prev_target_vel'):
            max_change = 0.5 if self.soc < self.low_soc else 1.0  # Smoother at low SOC
            velocity_change = target_velocity - self.prev_target_vel
            target_velocity = self.prev_target_vel + max(-max_change, min(velocity_change, max_change))

        self.prev_target_vel = target_velocity
        return target_velocity

    def calc_slope_speed_factor(self):
        """Calculate speed adjustment factor based on slope for energy optimization"""
        if self.slope > 6.0:
            return max(0.85, 1.0 - (self.slope - 6.0) * 0.03)
        elif self.slope > 3.0:
            return max(0.90, 1.0 - (self.slope - 3.0) * 0.02)
        elif self.slope > 1.0:
            return max(0.95, 1.0 - (self.slope - 1.0) * 0.015)
        elif self.slope < -4.0:
            # Downhill: can afford slightly higher speeds
            return min(1.05, 1.0 + abs(self.slope + 4.0) * 0.01)
        elif self.slope < -2.0:
            return min(1.02, 1.0 + abs(self.slope + 2.0) * 0.005)
        else:
            return 1.0

    def calc_distance_speed_factor(self, remaining_km):
        """Adjust speed based on remaining distance"""
        if remaining_km < 1.0:
            return 0.80  # Slow down for final approach
        elif remaining_km < 3.0:
            return 0.90  # Moderate reduction
        elif remaining_km < 8.0:
            return 0.95  # Slight reduction
        else:
            return 1.0

    def calc_efficiency_factor(self):
        """Calculate efficiency factor for speed optimization"""
        # Air density efficiency
        air_factor = max(0.96, min(1.02, 1.225 / self.air_density))

        # Temperature efficiency (motor efficiency varies with temperature)
        temp_factor = max(0.95, min(1.0, 1.0 - abs(self.motor_temp - 60.0) / 100.0))

        # SOC efficiency (battery efficiency decreases at very low SOC)
        soc_factor = max(0.92, min(1.0, self.soc / 0.3)) if self.soc < 0.3 else 1.0

        return air_factor * temp_factor * soc_factor

    def soc_aware_pid_control(self, target_vel, dt):
        """PID controller with SOC-aware tuning"""
        error = target_vel - self.act_vel

        # SOC-based PID gain adjustment
        if self.soc <= self.critical_soc:
            gain_factor = 0.5  # Very conservative
        elif self.soc <= self.low_soc:
            gain_factor = 0.7  # Conservative
        elif self.soc <= self.conservative_soc:
            gain_factor = 0.85  # Moderate
        else:
            gain_factor = 1.0  # Normal operation

        # Slope compensation for PID gains
        slope_factor = 1.1 if abs(self.slope) > 3.0 else 1.0

        # Final gain calculation
        total_factor = gain_factor * slope_factor
        kp = self.kp * total_factor
        ki = self.ki * total_factor * 0.8  # Reduce integral for energy efficiency
        kd = self.kd * total_factor

        # Enhanced integral windup protection
        integral_limit = 1.0 if self.soc < self.low_soc else 1.5
        self.vel_err_int += error * dt
        self.vel_err_int = max(min(self.vel_err_int, integral_limit), -integral_limit)

        # Integral decay for energy efficiency
        if abs(error) < 0.3:
            self.vel_err_int *= 0.98  # Slowly decay to prevent energy waste

        # PID calculation
        p_term = kp * error
        i_term = ki * self.vel_err_int
        d_term = kd * (error - self.vel_err_prev) / dt if dt > 0 else 0.0

        self.vel_err_prev = error

        # Convert to acceleration with slope compensation
        slope_compensation = math.sin(math.radians(self.slope)) * 0.2
        accel = (p_term + i_term + d_term) / 400.0 + slope_compensation

        # SOC-based acceleration limits
        if self.soc <= self.critical_soc:
            max_accel, min_accel = 0.8, -1.5
        elif self.soc <= self.low_soc:
            max_accel, min_accel = 1.2, -2.0
        elif self.soc <= self.conservative_soc:
            max_accel, min_accel = 1.5, -2.5
        else:
            max_accel, min_accel = 2.0, -3.0

        return max(min(accel, max_accel), min_accel)

    def calc_soc_optimized_torque(self, accel):
        """Calculate torque with enhanced SOC optimization"""
        g = 9.81

        # Enhanced physics model
        slope_rad = math.radians(self.slope)
        slope_force = self.vehicle_mass * g * math.sin(slope_rad)
        normal_force = self.vehicle_mass * g * math.cos(slope_rad)

        # Dynamic rolling resistance (increases with speed and load)
        speed_factor = 1.0 + (self.act_vel / 40.0) * 0.15
        rolling_force = normal_force * self.rolling_resistance * speed_factor

        # Aerodynamic drag with air density consideration
        drag_force = 0.5 * self.air_density * self.frontal_area * self.aero_drag_coeff * (self.act_vel ** 2)

        # Inertial force with rotational inertia
        effective_mass = self.vehicle_mass * 1.06
        inertial_force = effective_mass * accel

        total_force = slope_force + rolling_force + drag_force + inertial_force

        # SOC-dependent efficiency model
        if self.soc <= self.critical_soc:
            efficiency = self.motor_efficiency * 0.85  # Reduced efficiency at critical SOC
        elif self.soc <= self.low_soc:
            efficiency = self.motor_efficiency * 0.90
        elif self.soc <= self.conservative_soc:
            efficiency = self.motor_efficiency * 0.95
        else:
            efficiency = self.motor_efficiency

        # Speed-dependent efficiency
        optimal_speed = 17.0  # m/s for maximum efficiency
        speed_efficiency = max(0.88, 1.0 - abs(self.act_vel - optimal_speed) / 25.0)
        efficiency *= speed_efficiency

        # Calculate base torque
        torque = (total_force * self.wheel_radius) / efficiency

        return torque

    def calc_regenerative_braking(self, base_torque):
        """Boosted regenerative braking for better energy recovery"""
        if self.act_vel < 1.0:
            return 0.0  # No regen when stopped

        # Dynamic regen factor based on SOC
        if self.soc < 0.20:
            regen_factor = 2.0  # Aggressive regen at low SOC
        elif self.soc < 0.50:
            regen_factor = 1.5
        elif self.soc < 0.80:
            regen_factor = 1.2
        else:
            regen_factor = 1.0  # Normal regen

        # Braking intensity
        brake_intensity = min(self.brk_pedal, 1.0)
        base_regen = -brake_intensity * 200.0 * regen_factor  # Increased from 150

        # Thermal limiting (to avoid overheating motor)
        if self.motor_temp < 85.0:
            thermal_factor = 1.0
        else:
            thermal_factor = 0.8

        max_regen_torque = self.max_regen_trq * thermal_factor

        # Clamp regen torque to max allowed value
        regen_torque = max(base_regen, max_regen_torque)

        # Apply regen only above threshold brake intensity
        if brake_intensity > 0.05:
            return regen_torque
        else:
            return base_torque

    def apply_soc_torque_limits(self, torque):
        """Apply SOC-based torque limiting with slope climbing priority"""
        max_torque = self.get_dynamic_max_torque()

        # Additional energy conservation limits
        remaining_distance_km = (self.target_distance - self.distance_covered) / 1000.0
        available_energy = self.soc * self.batt_cap

        if hasattr(self, 'energy_deficit') and self.energy_deficit > 0:
            # Energy deficit detected - apply stricter limits, but less restrictive on slopes
            if self.slope > 3.0:
                deficit_factor = max(0.80,
                                     1.0 - self.energy_deficit / (self.batt_cap * 0.2))  # Less restrictive on slopes
            else:
                deficit_factor = max(0.6, 1.0 - self.energy_deficit / (self.batt_cap * 0.1))
            max_torque *= deficit_factor

        # Progressive torque limiting as SOC decreases - more generous for slopes
        if self.soc <= self.critical_soc:
            if self.slope > 3.0:
                max_torque = min(max_torque, 50.0)  # Allow more torque for steep slopes
            else:
                max_torque = min(max_torque, 30.0)
        elif self.soc <= self.low_soc:
            if self.slope > 3.0:
                max_torque = min(max_torque, 80.0)  # Allow more torque for steep slopes
            else:
                max_torque = min(max_torque, 50.0)
        elif self.soc <= self.conservative_soc:
            if self.slope > 3.0:
                max_torque = min(max_torque, 110.0)  # Allow more torque for steep slopes
            else:
                max_torque = min(max_torque, 70.0)

        # Smooth torque transitions to avoid energy spikes
        if hasattr(self, 'prev_torque') and self.prev_torque is not None:
            # Allow faster torque changes on slopes for responsiveness
            if self.slope > 2.0:
                max_change = 20.0 if self.soc < self.low_soc else 25.0
            else:
                max_change = 12.0 if self.soc < self.low_soc else 18.0
            torque_change = torque - self.prev_torque
            torque = self.prev_torque + max(-max_change, min(torque_change, max_change))

        return max(min(torque, max_torque), self.max_brake_trq)

    def get_dynamic_max_torque(self):
        """Calculate dynamic maximum torque based on current conditions with enhanced slope climbing"""
        # Enhanced base torque from slope requirements - prioritize slope climbing
        if self.slope > 6.0:
            base_torque = 160.0  # Very steep slopes need high torque
        elif self.slope > 4.0:
            base_torque = 140.0  # Steep slopes
        elif self.slope > 3.0:
            base_torque = 120.0  # Moderate steep slopes
        elif self.slope > 1.0:
            base_torque = 100.0  # Gentle slopes
        elif self.slope > 0.0:
            base_torque = 80.0  # Slight inclines
        else:
            base_torque = 70.0  # Flat or downhill

        # More generous SOC-based scaling for slope climbing
        if self.soc <= self.critical_soc:
            # Even at critical SOC, allow more torque for slopes > 3%
            if self.slope > 3.0:
                soc_factor = 0.60  # Allow more torque for steep slopes
            else:
                soc_factor = 0.35
        elif self.soc <= self.low_soc:
            if self.slope > 3.0:
                soc_factor = 0.75  # Generous torque for steep slopes
            else:
                soc_factor = 0.55
        elif self.soc <= self.conservative_soc:
            if self.slope > 3.0:
                soc_factor = 0.90  # Near full torque for steep slopes
            else:
                soc_factor = 0.75
        elif self.soc <= self.comfort_soc:
            soc_factor = 0.95  # Almost full torque
        else:
            soc_factor = 1.0  # Full torque available

        # Adjust speed factor for slope climbing (allow more torque at low speeds on slopes)
        if self.slope > 2.0:
            # On slopes, prioritize climbing ability over speed efficiency
            if self.act_vel < 10.0:
                speed_factor = 1.1  # Boost torque at low speeds on slopes
            elif self.act_vel < 15.0:
                speed_factor = 1.05
            else:
                speed_factor = 1.0
        else:
            # Normal speed-based adjustment for flat terrain
            if self.act_vel > 18.0:
                speed_factor = 0.90
            elif self.act_vel > 15.0:
                speed_factor = 0.95
            else:
                speed_factor = 1.0

        # Relaxed thermal protection for slope climbing
        if self.motor_temp > 90.0:
            thermal_factor = 0.75  # Only reduce if very hot
        elif self.motor_temp > 80.0:
            thermal_factor = 0.90  # Slight reduction
        else:
            thermal_factor = 1.0

        max_torque = base_torque * soc_factor * speed_factor * thermal_factor

        # Enhanced minimum torque for slope climbing
        if self.slope > 3.0:
            min_torque = 60.0 if self.soc > 0.05 else 40.0  # Higher minimum for slopes
        elif self.slope > 1.0:
            min_torque = 40.0 if self.soc > 0.05 else 25.0
        else:
            min_torque = 25.0 if self.soc > 0.05 else 15.0

        return max(min_torque, max_torque)

    def update_thermal_model(self, torque):
        """Simple thermal model for motor temperature"""
        # Heat generation from torque
        heat_generation = 0.008 * abs(torque) + 0.002 * (torque ** 2) / 100.0

        # Cooling (natural + forced)
        cooling_rate = 0.03 + 0.001 * self.act_vel
        ambient_temp = 25.0
        cooling = cooling_rate * (self.motor_temp - ambient_temp)

        # Update temperature
        self.motor_temp += heat_generation - cooling
        self.motor_temp = max(ambient_temp, min(self.motor_temp, 120.0))  # Clamp temperature