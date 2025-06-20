import math
from pythonfmu import Fmi2Slave, Fmi2Causality, Fmi2Variability, Real


class SupervisoryController2(Fmi2Slave):
    """
    EV Challenge FMU for Supervisory Control - Energy Optimized Version
    Generates Target Motor Torque based on optimal velocity while monitoring SoC,
    adapting to front vehicle behavior, and triggering overtaking with lane changes.

    KEY IMPROVEMENT: Advanced energy management to sustain battery until 150km
    """
    author = "Team EVolvers - Energy Optimized"
    description = "Supervisory controller with advanced energy management for 150km range"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Inputs (UNCHANGED as requested)
        self.act_vel = 0.0
        self.batt_current = 0.0
        self.brk_pedal = 0.0
        self.distance_covered = 0.0
        self.slope = 0.0
        self.soc = 1.0  # 100% by default
        self.front_vehicle_distance = 100.0  # m
        self.front_vehicle_velocity = 0.0  # m/s

        # Parameters (UNCHANGED as requested)
        self.batt_cap = 60.0  # kWh
        self.init_soc = 1.0
        self.max_current_threshold = 250.0  # Amps
        self.total_distance = 200000.0  # meters

        # Outputs (UNCHANGED as requested)
        self.motor_trq_target = 0.0

        # Internal state - ENHANCED for energy management
        self.vehicle_mass = 1500  # kg
        self.wheel_radius = 0.3  # m
        self.motor_efficiency = 0.9
        self.previous_distance = 0.0
        self.energy_used = 0.0  # kWh
        self.overtaking = False

        # === ENERGY MANAGEMENT ADDITIONS ===
        self.target_distance = 150000.0  # Target 150km range
        self.energy_consumption_history = []
        self.distance_history = []
        self.average_consumption_rate = 0.4  # kWh/km initial estimate
        self.energy_budget_per_km = 0.4  # Initial budget
        self.critical_soc_threshold = 0.1  # 10% critical
        self.conservative_soc_threshold = 0.25  # 25% conservative
        self.power_limit_factor = 1.0  # Dynamic power limiting

        # PID controller for smooth velocity control
        self.velocity_error_integral = 0.0
        self.velocity_error_previous = 0.0
        self.kp = 100.0  # Proportional gain
        self.ki = 15.0  # Integral gain
        self.kd = 8.0  # Derivative gain

        # Rolling resistance and aerodynamic parameters
        # self.rolling_resistance = 0.01
        self.rolling_resistance = 0.007  # Previously 0.01
        self.aero_drag_coeff = 0.28
        self.frontal_area = 2.3  # m²
        self.air_density = 1.225  # kg/m³

        self.register_variables()

    def register_variables(self):
        # Inputs (UNCHANGED)
        self.register_variable(Real("act_vel", causality=Fmi2Causality.input, variability=Fmi2Variability.continuous))
        self.register_variable(
            Real("batt_current", causality=Fmi2Causality.input, variability=Fmi2Variability.continuous))
        self.register_variable(Real("brk_pedal", causality=Fmi2Causality.input, variability=Fmi2Variability.continuous))
        self.register_variable(
            Real("distance_covered", causality=Fmi2Causality.input, variability=Fmi2Variability.continuous))
        self.register_variable(Real("slope", causality=Fmi2Causality.input, variability=Fmi2Variability.continuous))
        self.register_variable(Real("soc", causality=Fmi2Causality.input, variability=Fmi2Variability.continuous))
        self.register_variable(
            Real("front_vehicle_distance", causality=Fmi2Causality.input, variability=Fmi2Variability.continuous))
        self.register_variable(
            Real("front_vehicle_velocity", causality=Fmi2Causality.input, variability=Fmi2Variability.continuous))

        # Parameters (UNCHANGED)
        self.register_variable(Real("batt_cap", causality=Fmi2Causality.parameter, variability=Fmi2Variability.fixed))
        self.register_variable(Real("init_soc", causality=Fmi2Causality.parameter, variability=Fmi2Variability.fixed))
        self.register_variable(
            Real("max_current_threshold", causality=Fmi2Causality.parameter, variability=Fmi2Variability.fixed))
        self.register_variable(
            Real("total_distance", causality=Fmi2Causality.parameter, variability=Fmi2Variability.fixed))

        # Output (UNCHANGED)
        self.register_variable(
            Real("motor_trq_target", causality=Fmi2Causality.output, variability=Fmi2Variability.continuous))

    def do_step(self, current_time: float, step_size: float) -> bool:
        # === ENERGY TRACKING AND PREDICTION ===
        self.update_energy_consumption(step_size)
        energy_state = self.analyze_energy_state()

        # === INTELLIGENT VELOCITY PLANNING ===
        optimal_velocity = self.calculate_energy_optimal_velocity(energy_state)

        # === TRAFFIC ADAPTATION ===
        adapted_velocity = self.adapt_to_traffic_with_energy_awareness(optimal_velocity, energy_state)

        # === PID VELOCITY CONTROL ===
        target_acceleration = self.pid_velocity_control(adapted_velocity, step_size)

        # === ENERGY-AWARE TORQUE CALCULATION ===
        self.motor_trq_target = self.calculate_energy_aware_torque(target_acceleration, energy_state)

        if self.brk_pedal > 0.2:
            self.motor_trq_target = 0.0

        # Safety fallback if velocity is too low but torque is high
        if self.act_vel < 0.5 and abs(self.motor_trq_target) > 200.0:
            self.motor_trq_target = 0.0  # Cancel torque to avoid launch error

        return True

    def update_energy_consumption(self, step_size):
        """Track energy consumption and update predictions"""
        voltage = 400.0  # V, assume constant
        power_kW = abs(self.batt_current * voltage) / 1000
        energy_delta = power_kW * step_size / 3600  # kWh
        self.energy_used += energy_delta

        # Update consumption history for learning
        distance_delta = self.distance_covered - self.previous_distance
        if distance_delta > 0.01:  # Only update if we've moved significantly
            consumption_rate = energy_delta / (distance_delta / 1000.0)  # kWh/km
            self.energy_consumption_history.append(consumption_rate)
            self.distance_history.append(self.distance_covered)

            # Keep only recent history (last 50 data points)
            if len(self.energy_consumption_history) > 50:
                self.energy_consumption_history.pop(0)
                self.distance_history.pop(0)

            # Update average consumption rate
            if len(self.energy_consumption_history) > 5:
                self.average_consumption_rate = sum(self.energy_consumption_history[-10:]) / min(10,
                                                                                                 len(self.energy_consumption_history))

        self.previous_distance = self.distance_covered

    def analyze_energy_state(self):
        """Analyze current energy state and predict range"""
        distance_remaining_to_target = max(0.0, self.target_distance - self.distance_covered)
        energy_remaining = self.soc * self.batt_cap

        # Predict energy needed based on consumption history
        if distance_remaining_to_target > 0:
            # Adjust consumption prediction based on terrain and conditions
            terrain_factor = 1.0
            if self.slope > 2:
                terrain_factor = 1.0 + (self.slope * 0.1)  # 10% more per degree uphill
            elif self.slope < -2:
                terrain_factor = max(0.7, 1.0 + (self.slope * 0.05))  # Some benefit from downhill

            predicted_consumption = self.average_consumption_rate * terrain_factor
            energy_needed = (distance_remaining_to_target / 1000.0) * predicted_consumption

            # Energy margin (positive = surplus, negative = deficit)
            energy_margin = energy_remaining - energy_needed
            energy_margin_percent = energy_margin / self.batt_cap

            return {
                'soc': self.soc,
                'energy_remaining': energy_remaining,
                'distance_to_target': distance_remaining_to_target,
                'energy_needed': energy_needed,
                'energy_margin': energy_margin,
                'energy_margin_percent': energy_margin_percent,
                'predicted_consumption': predicted_consumption,
                'is_critical': self.soc < self.critical_soc_threshold or energy_margin_percent < -0.05,
                'is_conservative': self.soc < self.conservative_soc_threshold or energy_margin_percent < 0.1,
                'can_reach_target': energy_margin > 0
            }
        else:
            return {
                'soc': self.soc,
                'energy_remaining': energy_remaining,
                'distance_to_target': 0,
                'energy_needed': 0,
                'energy_margin': energy_remaining,
                'energy_margin_percent': 1.0,
                'predicted_consumption': self.average_consumption_rate,
                'is_critical': False,
                'is_conservative': False,
                'can_reach_target': True
            }

    def calculate_energy_optimal_velocity(self, energy_state):
        """Calculate optimal velocity for energy efficiency with traffic & slope awareness"""

        # Energy-optimal base velocity (EVs most efficient around 45–55 km/h)
        base_velocity = 14.0  # m/s (~50 km/h)

        # ==== Low-speed Traffic Compensation ====
        if self.act_vel < 2.0:
            base_velocity = 10.0  # lower base speed in traffic

        # === Energy State Adaptation ===
        if energy_state['is_critical']:
            velocity_factor = 0.55  # Very conservative
        elif energy_state['is_conservative']:
            velocity_factor = 0.7  # Reduced to stretch battery
        elif not energy_state['can_reach_target']:
            velocity_factor = 0.75  # Energy deficit fallback
        else:
            velocity_factor = 1.0  # Normal case

        # === Terrain-based slope adaptation ===
        if self.slope > 3:
            velocity_factor *= max(0.45, 1.0 - (self.slope - 3) * 0.08)  # steep uphill
        elif self.slope < -3:
            velocity_factor *= min(1.25, 1.0 + abs(self.slope) * 0.035)  # downhill roll

        # === Target distance-based urgency (final push) ===
        if energy_state['distance_to_target'] < 20000:
            urgency_boost = min(1.1, 1.0 + (20000 - energy_state['distance_to_target']) / 80000)
            velocity_factor *= urgency_boost

        # === Final velocity clamp and output ===
        optimal_velocity = base_velocity * velocity_factor
        return max(6.0, min(optimal_velocity, 22.0))  # ~21–22 m/s max for efficiency

    def adapt_to_traffic_with_energy_awareness(self, target_vel, energy_state):
        """Adapt to traffic while considering energy constraints"""
        safe_distance = max(12.0, self.act_vel * 1.8)  # 1.8s following distance

        # If clear road, use target velocity
        if self.front_vehicle_distance > safe_distance:
            self.overtaking = False
            return target_vel

        # Traffic adaptation with energy awareness
        if self.front_vehicle_distance <(safe_distance + 5.0):
            front_vel = self.front_vehicle_velocity

            # Decide whether to overtake based on energy state
            should_overtake = (
                    front_vel < target_vel * 0.8 and  # Front vehicle significantly slower
                    not energy_state['is_critical'] and  # Not in critical energy state
                    self.front_vehicle_distance > 8.0 and  # Safe distance for overtaking
                    energy_state['energy_margin_percent'] > 0.05  # Have some energy margin
            )

            if should_overtake and not self.overtaking:
                self.overtaking = True
                return min(target_vel, front_vel + 3.0)  # Moderate overtaking speed
            elif self.overtaking:
                # Continue overtaking but be energy-conscious
                if energy_state['is_conservative']:
                    return min(target_vel, front_vel + 1.5)  # Gentle overtaking
                else:
                    return min(target_vel, front_vel + 4.0)  # Normal overtaking
            else:
                # Follow at efficient speed
                self.overtaking = False
                return min(target_vel, front_vel - 0.5)  # Follow slightly slower for efficiency

        return target_vel

    def pid_velocity_control(self, target_velocity, step_size):
        """PID controller for smooth velocity tracking"""
        # Velocity error
        velocity_error = target_velocity - self.act_vel
        # Add this at the beginning:
        if self.act_vel < 2.0:
            self.kp = 50.0
            self.ki = 5.0
            self.kd = 2.0
        else:
            self.kp = 100.0
            self.ki = 15.0
            self.kd = 8.0

        # Proportional term
        p_term = self.kp * velocity_error

        # Integral term with anti-windup
        self.velocity_error_integral += velocity_error * step_size
        self.velocity_error_integral = max(min(self.velocity_error_integral, 5.0), -5.0)  # Anti-windup
        i_term = self.ki * self.velocity_error_integral

        # Derivative term
        velocity_error_derivative = (velocity_error - self.velocity_error_previous) / step_size if step_size > 0 else 0
        d_term = self.kd * velocity_error_derivative

        # PID output (acceleration)
        # pid_acceleration = (p_term + i_term + d_term) / 100.0  # Scale down
        pid_acceleration = (p_term + i_term + d_term) / 150.0  # Previously 100.0

        # Limit acceleration for energy efficiency
        max_accel = 1.0  # m/s² - gentle acceleration for efficiency
        max_decel = -2.5  # m/s² - allow stronger deceleration

        self.velocity_error_previous = velocity_error

        return max(min(pid_acceleration, max_accel), max_decel)

    def calculate_energy_aware_torque(self, target_acceleration, energy_state):
        """Calculate motor torque with energy efficiency optimization and smoother control"""

        # === COMPREHENSIVE FORCE CALCULATION ===

        # 1. Slope resistance
        slope_force = self.vehicle_mass * 9.81 * math.sin(math.radians(self.slope))

        # 2. Rolling resistance
        rolling_force = (
                self.vehicle_mass * 9.81 *
                math.cos(math.radians(self.slope)) *
                self.rolling_resistance
        )

        # 3. Aerodynamic drag
        aero_force = (
                0.5 * self.air_density * self.frontal_area *
                self.aero_drag_coeff * self.act_vel ** 2
        )

        # 4. Acceleration (inertial) force
        inertial_force = self.vehicle_mass * target_acceleration

        # Total tractive force
        total_force = slope_force + rolling_force + aero_force + inertial_force

        # === MOTOR TORQUE CALCULATION ===
        base_torque = total_force * self.wheel_radius / self.motor_efficiency

        # === LOW-SPEED HANDLING & TORQUE CLAMPING ===

        # Prevent wheelspin and stiction at near standstill
        if self.act_vel < 1.0 and target_acceleration > 0:
            base_torque = min(base_torque, 40.0)  # gentler than 100.0
        elif self.act_vel < 0.3 and target_acceleration > 0:
            base_torque = 15.0  # max torque from 0 to creep smoothly

        # === ENERGY-AWARE TORQUE LIMITS ===
        if energy_state['is_critical']:
            max_torque = 110.0
            max_regen = -300.0
        elif energy_state['is_conservative']:
            max_torque = 160.0
            max_regen = -350.0
        else:
            max_torque = 250.0
            max_regen = -400.0

        # === CURRENT PROTECTION THRESHOLD ===
        if abs(self.batt_current) > self.max_current_threshold * 0.85:
            limit_factor = 0.6  # more strict
            max_torque *= limit_factor
            max_regen *= limit_factor

        # === BRAKING-BASED REGEN CONTROL ===
        if self.brk_pedal > 0.4 and self.act_vel > 3.0:
            motor_torque = max(-150.0, base_torque - 50.0)  # regen bias
        else:
            motor_torque = base_torque

        # === FINAL TORQUE LIMITING AND SANITY ===
        motor_torque = max(min(motor_torque, max_torque), max_regen)

        # Remove NaNs/Infs
        if not math.isfinite(motor_torque):
            motor_torque = 0.0

        return motor_torque

