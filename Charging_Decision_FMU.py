import math
from pythonfmu import Fmi2Slave, Fmi2Causality, Fmi2Variability, Real, Boolean, Integer


class ChargingDecisionController(Fmi2Slave):
    author = "Team EVolvers - Phase 2 Charging Decision"
    description = "Intelligent charging decision FMU for EV Challenge Phase 2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Inputs from CarMaker (keep existing parameters)
        self.act_vel = 0.0
        self.brk_pedal = 0.0
        self.distance_covered = 0.0
        self.slope = 0.0
        self.soc = 1.0
        self.rolling_resistance = 0.008
        self.air_density = 1.225

        # Additional inputs for charging decision
        self.target_soc = 0.8  # Target SoC after charging
        self.charging_station_selected = 0  # Selected charging station ID
        self.charging_available = False  # Charging slot availability

        # Outputs
        self.motor_trq_target = 0.0
        self.charging_request = False  # Request to charge
        self.reachable_distance = 0.0  # Maximum reachable distance
        self.reachable_stations = []  # List of reachable charging stations
        self.target_soc_output = 0.0  # Target SoC for completing route
        self.selected_station_id = 0  # Selected charging station

        # Constants
        self.batt_cap = 60.0  # kWh
        self.target_distance = 200000.0  # meters (200 km route)
        self.vehicle_mass = 1500.0
        self.wheel_radius = 0.3
        self.motor_efficiency = 0.88
        self.aero_drag_coeff = 0.28
        self.frontal_area = 2.3

        # Charging station data (from Phase 2 announcement)
        self.charging_stations = {
            1: {"label": "CH1", "distance": 2.38, "available_slots": 2},
            2: {"label": "CH2", "distance": 12.62, "available_slots": 1},
            3: {"label": "CH3", "distance": 24.30, "available_slots": 3},
            4: {"label": "CH4", "distance": 55.37, "available_slots": 1},
            5: {"label": "CH5", "distance": 95.11, "available_slots": 2}
        }

        # Energy consumption parameters
        self.avg_kwh_per_km = 0.15  # Average energy consumption
        self.energy_buffer_factor = 1.2  # 20% safety margin
        self.min_charging_soc = 0.15  # Minimum SoC to trigger charging
        self.emergency_soc = 0.08  # Emergency SoC threshold

        # Charging decision parameters
        self.charging_time_estimate = 30.0  # minutes for full charge
        self.charging_efficiency = 0.95  # Charging efficiency
        self.is_charging = False
        self.charging_start_time = 0.0

        # PID controller parameters (from original FMU)
        self.kp = 50.0
        self.ki = 6.0
        self.kd = 4.0
        self.vel_err_int = 0.0
        self.vel_err_prev = 0.0
        self.max_torque = 100.0
        self.max_brake_trq = -180.0

        self.register_variables()

    def register_variables(self):
        # Input variables
        for name in ["act_vel", "brk_pedal", "distance_covered", "slope", "soc",
                     "rolling_resistance", "air_density", "target_soc",
                     "charging_station_selected", "charging_available"]:
            self.register_variable(
                Real(name, causality=Fmi2Causality.input, variability=Fmi2Variability.continuous)
            )

        # Output variables
        for name in ["motor_trq_target", "reachable_distance", "target_soc_output"]:
            self.register_variable(
                Real(name, causality=Fmi2Causality.output, variability=Fmi2Variability.continuous)
            )

        for name in ["charging_request", "charging_available"]:
            self.register_variable(
                Boolean(name, causality=Fmi2Causality.output, variability=Fmi2Variability.discrete)
            )

        self.register_variable(
            Integer("selected_station_id", causality=Fmi2Causality.output, variability=Fmi2Variability.discrete)
        )

    def do_step(self, current_time: float, step_size: float) -> bool:
        # Calculate reachable distance based on current SoC
        self.reachable_distance = self.calculate_reachable_distance()

        # Check if charging is needed
        charging_needed = self.evaluate_charging_need()

        # Find reachable charging stations
        self.reachable_stations = self.find_reachable_stations()

        # Calculate target SoC for route completion
        self.target_soc_output = self.calculate_target_soc()

        # Make charging decision
        if charging_needed and self.reachable_stations:
            self.charging_request = True
            self.selected_station_id = self.select_optimal_station()
        else:
            self.charging_request = False
            self.selected_station_id = 0

        # Calculate motor torque (keep existing logic)
        self.motor_trq_target = self.calculate_motor_torque(step_size)

        # Handle charging process
        if self.charging_available and self.charging_request:
            self.handle_charging_process(current_time, step_size)

        return True

    def calculate_reachable_distance(self):
        """Calculate maximum reachable distance with current SoC"""
        available_energy = self.soc * self.batt_cap  # kWh
        # Account for energy buffer and efficiency losses
        usable_energy = available_energy * 0.9  # 10% reserve
        max_distance_km = usable_energy / self.avg_kwh_per_km
        return max_distance_km * 1000  # Convert to meters

    def evaluate_charging_need(self):
        """Evaluate if charging is needed based on route requirements"""
        remaining_distance = self.target_distance - self.distance_covered

        # Check if current SoC is below minimum threshold
        if self.soc <= self.min_charging_soc:
            return True

        # Check if reachable distance is insufficient for remaining route
        if self.reachable_distance < (remaining_distance * self.energy_buffer_factor):
            return True

        # Check for emergency situation
        if self.soc <= self.emergency_soc:
            return True

        return False

    def find_reachable_stations(self):
        """Find charging stations within reachable distance"""
        reachable = []
        current_position_km = self.distance_covered / 1000.0

        for station_id, station_data in self.charging_stations.items():
            station_distance_from_start = station_data["distance"]
            distance_to_station = abs(station_distance_from_start - current_position_km)

            # Check if station is reachable with current SoC
            if distance_to_station * 1000 <= self.reachable_distance:
                # Check if station has available slots
                if station_data["available_slots"] > 0:
                    reachable.append({
                        "id": station_id,
                        "distance": distance_to_station,
                        "available_slots": station_data["available_slots"],
                        "label": station_data["label"]
                    })

        # Sort by distance (closest first)
        reachable.sort(key=lambda x: x["distance"])
        return reachable

    def calculate_target_soc(self):
        """Calculate target SoC needed to complete the route"""
        remaining_distance_km = (self.target_distance - self.distance_covered) / 1000.0

        # Energy needed for remaining distance
        energy_needed = remaining_distance_km * self.avg_kwh_per_km * self.energy_buffer_factor

        # Target SoC
        target_soc = energy_needed / self.batt_cap

        # Ensure target SoC is within reasonable bounds
        target_soc = max(0.2, min(target_soc, 1.0))

        return target_soc

    def select_optimal_station(self):
        """Select the optimal charging station based on distance and availability"""
        if not self.reachable_stations:
            return 0

        # Simple strategy: select closest station with available slots
        for station in self.reachable_stations:
            if station["available_slots"] > 0:
                return station["id"]

        return 0

    def calculate_motor_torque(self, step_size):
        """Calculate motor torque using PID controller (simplified version)"""
        # Target velocity based on current conditions
        target_vel = self.get_target_velocity()

        # PID control
        error = target_vel - self.act_vel
        self.vel_err_int += error * step_size

        # Limit integral windup
        self.vel_err_int = max(min(self.vel_err_int, 1.0), -1.0)

        # PID calculation
        p_term = self.kp * error
        i_term = self.ki * self.vel_err_int
        d_term = self.kd * (error - self.vel_err_prev) / step_size if step_size > 0 else 0.0

        self.vel_err_prev = error

        # Convert to torque
        accel = (p_term + i_term + d_term) / 400.0
        torque = self.calculate_torque_from_accel(accel)

        # Apply torque limits
        return max(min(torque, self.max_torque), self.max_brake_trq)

    def get_target_velocity(self):
        """Get target velocity based on current conditions"""
        # Base target velocity (60-70 km/h range)
        base_speed = 18.0  # m/s (65 km/h)

        # Adjust based on SoC
        if self.soc < 0.2:
            return base_speed * 0.8  # Slower for energy conservation
        elif self.soc < 0.5:
            return base_speed * 0.9
        else:
            return base_speed

    def calculate_torque_from_accel(self, accel):
        """Calculate required torque from desired acceleration"""
        g = 9.81

        # Forces calculation
        slope_force = self.vehicle_mass * g * math.sin(math.radians(self.slope))
        rolling_force = self.vehicle_mass * g * self.rolling_resistance
        drag_force = 0.5 * self.air_density * self.frontal_area * self.aero_drag_coeff * (self.act_vel ** 2)
        inertial_force = self.vehicle_mass * accel

        total_force = slope_force + rolling_force + drag_force + inertial_force
        torque = (total_force * self.wheel_radius) / self.motor_efficiency

        return torque

    def handle_charging_process(self, current_time, step_size):
        """Handle the charging process simulation"""
        if not self.is_charging:
            self.is_charging = True
            self.charging_start_time = current_time

        # Simulate charging progress
        charging_duration = current_time - self.charging_start_time
        charging_rate = 0.5 / 60.0  # 50% charge per hour (simplified)

        # Update SoC during charging
        if charging_duration > 0:
            soc_increase = charging_rate * (step_size / 60.0)  # Convert to minutes
            self.soc = min(self.soc + soc_increase, self.target_soc)

        # Check if charging is complete
        if self.soc >= self.target_soc or charging_duration > self.charging_time_estimate * 60:
            self.is_charging = False
            self.charging_request = False