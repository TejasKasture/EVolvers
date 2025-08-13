from pythonfmu import Fmi2Slave, Fmi2Causality, Fmi2Variability, Real
import math
from enum import Enum
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


class ChargingMode(Enum):
    CONSERVATIVE = 1
    BALANCED = 2
    AGGRESSIVE = 3


@dataclass
class TravelSegment:
    """Data class to store 1km travel segment information"""
    distance: float  # km
    energy_consumed: float  # kWh
    avg_speed: float  # km/h
    terrain_factor: float
    soc_start: float  # %
    soc_end: float  # %
    timestamp: float
    consumption_rate: float  # kWh/km


class EnhancedHistoricalData:
    """Enhanced historical data management with 1km segment storage"""

    def __init__(self, max_segments: int = 200):
        self.segments: List[TravelSegment] = []
        self.max_segments = max_segments
        self.current_segment_start_distance = 0.0
        self.current_segment_start_soc = 0.0
        self.current_segment_start_time = 0.0

    def should_record_segment(self, current_distance: float) -> bool:
        """Check if we should record a new 1km segment"""
        return (current_distance - self.current_segment_start_distance) >= 1.0

    def record_segment(self, distance: float, soc: float, speed: float,
                       terrain: float, current_time: float, energy_consumed: float):
        """Record a completed 1km segment"""
        if len(self.segments) == 0:
            # First segment initialization
            self.current_segment_start_distance = max(0, distance - 1.0)
            self.current_segment_start_soc = soc + (energy_consumed / 75.0 * 100)  # Estimate start SoC
            self.current_segment_start_time = current_time - (1.0 / max(speed, 1.0))

        segment_distance = distance - self.current_segment_start_distance
        if segment_distance <= 0:
            return

        consumption_rate = energy_consumed / segment_distance if segment_distance > 0 else 0.2

        # Validate consumption rate is reasonable (0.1 - 0.5 kWh/km)
        if 0.1 <= consumption_rate <= 0.5:
            segment = TravelSegment(
                distance=segment_distance,
                energy_consumed=energy_consumed,
                avg_speed=speed,
                terrain_factor=terrain,
                soc_start=self.current_segment_start_soc,
                soc_end=soc,
                timestamp=current_time,
                consumption_rate=consumption_rate
            )

            self.segments.append(segment)

            # Remove oldest segments if we exceed max
            if len(self.segments) > self.max_segments:
                self.segments.pop(0)

        # Update for next segment
        self.current_segment_start_distance = distance
        self.current_segment_start_soc = soc
        self.current_segment_start_time = current_time

    def get_adaptive_consumption(self) -> float:
        """Calculate adaptive energy consumption based on historical data"""
        if len(self.segments) < 3:
            return 0.22 * 1.1  # 10% safety margin when no data

        # Calculate recent consumption (last 5 segments for current conditions)
        recent_segments = self.segments[-5:] if len(self.segments) >= 5 else self.segments
        recent_consumption = sum(s.consumption_rate for s in recent_segments) / len(recent_segments)

        # Calculate overall historical average
        total_distance = sum(s.distance for s in self.segments)
        total_energy = sum(s.energy_consumed for s in self.segments)
        historical_avg = total_energy / total_distance if total_distance > 0 else 0.22

        # Baseline from vehicle characteristics
        baseline = 0.20  # Conservative baseline

        # Weighted blend: 60% recent, 20% historical, 20% baseline
        adaptive_consumption = (recent_consumption * 0.6 +
                                historical_avg * 0.2 +
                                baseline * 0.2)

        return adaptive_consumption

    def predict_consumption_for_conditions(self, speed: float, terrain: float) -> float:
        """Predict consumption rate for given driving conditions"""
        base_consumption = self.get_adaptive_consumption()

        # Speed factor (efficiency curve)
        if speed <= 30:
            speed_factor = 1.2  # City driving with stops
        elif speed <= 50:
            speed_factor = 0.9  # Optimal city speed
        elif speed <= 80:
            speed_factor = 1.0  # Optimal highway speed
        elif speed <= 120:
            speed_factor = 1.3  # Higher highway speed
        else:
            speed_factor = 1.5  # Very high speed

        # Apply factors
        predicted_consumption = base_consumption * speed_factor * terrain

        return predicted_consumption


class charge_fmu:
    """
    FIXED Charging Decision FMU - Corrected charging session management
    """

    def __init__(self):
        # FMU Parameters
        self.last_distance = 0.0
        self.initial_soc = 0.0  # %
        self.minimum_threshold_soc = 10.0  # %
        self.total_route_distance = 153.0  # km
        self.station_distances = [2.38, 12.62, 24.30, 55.37, 95.11]  # km
        self.distance_travelled = 0.0  # km

        # Vehicle Parameters
        self.battery_capacity = 75.0  # kWh
        self.vehicle_mass = 1500.0  # kg

        # Enhanced Historical Data
        self.historical_data = EnhancedHistoricalData()

        # Legacy historical data for compatibility
        self.historical_consumption = []
        self.max_history_entries = 100

        # Internal Parameters
        self.reserve_soc_margin = 5.0  # %
        self.average_energy_consumption = 0.22  # kWh/km
        self.charging_mode = ChargingMode.BALANCED

        # FIXED: Charging State Management with proper session tracking
        self.charging_session_active = False
        self.charging_target_soc = 0.0
        self.charging_hysteresis = 3.0  # % - increased to prevent oscillation
        self.last_charge_location = -1  # Track where we last charged
        self.charge_completion_distance = -1  # Distance where charging completed

        # Outputs
        self.charging_request = False
        self.reachable_stations = 0  # Changed to single int (station ID)
        self.target_soc = 0.0

        # Driving conditions
        self.current_speed = 60.0  # km/h
        self.terrain_factor = 1.0
        self.current_time = 0.0

        # Debug flag
        self.debug = True

    def set_inputs(self, soc: float, min_threshold: float, distance: float):
        """Set FMU input parameters"""
        self.initial_soc = soc
        self.minimum_threshold_soc = min_threshold
        self.distance_travelled = distance

    def set_driving_conditions(self, speed: float = None, terrain: float = None, time: float = None):
        """Set current driving conditions"""
        if speed is not None:
            self.current_speed = speed
        if terrain is not None:
            self.terrain_factor = terrain
        if time is not None:
            self.current_time = time

    def add_historical_data(self, distance: float, energy_used: float):
        """Add historical consumption data for adaptive learning"""
        if distance > 0 and energy_used > 0:
            consumption_rate = energy_used / distance
            # Validate consumption is within reasonable bounds (0.10 - 0.50 kWh/km)
            if 0.10 <= consumption_rate <= 0.50:
                self.historical_consumption.append((distance, energy_used))
                if len(self.historical_consumption) > self.max_history_entries:
                    self.historical_consumption.pop(0)

    def update_enhanced_historical_data(self):
        """Update enhanced historical data with 1km segments"""
        if self.historical_data.should_record_segment(self.distance_travelled):
            distance_increment = self.distance_travelled - self.historical_data.current_segment_start_distance
            if distance_increment > 0:
                energy_estimate = self.estimate_energy_for_segment(distance_increment)

                self.historical_data.record_segment(
                    distance=self.distance_travelled,
                    soc=self.initial_soc,
                    speed=self.current_speed,
                    terrain=self.terrain_factor,
                    current_time=self.current_time,
                    energy_consumed=energy_estimate
                )

                if self.debug:
                    print(f"Debug - Enhanced segment: {distance_increment:.2f}km, "
                          f"Energy: {energy_estimate:.3f}kWh, Rate: {energy_estimate / distance_increment:.3f}kWh/km")

    def estimate_energy_for_segment(self, distance: float) -> float:
        """Estimate energy consumption for a distance segment"""
        consumption_rate = self.historical_data.predict_consumption_for_conditions(
            self.current_speed, self.terrain_factor)
        return distance * consumption_rate

    def calculate_adaptive_consumption(self) -> float:
        """Calculate adaptive energy consumption"""
        # Use enhanced data if available, otherwise fall back to legacy method
        if len(self.historical_data.segments) >= 3:
            adaptive_consumption = self.historical_data.get_adaptive_consumption()
        else:
            # Legacy fallback logic
            baseline = self.calculate_baseline_consumption()

            if len(self.historical_consumption) < 3:
                return baseline * 1.1  # 10% safety margin when no data

            # Calculate recent consumption (last 5 entries for current conditions)
            recent_entries = self.historical_consumption[-5:] if len(
                self.historical_consumption) >= 5 else self.historical_consumption
            recent_consumption = sum(energy / dist for dist, energy in recent_entries) / len(recent_entries)

            # Calculate overall historical average
            total_distance = sum(dist for dist, _ in self.historical_consumption)
            total_energy = sum(energy for _, energy in self.historical_consumption)
            historical_avg = total_energy / total_distance if total_distance > 0 else baseline

            # Weighted blend: 60% recent, 20% historical, 20% baseline
            adaptive_consumption = (recent_consumption * 0.6 +
                                    historical_avg * 0.2 +
                                    baseline * 0.2)

        return adaptive_consumption

    def calculate_baseline_consumption(self) -> float:
        """Calculate baseline energy consumption"""
        mass_factor = self.vehicle_mass / 1800.0
        base_consumption = 0.18 * mass_factor
        return base_consumption

    def calculate_energy_consumption(self) -> float:
        """Calculate energy consumption using adaptive method"""
        consumption = self.calculate_adaptive_consumption()

        # Apply charging mode adjustment
        if self.charging_mode == ChargingMode.CONSERVATIVE:
            consumption *= 1.15  # 15% higher estimate for safety
        elif self.charging_mode == ChargingMode.AGGRESSIVE:
            consumption *= 0.95  # 5% lower estimate for efficiency

        return consumption

    def estimate_energy_used(self, distance: float) -> float:
        """Improved energy estimation"""
        if distance <= 0:
            return 0.0

        # Use enhanced prediction if available
        if len(self.historical_data.segments) >= 3:
            return self.estimate_energy_for_segment(distance)

        # Fall back to working version
        base_consumption = self.calculate_energy_consumption()

        # Speed factor (efficiency curve - most efficient around 50-70 km/h)
        if self.current_speed <= 30:
            speed_factor = 1.2
        elif self.current_speed <= 50:
            speed_factor = 0.9
        elif self.current_speed <= 80:
            speed_factor = 1.0
        elif self.current_speed <= 120:
            speed_factor = 1.3
        else:
            speed_factor = 1.5

        # Apply factors
        estimated_consumption = base_consumption * speed_factor * self.terrain_factor

        return distance * estimated_consumption

    def calculate_reachable_distance(self) -> float:
        """Calculate maximum reachable distance"""
        self.average_energy_consumption = self.calculate_energy_consumption()
        usable_soc = max(0.0, self.initial_soc - self.reserve_soc_margin)
        usable_energy = (usable_soc / 100) * self.battery_capacity
        reachable_distance = usable_energy / self.average_energy_consumption
        return reachable_distance

    def check_charging_needed(self) -> bool:
        """FIXED: Check if charging is required with better logic"""
        remaining_distance = self.total_route_distance - self.distance_travelled
        reachable_distance = self.calculate_reachable_distance()

        if self.debug:
            print(
                f"Debug - SoC: {self.initial_soc:.1f}%, Remaining: {remaining_distance:.1f}km, Reachable: {reachable_distance:.1f}km")

        # FIXED: Add minimum distance buffer to prevent charging immediately after completion
        min_distance_since_charge = 5.0  # km - minimum distance before next charge consideration
        distance_since_charge = self.distance_travelled - self.charge_completion_distance

        if self.charge_completion_distance > 0 and distance_since_charge < min_distance_since_charge:
            if self.debug:
                print(f"Debug - Too soon since last charge: {distance_since_charge:.1f}km < {min_distance_since_charge}km")
            return False

        # Check if can't reach destination with current charge
        if reachable_distance < remaining_distance:
            if self.debug:
                print(f"Debug - Charging needed: Cannot reach destination")
            return True

        # Check if final SoC would be below minimum threshold
        energy_needed = remaining_distance * self.average_energy_consumption
        current_energy = (self.initial_soc / 100) * self.battery_capacity
        final_energy = current_energy - energy_needed
        final_soc = (final_energy / self.battery_capacity) * 100

        if final_soc < self.minimum_threshold_soc:
            if self.debug:
                print(
                    f"Debug - Charging needed: Final SoC ({final_soc:.1f}%) below threshold ({self.minimum_threshold_soc}%)")
            return True

        if self.debug:
            print(f"Debug - No charging needed: Final SoC would be {final_soc:.1f}%")
        return False

    def manage_charging_session(self) -> bool:
        """FIXED: Manage charging session with proper state tracking"""
        # If no charging session is active, check if we need to start one
        if not self.charging_session_active:
            charging_needed = self.check_charging_needed()
            if self.debug:
                print(f"Debug - No active session, charging needed: {charging_needed}")

            if charging_needed:
                # Start new charging session
                self.charging_session_active = True
                self.charging_target_soc = self.calculate_target_soc()
                if self.debug:
                    print(f"Debug - Starting charging session, target SoC: {self.charging_target_soc:.1f}%")
                return True
            else:
                return False

        # If charging session is active, check if we should stop it
        else:
            if self.debug:
                print(
                    f"Debug - Active session, current SoC: {self.initial_soc:.1f}%, target: {self.charging_target_soc:.1f}%")

            # FIXED: Stop charging when target is reached
            if self.initial_soc >= self.charging_target_soc:
                if self.debug:
                    print(f"Debug - Charging target reached, stopping session")
                self.charging_session_active = False
                self.charge_completion_distance = self.distance_travelled  # Track where charging completed
                self.charging_target_soc = 0.0
                return False
            else:
                # Continue charging - maintain the original target
                self.target_soc = self.charging_target_soc
                if self.debug:
                    print(f"Debug - Continuing charging session")
                return True

    def find_reachable_stations(self) -> int:
        """FIXED: Find reachable stations - return station ID, not list"""
        reachable_distance = self.calculate_reachable_distance()
        reachable_stations = []

        for i, station_absolute_dist in enumerate(self.station_distances):
            distance_to_station = station_absolute_dist - self.distance_travelled

            # Station must be ahead of current position and within reachable distance
            if distance_to_station > 0 and distance_to_station <= reachable_distance:
                reachable_stations.append(i + 1)  # Station IDs start from 1

        # FIXED: Return the highest reachable station ID (farthest station we can reach)
        return max(reachable_stations) if reachable_stations else 0

    def calculate_target_soc(self) -> float:
        """Calculate target SoC for charging"""
        remaining_distance = self.total_route_distance - self.distance_travelled

        # Use enhanced consumption if available
        if len(self.historical_data.segments) >= 3:
            required_energy = remaining_distance * self.historical_data.predict_consumption_for_conditions(
                self.current_speed, self.terrain_factor)
        else:
            required_energy = remaining_distance * self.average_energy_consumption

        # Apply charging mode strategy with enhanced logic
        confidence = min(len(self.historical_data.segments) / 50.0, 1.0)

        if self.charging_mode == ChargingMode.CONSERVATIVE:
            safety_factor = 1.20 + (0.10 * (1 - confidence))
            max_target = 85.0
        elif self.charging_mode == ChargingMode.BALANCED:
            safety_factor = 1.15 + (0.05 * (1 - confidence))
            max_target = 80.0
        else:  # AGGRESSIVE
            safety_factor = 1.10 + (0.05 * (1 - confidence))
            max_target = 75.0

        required_energy_with_safety = required_energy * safety_factor
        target_soc = (required_energy_with_safety / self.battery_capacity) * 100

        # Add reserve margin
        target_soc += self.reserve_soc_margin

        # FIXED: Ensure target is significantly higher than current to justify charging
        min_charging_gain = 20.0  # Minimum 20% gain to justify charging
        target_soc = max(target_soc, self.initial_soc + min_charging_gain)

        # Ensure target is achievable and within limits
        target_soc = min(target_soc, max_target)

        return target_soc

    def execute_charging_decision(self) -> List:
        """FIXED: Main execution function with corrected logic"""
        # Step 1: Update enhanced historical data
        self.update_enhanced_historical_data()

        # Step 2: Manage charging session
        self.charging_request = self.manage_charging_session()

        # Step 3: Find reachable stations (return single station ID)
        self.reachable_stations = self.find_reachable_stations()

        # Step 4: Set target SoC
        if self.charging_request and self.charging_session_active:
            self.target_soc = self.charging_target_soc
        else:
            self.target_soc = self.initial_soc

        if self.debug:
            segments_count = len(self.historical_data.segments)
            print(
                f"Debug - FIXED decision: Request={self.charging_request}, MaxStation={self.reachable_stations}, "
                f"Target={self.target_soc:.1f}%, Segments={segments_count}")

        return [self.charging_request, self.reachable_stations, self.target_soc]

    def do_step(self, soc: float, min_threshold: float, distance: float) -> List:
        """Execute one step"""
        # Set charging mode to balanced by default
        self.set_charging_mode(ChargingMode.BALANCED)

        # Set inputs
        self.set_inputs(soc, min_threshold, distance)

        # Execute decision logic
        result = self.execute_charging_decision()

        # Add estimated energy consumption as historical data
        if self.last_distance is not None and distance > self.last_distance:
            distance_increment = distance - self.last_distance
            if distance_increment > 0:
                energy_used = self.estimate_energy_used(distance_increment)
                self.add_historical_data(distance_increment, energy_used)

        self.last_distance = distance

        return result

    def set_charging_mode(self, mode: ChargingMode):
        """Set charging strategy mode"""
        self.charging_mode = mode


class ChargingFMU(Fmi2Slave):
    """FIXED FMI2 wrapper with corrected charging logic"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Create internal FMU logic
        self.fmu = charge_fmu()

        # ---- Inputs ----
        self.soc = 0.0
        self.distance_covered = 0.0
        self.current_speed = 60.0

        # ---- Outputs ----
        self.charging_request = 0
        self.reachable_stations = 0  # Now single station ID
        self.target_soc = 0.0

        # Initialize step counter for debugging
        self.step_count = 0

        self.register_variable(
            Real("soc", causality=Fmi2Causality.input, variability=Fmi2Variability.continuous))

        self.register_variable(
            Real("distance_covered", causality=Fmi2Causality.input, variability=Fmi2Variability.continuous))

        self.register_variable(
            Real("current_speed", causality=Fmi2Causality.input, variability=Fmi2Variability.continuous))

        self.register_variable(
            Real("charging_request", causality=Fmi2Causality.output, variability=Fmi2Variability.continuous))

        self.register_variable(
            Real("reachable_stations", causality=Fmi2Causality.output, variability=Fmi2Variability.continuous))

        self.register_variable(
            Real("target_soc", causality=Fmi2Causality.output, variability=Fmi2Variability.continuous))

    def do_step(self, current_time: float, step_size: float) -> bool:
        """FIXED do_step with corrected logic"""
        self.step_count += 1

        try:
            # Set driving conditions using current speed
            self.fmu.set_driving_conditions(speed=self.current_speed, time=current_time)

            # Convert distance from meters to kilometers
            distance_km = self.distance_covered / 1000.0

            # Perform FMU step
            result = self.fmu.do_step(self.soc, 10.0, distance_km)

            # Convert boolean to int for charging_request
            self.charging_request = 1 if result[0] else 0
            self.reachable_stations = result[1]  # Single station ID
            self.target_soc = result[2]

            # Enhanced debugging
            if self.step_count <= 10 or self.step_count % 50 == 0:
                print(f"FIXED Step {self.step_count}: SoC={self.soc:.1f}%, "
                      f"Dist={distance_km:.2f}km, Charge={self.charging_request}, "
                      f"MaxStation={self.reachable_stations}, Target={self.target_soc:.1f}%")

            return True

        except Exception as e:
            print(f"Error in FIXED ChargingFMU do_step: {e}")
            # Set safe default values
            self.charging_request = 0
            self.reachable_stations = 0
            self.target_soc = self.soc
            return False