import asyncio
import json
import time
import logging
from typing import Dict, Any, List
from kuksa_client.grpc import VSSClient
from kuksa_client.grpc import Datapoint
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PyCarmakerIntegration:
    """
    PyCarmaker Integration Layer for EV Challenge Phase 2
    Handles communication between CarMaker, FMUs, and Fetch.ai agents via Kuksa
    """

    def __init__(self, kuksa_host="localhost", kuksa_port=55555):
        self.kuksa_host = kuksa_host
        self.kuksa_port = kuksa_port
        self.client = None
        self.running = False

        # Data queues for communication
        self.carmaker_data_queue = queue.Queue()
        self.charging_data_queue = queue.Queue()
        self.agent_data_queue = queue.Queue()

        # Current vehicle state
        self.vehicle_state = {
            "act_vel": 0.0,
            "brk_pedal": 0.0,
            "distance_covered": 0.0,
            "slope": 0.0,
            "soc": 1.0,
            "rolling_resistance": 0.008,
            "air_density": 1.225
        }

        # Charging decision state
        self.charging_state = {
            "charging_request": False,
            "reachable_distance": 0.0,
            "reachable_stations": [],
            "target_soc": 0.8,
            "selected_station_id": 0,
            "charging_available": False
        }

        # Motor control state
        self.motor_state = {
            "motor_trq_target": 0.0
        }

        # VSS signal paths for Kuksa
        self.vss_paths = {
            # Vehicle inputs
            "vehicle.speed": "Vehicle.Speed",
            "vehicle.brake_pedal": "Vehicle.Chassis.Brake.PedalPosition",
            "vehicle.distance": "Vehicle.TraveledDistance",
            "vehicle.slope": "Vehicle.Slope",
            "vehicle.soc": "Vehicle.Powertrain.TractionBattery.StateOfCharge.Current",

            # Charging outputs
            "charging.request": "Vehicle.Charging.Request",
            "charging.reachable_distance": "Vehicle.Charging.ReachableDistance",
            "charging.target_soc": "Vehicle.Charging.TargetSoC",
            "charging.selected_station": "Vehicle.Charging.SelectedStation",

            # Motor control
            "motor.torque_target": "Vehicle.Powertrain.ElectricMotor.TorqueTarget",

            # Agent communication
            "agent.available_slots": "Vehicle.Charging.AvailableSlots",
            "agent.slot_allocation": "Vehicle.Charging.SlotAllocation"
        }

        # Initialize FMU integration
        self.fmu_integration = None

    async def initialize(self):
        """Initialize Kuksa client and connections"""
        try:
            self.client = VSSClient(self.kuksa_host, self.kuksa_port)
            self.client.connect()
            logger.info(f"Connected to Kuksa DataBroker at {self.kuksa_host}:{self.kuksa_port}")

            # Set up initial VSS signals
            await self.setup_vss_signals()

            return True
        except Exception as e:
            logger.error(f"Failed to initialize Kuksa client: {e}")
            return False

    async def setup_vss_signals(self):
        """Set up VSS signal paths in Kuksa"""
        logger.info("Setting up VSS signals for EV Challenge Phase 2")

        initial_signals = {
            "Vehicle.Speed": 0.0,
            "Vehicle.Chassis.Brake.PedalPosition": 0.0,
            "Vehicle.TraveledDistance": 0.0,
            "Vehicle.Slope": 0.0,
            "Vehicle.Powertrain.TractionBattery.StateOfCharge.Current": 100.0,
            "Vehicle.Charging.Request": False,
            "Vehicle.Charging.ReachableDistance": 0.0,
            "Vehicle.Charging.TargetSoC": 80.0,
            "Vehicle.Charging.SelectedStation": 0,
            "Vehicle.Powertrain.ElectricMotor.TorqueTarget": 0.0
        }

        for path, value in initial_signals.items():
            try:
                if self.client is not None:
                    self.client.set_current_values({path: Datapoint(value)})
            except Exception as e:
                logger.warning(f"Could not set initial value for {path}: {e}")

    async def start(self):
        """Start the PyCarmaker integration"""
        if not await self.initialize():
            return False

        self.running = True

        # Start communication threads
        data_thread = threading.Thread(target=self.data_processing_loop)
        kuksa_pub_thread = threading.Thread(target=asyncio.run, args=(self.kuksa_publisher_loop(),))
        kuksa_sub_thread = threading.Thread(target=asyncio.run, args=(self.kuksa_subscriber_loop(),))

        data_thread.start()
        kuksa_pub_thread.start()
        kuksa_sub_thread.start()

        logger.info("PyCarmaker integration started")
        return True

    def data_processing_loop(self):
        """Main data processing loop"""
        while self.running:
            try:
                # Process CarMaker data
                self.process_carmaker_data()

                # Process charging decisions
                self.process_charging_decisions()

                # Process agent communications
                self.process_agent_communications()

                time.sleep(0.1)  # 10Hz processing rate

            except Exception as e:
                logger.error(f"Error in data processing loop: {e}")

    def process_carmaker_data(self):
        """Process data from CarMaker simulation"""
        try:
            # In a real implementation, this would read from CarMaker
            # For now, we'll simulate some data updates

            # Update vehicle state (this would come from CarMaker)
            # self.vehicle_state["act_vel"] = get_from_carmaker("velocity")
            # self.vehicle_state["soc"] = get_from_carmaker("soc")
            # etc.

            # Put data in queue for FMU processing
            self.carmaker_data_queue.put(self.vehicle_state.copy())

        except Exception as e:
            logger.error(f"Error processing CarMaker data: {e}")

    def process_charging_decisions(self):
        """Process charging decisions from FMU"""
        try:
            # This would integrate with your ChargingDecisionController FMU
            # For now, we'll simulate the FMU processing

            if not self.carmaker_data_queue.empty():
                vehicle_data = self.carmaker_data_queue.get()

                # Simulate FMU processing
                charging_decision = self.simulate_charging_fmu(vehicle_data)

                # Update charging state
                self.charging_state.update(charging_decision)

                # Put charging decision in queue for Kuksa publishing
                self.charging_data_queue.put(charging_decision)

        except Exception as e:
            logger.error(f"Error processing charging decisions: {e}")

    def simulate_charging_fmu(self, vehicle_data):
        """Simulate the ChargingDecisionController FMU"""
        # This is a simplified simulation of your FMU logic
        soc = vehicle_data.get("soc", 1.0)
        distance_covered = vehicle_data.get("distance_covered", 0.0)

        # Simple charging logic
        charging_needed = soc < 0.3  # Charge if SoC below 30%
        reachable_distance = soc * 60 * 1000 / 0.15  # Simplified range calculation

        # Charging stations (from Phase 2 data)
        stations = [
            {"id": 1, "distance": 2.38, "available": True},
            {"id": 2, "distance": 12.62, "available": True},
            {"id": 3, "distance": 24.30, "available": True},
            {"id": 4, "distance": 55.37, "available": False},
            {"id": 5, "distance": 95.11, "available": True}
        ]

        # Find reachable stations
        reachable_stations = [s for s in stations if s["distance"] * 1000 <= reachable_distance and s["available"]]

        # Select closest station if charging needed
        selected_station = 0
        if charging_needed and reachable_stations:
            selected_station = min(reachable_stations, key=lambda x: x["distance"])["id"]

        return {
            "charging_request": charging_needed,
            "reachable_distance": reachable_distance,
            "reachable_stations": reachable_stations,
            "target_soc": 0.8,
            "selected_station_id": selected_station
        }

    def process_agent_communications(self):
        """Process communications with Fetch.ai agents"""
        try:
            # This would handle communication with Fetch.ai agents
            # For now, we'll simulate agent responses

            if not self.agent_data_queue.empty():
                agent_data = self.agent_data_queue.get()

                # Process agent response
                if agent_data.get("type") == "slot_allocation":
                    self.charging_state["charging_available"] = agent_data.get("available", False)
                    self.charging_state["allocated_station"] = agent_data.get("station_id", 0)

        except Exception as e:
            logger.error(f"Error processing agent communications: {e}")

    async def kuksa_publisher_loop(self):
        """Kuksa publisher loop for sending data to agents"""
        while self.running:
            try:
                # Publish charging decisions
                if not self.charging_data_queue.empty():
                    charging_data = self.charging_data_queue.get()
                    await self.publish_charging_data(charging_data)

                # Publish motor control commands
                await self.publish_motor_data()

                await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in Kuksa publisher loop: {e}")
                await asyncio.sleep(1)

    async def publish_charging_data(self, charging_data):
        """Publish charging data to Kuksa"""
        try:
            updates = {
                "Vehicle.Charging.Request": charging_data.get("charging_request", False),
                "Vehicle.Charging.ReachableDistance": charging_data.get("reachable_distance", 0.0),
                "Vehicle.Charging.TargetSoC": charging_data.get("target_soc", 0.8),
                "Vehicle.Charging.SelectedStation": charging_data.get("selected_station_id", 0)
            }

            if self.client is not None:
                datapoints = {path: Datapoint(value) for path, value in updates.items()}
                self.client.set_current_values(datapoints)

            logger.info(f"Published charging data: {charging_data}")

        except Exception as e:
            logger.error(f"Error publishing charging data: {e}")

    async def publish_motor_data(self):
        """Publish motor control data to Kuksa"""
        try:
            motor_data = {
                "Vehicle.Powertrain.ElectricMotor.TorqueTarget": self.motor_state["motor_trq_target"]
            }

            if self.client is not None:
                datapoints = {path: Datapoint(value) for path, value in motor_data.items()}
                self.client.set_current_values(datapoints)

        except Exception as e:
            logger.error(f"Error publishing motor data: {e}")

    async def kuksa_subscriber_loop(self):
        """Kuksa subscriber loop for receiving data from agents"""
        while self.running:
            try:
                agent_signals = [
                    "Vehicle.Charging.AvailableSlots",
                    "Vehicle.Charging.SlotAllocation"
                ]

                for signal in agent_signals:
                    try:
                        if self.client is not None:
                            response = self.client.get_current_values([signal])
                            if response and signal in response:
                                self.handle_agent_response(signal, response[signal])
                    except Exception as e:
                        logger.debug(f"No data for signal {signal}: {e}")

                await asyncio.sleep(0.5)

            except Exception as e:
                logger.error(f"Error in Kuksa subscriber loop: {e}")
                await asyncio.sleep(1)

    def handle_agent_response(self, signal, datapoint):
        """Handle responses from Fetch.ai agents"""
        try:
            if signal == "Vehicle.Charging.SlotAllocation":
                # Handle slot allocation from charge agent
                allocation_data = {
                    "type": "slot_allocation",
                    "available": datapoint.value if hasattr(datapoint, 'value') else False,
                    "station_id": 1  # This would come from the actual agent response
                }
                self.agent_data_queue.put(allocation_data)

            elif signal == "Vehicle.Charging.AvailableSlots":
                # Handle available slots information
                slots_data = {
                    "type": "available_slots",
                    "slots": datapoint.value if hasattr(datapoint, 'value') else []
                }
                self.agent_data_queue.put(slots_data)

        except Exception as e:
            logger.error(f"Error handling agent response: {e}")

    def update_vehicle_state(self, **kwargs):
        """Update vehicle state from external source (CarMaker)"""
        self.vehicle_state.update(kwargs)

    def update_motor_torque(self, torque):
        """Update motor torque command"""
        self.motor_state["motor_trq_target"] = torque

    def get_charging_state(self):
        """Get current charging state"""
        return self.charging_state.copy()

    def get_vehicle_state(self):
        """Get current vehicle state"""
        return self.vehicle_state.copy()

    async def stop(self):
        """Stop the PyCarmaker integration"""
        self.running = False
        if self.client is not None:
            self.client.disconnect()
        logger.info("PyCarmaker integration stopped")


# Example usage and testing
if __name__ == "__main__":
    async def main():
        # Initialize PyCarmaker integration
        integration = PyCarmakerIntegration()

        # Start the integration
        if await integration.start():
            logger.info("PyCarmaker integration running...")

            # Simulate some vehicle data updates
            for i in range(100):
                integration.update_vehicle_state(
                    act_vel=15.0 + i * 0.1,
                    soc=1.0 - i * 0.01,
                    distance_covered=i * 100
                )
                await asyncio.sleep(1)

            # Stop the integration
            await integration.stop()
        else:
            logger.error("Failed to start PyCarmaker integration")


    # Run the example
    asyncio.run(main())