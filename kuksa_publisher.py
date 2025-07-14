import asyncio
import json
import logging
import time
from typing import Dict, Any, List
from dataclasses import dataclass
from kuksa_client.grpc import VSSClient, Datapoint
import threading
import queue
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataUpdate:
    """Data update structure for Kuksa publishing"""
    signal_path: str
    value: Any
    timestamp: float
    source: str


class KuksaPublisher:
    """
    Kuksa Data Publisher for EV Challenge Phase 2
    Handles publishing data from FMUs and agents to Kuksa DataBroker
    """

    def __init__(self, host: str = "localhost", port: int = 55555):
        self.host = host
        self.port = port
        self.client = None
        self.connected = False
        self.data_queue = queue.Queue()
        self.is_running = False
        self.publish_thread = None

        # Signal definitions for EV Challenge
        self.signal_definitions = {
            # Vehicle state signals
            "Vehicle.Speed": "m/s",
            "Vehicle.Powertrain.TractionBattery.StateOfCharge.Current": "%",
            "Vehicle.TraveledDistance": "m",
            "Vehicle.Charging.ReachableDistance": "m",
            "Vehicle.Charging.Request": "boolean",
            "Vehicle.Charging.TargetSoC": "%",

            # Charging station signals
            "Vehicle.Charging.Station.1.AvailableSlots": "count",
            "Vehicle.Charging.Station.1.TotalSlots": "count",
            "Vehicle.Charging.Station.1.QueueLength": "count",
            "Vehicle.Charging.Station.2.AvailableSlots": "count",
            "Vehicle.Charging.Station.2.TotalSlots": "count",
            "Vehicle.Charging.Station.2.QueueLength": "count",
            "Vehicle.Charging.Station.3.AvailableSlots": "count",
            "Vehicle.Charging.Station.3.TotalSlots": "count",
            "Vehicle.Charging.Station.3.QueueLength": "count",
            "Vehicle.Charging.Station.4.AvailableSlots": "count",
            "Vehicle.Charging.Station.4.TotalSlots": "count",
            "Vehicle.Charging.Station.4.QueueLength": "count",
            "Vehicle.Charging.Station.5.AvailableSlots": "count",
            "Vehicle.Charging.Station.5.TotalSlots": "count",
            "Vehicle.Charging.Station.5.QueueLength": "count",

            # Charging decision signals
            "Vehicle.Charging.Decision.ShouldCharge": "boolean",
            "Vehicle.Charging.Decision.SelectedStation": "id",
            "Vehicle.Charging.Decision.Priority": "level",
            "Vehicle.Charging.Decision.EstimatedTime": "minutes",
            "Vehicle.Charging.Decision.RouteFeasible": "boolean",

            # Agent status signals
            "Vehicle.Agent.Car.Status": "string",
            "Vehicle.Agent.Charge.Status": "string",
            "Vehicle.Agent.Car.Performance.SuccessRate": "%",
            "Vehicle.Agent.Charge.Performance.SuccessRate": "%"
        }

        # Performance metrics
        self.total_published = 0
        self.failed_publishes = 0

    def connect(self):
        """Connect to Kuksa DataBroker"""
        try:
            self.client = VSSClient(self.host, self.port)
            self.client.connect()
            self.connected = True
            logger.info(f"Connected to Kuksa DataBroker at {self.host}:{self.port}")

            # Initialize signal definitions
            self.initialize_signals()

            return True
        except Exception as e:
            logger.error(f"Failed to connect to Kuksa DataBroker: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """Disconnect from Kuksa DataBroker"""
        if self.client and self.connected:
            self.client.disconnect()
            self.connected = False
            logger.info("Disconnected from Kuksa DataBroker")

    def initialize_signals(self):
        """Initialize signal definitions in Kuksa"""
        try:
            logger.info(f"Initialized {len(self.signal_definitions)} signal definitions")
        except Exception as e:
            logger.error(f"Error initializing signals: {e}")

    def publish_data(self, signal_path: str, value: Any, source: str = "FMU"):
        """Queue data for publishing to Kuksa"""
        if not self.connected:
            logger.warning("Not connected to Kuksa DataBroker")
            return False

        update = DataUpdate(
            signal_path=signal_path,
            value=value,
            timestamp=time.time(),
            source=source
        )

        self.data_queue.put(update)
        return True

    def publish_batch(self, data_batch: List[DataUpdate]):
        """Publish a batch of data updates"""
        if not self.connected:
            return False

        try:
            datapoints = {}
            for update in data_batch:
                datapoints[update.signal_path] = Datapoint(update.value)

            if self.client is not None:
                self.client.set_current_values(datapoints)

            self.total_published += len(data_batch)
            logger.debug(f"Published {len(data_batch)} signals to Kuksa")
            return True

        except Exception as e:
            logger.error(f"Error publishing batch to Kuksa: {e}")
            self.failed_publishes += len(data_batch)
            return False

    def publish_worker(self):
        """Background worker for publishing data"""
        batch_size = 10
        batch_timeout = 0.1  # 100ms

        while self.is_running:
            try:
                batch = []
                start_time = time.time()

                # Collect batch of updates
                while len(batch) < batch_size and (time.time() - start_time) < batch_timeout:
                    try:
                        update = self.data_queue.get(timeout=0.05)
                        batch.append(update)
                    except queue.Empty:
                        break

                # Publish batch if we have data
                if batch:
                    self.publish_batch(batch)

                # Small delay to prevent CPU spinning
                time.sleep(0.01)

            except Exception as e:
                logger.error(f"Error in publish worker: {e}")
                time.sleep(1)

    def start_publishing(self):
        """Start the publishing thread"""
        if not self.is_running:
            self.is_running = True
            # Start the async event loop in a separate thread
            self.publish_thread = threading.Thread(target=self._run_async_loop)
            self.publish_thread.daemon = True
            self.publish_thread.start()
            logger.info("Started Kuksa publishing worker")

    def stop_publishing(self):
        """Stop the publishing thread"""
        if self.is_running:
            self.is_running = False
            if self.publish_thread:
                self.publish_thread.join(timeout=5)
            logger.info("Stopped Kuksa publishing worker")

    def _run_async_loop(self):
        """Run the async event loop in a separate thread"""
        self.publish_worker()

    def publish_vehicle_state(self, vehicle_data: Dict[str, Any]):
        """Publish vehicle state data"""
        signal_mappings = {
            "speed": "Vehicle.Speed",
            "soc": "Vehicle.Powertrain.TractionBattery.StateOfCharge.Current",
            "distance": "Vehicle.TraveledDistance",
            "reachable_distance": "Vehicle.Charging.ReachableDistance",
            "charging_request": "Vehicle.Charging.Request",
            "target_soc": "Vehicle.Charging.TargetSoC"
        }

        for key, signal_path in signal_mappings.items():
            if key in vehicle_data:
                self.publish_data(signal_path, vehicle_data[key], "CarMaker")

    def publish_charging_stations(self, station_data: Dict[int, Dict[str, Any]]):
        """Publish charging station data"""
        for station_id, station_info in station_data.items():
            base_path = f"Vehicle.Charging.Station.{station_id}"

            self.publish_data(f"{base_path}.AvailableSlots",
                              station_info.get("available_slots", 0), "ChargeAgent")
            self.publish_data(f"{base_path}.TotalSlots",
                              station_info.get("total_slots", 0), "ChargeAgent")
            self.publish_data(f"{base_path}.QueueLength",
                              station_info.get("queue_length", 0), "ChargeAgent")

    def publish_charging_decision(self, decision_data: Dict[str, Any]):
        """Publish charging decision data"""
        signal_mappings = {
            "should_charge": "Vehicle.Charging.Decision.ShouldCharge",
            "selected_station": "Vehicle.Charging.Decision.SelectedStation",
            "priority": "Vehicle.Charging.Decision.Priority",
            "estimated_time": "Vehicle.Charging.Decision.EstimatedTime",
            "route_feasible": "Vehicle.Charging.Decision.RouteFeasible"
        }

        for key, signal_path in signal_mappings.items():
            if key in decision_data:
                self.publish_data(signal_path, decision_data[key], "CarAgent")

    def publish_agent_status(self, agent_name: str, status_data: Dict[str, Any]):
        """Publish agent status data"""
        base_path = f"Vehicle.Agent.{agent_name}"

        self.publish_data(f"{base_path}.Status",
                          status_data.get("status", "unknown"), f"{agent_name}Agent")

        if "performance" in status_data:
            perf = status_data["performance"]
            self.publish_data(f"{base_path}.Performance.SuccessRate",
                              perf.get("success_rate", 0.0), f"{agent_name}Agent")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get publishing performance metrics"""
        total_attempts = self.total_published + self.failed_publishes
        success_rate = self.total_published / total_attempts if total_attempts > 0 else 0

        return {
            "total_published": self.total_published,
            "failed_publishes": self.failed_publishes,
            "success_rate": success_rate,
            "queue_size": self.data_queue.qsize(),
            "connected": self.connected
        }


# Example usage
async def main():
    publisher = KuksaPublisher()

    # Connect to Kuksa
    if publisher.connect():
        # Start publishing worker
        publisher.start_publishing()

        # Example data publishing
        vehicle_data = {
            "speed": 15.5,
            "soc": 45.2,
            "distance": 15000,
            "reachable_distance": 25000,
            "charging_request": True,
            "target_soc": 80.0
        }

        publisher.publish_vehicle_state(vehicle_data)

        # Let it run for a bit
        await asyncio.sleep(2)

        # Stop and disconnect
        publisher.stop_publishing()
        publisher.disconnect()

    # Print performance metrics
    metrics = publisher.get_performance_metrics()
    print(f"Performance metrics: {metrics}")


if __name__ == "__main__":
    asyncio.run(main())