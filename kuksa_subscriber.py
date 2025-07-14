import asyncio
import json
import logging
import time
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass
from kuksa_client.grpc import VSSClient, Datapoint
import threading
import queue
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SubscriptionData:
    """Data structure for subscription updates"""
    signal_path: str
    value: Any
    timestamp: float
    quality: str = "good"


class KuksaSubscriber:
    """
    Kuksa Data Subscriber for EV Challenge Phase 2
    Handles subscribing to data from Kuksa DataBroker and forwarding to agents
    """

    def __init__(self, host: str = "localhost", port: int = 55555):
        self.host = host
        self.port = port
        self.client = None
        self.connected = False
        self.subscriptions = {}  # signal_path -> callback
        self.subscription_data = {}  # signal_path -> latest_value
        self.is_running = False
        self.subscription_thread = None

        # Callback registry
        self.callbacks = {}

        # Performance metrics
        self.total_received = 0
        self.callback_errors = 0

    def connect(self):
        """Connect to Kuksa DataBroker"""
        try:
            self.client = VSSClient(self.host, self.port)
            self.client.connect()
            self.connected = True
            logger.info(f"Connected to Kuksa DataBroker at {self.host}:{self.port}")
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

    def register_callback(self, signal_path: str, callback: Callable[[str, Any], None]):
        """Register a callback for a specific signal"""
        if signal_path not in self.callbacks:
            self.callbacks[signal_path] = []
        self.callbacks[signal_path].append(callback)
        logger.info(f"Registered callback for signal: {signal_path}")

    def unregister_callback(self, signal_path: str, callback: Callable[[str, Any], None]):
        """Unregister a callback for a specific signal"""
        if signal_path in self.callbacks:
            if callback in self.callbacks[signal_path]:
                self.callbacks[signal_path].remove(callback)
                if not self.callbacks[signal_path]:
                    del self.callbacks[signal_path]
                logger.info(f"Unregistered callback for signal: {signal_path}")

    def subscribe_to_signal(self, signal_path: str, callback: Optional[Callable[[str, Any], None]] = None):
        """Subscribe to a specific signal"""
        if not self.connected:
            logger.warning("Not connected to Kuksa DataBroker")
            return False

        try:
            # Register callback if provided
            if callback:
                self.register_callback(signal_path, callback)

            # Subscribe to signal
            self.subscriptions[signal_path] = True
            logger.info(f"Subscribed to signal: {signal_path}")
            return True

        except Exception as e:
            logger.error(f"Error subscribing to signal {signal_path}: {e}")
            return False

    def subscribe_to_multiple_signals(self, signal_paths: List[str],
                                      callback: Optional[Callable[[str, Any], None]] = None):
        """Subscribe to multiple signals"""
        success_count = 0
        for signal_path in signal_paths:
            if self.subscribe_to_signal(signal_path, callback):
                success_count += 1

        logger.info(f"Subscribed to {success_count}/{len(signal_paths)} signals")
        return success_count == len(signal_paths)

    def get_current_values(self, signal_paths: List[str]) -> Dict[str, Any]:
        """Get current values for specified signals"""
        if not self.connected:
            return {}

        try:
            if self.client is not None:
                response = self.client.get_current_values(signal_paths)
            else:
                response = {}

            result = {}
            for signal_path, datapoint in response.items():
                result[signal_path] = datapoint.value

            return result

        except Exception as e:
            logger.error(f"Error getting current values: {e}")
            return {}

    def subscription_worker(self):
        """Background worker for handling subscriptions"""
        while self.is_running:
            try:
                if not self.connected:
                    time.sleep(1)
                    continue

                # Get current values for all subscribed signals
                subscribed_signals = list(self.subscriptions.keys())
                if not subscribed_signals:
                    time.sleep(0.1)
                    continue

                current_values = self.get_current_values(subscribed_signals)

                # Process updates
                for signal_path, value in current_values.items():
                    self.process_signal_update(signal_path, value)

                # Sleep to prevent excessive polling
                time.sleep(0.1)  # 10Hz polling rate

            except Exception as e:
                logger.error(f"Error in subscription worker: {e}")
                time.sleep(1)

    def process_signal_update(self, signal_path: str, value: Any):
        """Process a signal update"""
        try:
            self.subscription_data[signal_path] = {
                "value": value,
                "timestamp": time.time()
            }

            self.total_received += 1

            if signal_path in self.callbacks:
                for callback in self.callbacks[signal_path]:
                    try:
                        callback(signal_path, value)
                    except Exception as e:
                        logger.error(f"Error in callback for {signal_path}: {e}")
                        self.callback_errors += 1

        except Exception as e:
            logger.error(f"Error processing signal update for {signal_path}: {e}")

    def start_subscription_worker(self):
        """Start the subscription worker"""
        if not self.is_running:
            self.is_running = True
            # Start the async event loop in a separate thread
            self.subscription_thread = threading.Thread(target=self._run_async_loop)
            self.subscription_thread.daemon = True
            self.subscription_thread.start()
            logger.info("Started Kuksa subscription worker")

    def stop_subscription_worker(self):
        """Stop the subscription worker"""
        if self.is_running:
            self.is_running = False
            if self.subscription_thread:
                self.subscription_thread.join(timeout=5)
            logger.info("Stopped Kuksa subscription worker")

    def _run_async_loop(self):
        """Run the async event loop in a separate thread"""
        self.subscription_worker()

    def get_latest_value(self, signal_path: str) -> Optional[Any]:
        """Get the latest value for a signal"""
        if signal_path in self.subscription_data:
            return self.subscription_data[signal_path]["value"]
        return None

    def get_all_latest_values(self) -> Dict[str, Any]:
        """Get all latest values"""
        result = {}
        for signal_path, data in self.subscription_data.items():
            result[signal_path] = data["value"]
        return result

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get subscription performance metrics"""
        return {
            "total_received": self.total_received,
            "callback_errors": self.callback_errors,
            "active_subscriptions": len(self.subscriptions),
            "connected": self.connected,
            "latest_signals": len(self.subscription_data)
        }


class EVChallengeSub:
    """
    EV Challenge specific subscriber with predefined signal patterns
    """

    def __init__(self, host: str = "localhost", port: int = 55555):
        self.subscriber = KuksaSubscriber(host, port)
        self.vehicle_data_callback = None
        self.charging_station_callback = None
        self.charging_decision_callback = None

        # EV Challenge signal patterns
        self.vehicle_signals = [
            "Vehicle.Speed",
            "Vehicle.Powertrain.TractionBattery.StateOfCharge.Current",
            "Vehicle.TraveledDistance",
            "Vehicle.Charging.ReachableDistance",
            "Vehicle.Charging.Request",
            "Vehicle.Charging.TargetSoC"
        ]

        self.charging_station_signals = []
        for i in range(1, 6):  # Stations 1-5
            self.charging_station_signals.extend([
                f"Vehicle.Charging.Station.{i}.AvailableSlots",
                f"Vehicle.Charging.Station.{i}.TotalSlots",
                f"Vehicle.Charging.Station.{i}.QueueLength"
            ])

        self.charging_decision_signals = [
            "Vehicle.Charging.Decision.ShouldCharge",
            "Vehicle.Charging.Decision.SelectedStation",
            "Vehicle.Charging.Decision.Priority",
            "Vehicle.Charging.Decision.EstimatedTime",
            "Vehicle.Charging.Decision.RouteFeasible"
        ]

    def connect(self):
        """Connect to Kuksa DataBroker"""
        return self.subscriber.connect()

    def disconnect(self):
        """Disconnect from Kuksa DataBroker"""
        self.subscriber.disconnect()

    def subscribe_to_vehicle_data(self, callback: Callable[[Dict[str, Any]], None]):
        """Subscribe to vehicle data signals"""
        self.vehicle_data_callback = callback

        def vehicle_signal_callback(signal_path: str, value: Any):
            # Collect all vehicle data
            vehicle_data = {}
            for signal in self.vehicle_signals:
                latest_value = self.subscriber.get_latest_value(signal)
                if latest_value is not None:
                    vehicle_data[signal] = latest_value

            # Call user callback with aggregated data
            if self.vehicle_data_callback:
                self.vehicle_data_callback(vehicle_data)

        success = self.subscriber.subscribe_to_multiple_signals(
            self.vehicle_signals, vehicle_signal_callback
        )

        return success

    def subscribe_to_charging_stations(self, callback: Callable[[Dict[str, Any]], None]):
        """Subscribe to charging station signals"""
        self.charging_station_callback = callback

        def station_signal_callback(signal_path: str, value: Any):
            # Collect all station data
            station_data = {}
            for signal in self.charging_station_signals:
                latest_value = self.subscriber.get_latest_value(signal)
                if latest_value is not None:
                    station_data[signal] = latest_value

            # Call user callback with aggregated data
            if self.charging_station_callback:
                self.charging_station_callback(station_data)

        success = self.subscriber.subscribe_to_multiple_signals(
            self.charging_station_signals, station_signal_callback
        )

        return success

    def subscribe_to_charging_decisions(self, callback: Callable[[Dict[str, Any]], None]):
        """Subscribe to charging decision signals"""
        self.charging_decision_callback = callback

        def decision_signal_callback(signal_path: str, value: Any):
            # Collect all decision data
            decision_data = {}
            for signal in self.charging_decision_signals:
                latest_value = self.subscriber.get_latest_value(signal)
                if latest_value is not None:
                    decision_data[signal] = latest_value

            # Call user callback with aggregated data
            if self.charging_decision_callback:
                self.charging_decision_callback(decision_data)

        success = self.subscriber.subscribe_to_multiple_signals(
            self.charging_decision_signals, decision_signal_callback
        )

        return success

    def start_subscription_worker(self):
        """Start the subscription worker"""
        self.subscriber.start_subscription_worker()

    def stop_subscription_worker(self):
        """Stop the subscription worker"""
        self.subscriber.stop_subscription_worker()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self.subscriber.get_performance_metrics()


# Example usage
def main():
    ev_subscriber = EVChallengeSub()

    # Define callbacks
    def vehicle_data_handler(data):
        print(f"Vehicle data update: {data}")

    def station_data_handler(data):
        print(f"Station data update: {data}")

    def decision_data_handler(data):
        print(f"Decision data update: {data}")

    # Connect and subscribe
    if ev_subscriber.connect():
        ev_subscriber.subscribe_to_vehicle_data(vehicle_data_handler)
        ev_subscriber.subscribe_to_charging_stations(station_data_handler)
        ev_subscriber.subscribe_to_charging_decisions(decision_data_handler)

        # Start subscription worker
        ev_subscriber.start_subscription_worker()

        # Let it run for a bit
        time.sleep(10)

        # Stop and disconnect
        ev_subscriber.stop_subscription_worker()
        ev_subscriber.disconnect()

    # Print performance metrics
    metrics = ev_subscriber.get_performance_metrics()
    print(f"Performance metrics: {metrics}")


if __name__ == "__main__":
    main()