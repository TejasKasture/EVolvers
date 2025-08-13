from kuksa_client.grpc import VSSClient, Field, SubscribeEntry, View
import time
import signal
import sys
import json

class CarMakerSubscriber:
    def __init__(self, host: str = "127.0.0.1", port: int = 55555, json_file: str = "kuksa.json"):
        self.host = host
        self.port = port
        self.json_file = json_file
        self.client = None
        self.is_connected = False
        self.running = True

    def connect(self):
        """Connect to Kuksa Data Broker"""
        try:
            self.client = VSSClient(self.host, self.port)
            self.client.connect()
            self.is_connected = True
            print(f"Subscriber connected to KUKSA at {self.host}:{self.port}")
        except Exception as e:
            print(f"Failed to connect subscriber: {e}")
            self.is_connected = False

    def write_data_to_json(self, data):
        """Write received data to JSON file"""
        try:
            json_data = {
                "station_id": int(data["Vehicle.Speed"].value),
                "slot_number": int(data["Vehicle.CurrentLocation.Latitude"].value),
                "last_update": time.time(),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }

            with open(self.json_file, 'w') as f:
                json.dump(json_data, f, indent=2)

            print(
                f"📄 Data written to {self.json_file}: Station ID: {json_data['station_id']}, Slot: {json_data['slot_number']}")
            return True
        except Exception as e:
            print(f"❌ Error writing to JSON file: {e}")
            return False

    def disconnect(self):
        """Disconnect from Kuksa Data Broker"""
        if self.client:
            self.client.disconnect()
            self.is_connected = False
            print("Subscriber disconnected")

    def subscribe_to_all(self):
        """Subscribe to vehicle signals and return their values"""
        if not self.is_connected:
            print("Subscriber not connected")
            return None

        # VSS paths for the signals published by the agent
        paths = [
            "Vehicle.Speed",  # station id
            "Vehicle.CurrentLocation.Latitude"  # slot number
        ]

        print(f"Subscribing to paths: {paths}")

        try:
            # Create subscription entries
            entries = [SubscribeEntry(path, View.FIELDS, (Field.VALUE,)) for path in paths]

            # Subscribe to all paths and listen for updates
            for updates in self.client.subscribe(entries=entries):
                if not self.running:
                    break

                current_values = {}
                for update in updates:
                    if update.entry.value is not None:
                        path = update.entry.path
                        value = update.entry.value
                        current_values[path] = value
                        print(f"Updated value for {path}: {value}")

                # Write to JSON file when we receive any updates
                if current_values:
                    self.write_data_to_json(current_values)
                    yield current_values

        except Exception as e:
            print(f"Error during subscription: {e}")

    def signal_handler(self, signum, frame):
        """Handle interrupt signal gracefully"""
        print("\nReceived interrupt signal. Stopping subscriber...")
        self.running = False
        self.disconnect()
        sys.exit(0)

    def start_continuous_subscription(self):
        """Start continuous subscription with proper error handling"""
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)

        while not self.is_connected and self.running:
            print("Attempting to connect...")
            self.connect()
            if not self.is_connected:
                time.sleep(2)

        if self.is_connected:
            print("Starting continuous subscription...")
            print(f"📄 Writing data to JSON file: {self.json_file}")
            try:
                for values_batch in self.subscribe_to_all():
                    if not self.running:
                        break
                    # Process the received values here
                    print(f"Received batch: {values_batch}")

            except KeyboardInterrupt:
                print("\nSubscription interrupted by user")
            finally:
                self.disconnect()


# Example usage
if __name__ == "__main__":
    # Option 1: Use the class-based approach (recommended)
    subscriber = CarMakerSubscriber()
    subscriber.start_continuous_subscription()

    # Option 2: Use the original function
    # subscribe_to_all()


"""
{
  "station_id": 3,     #take valid vss paths   #int
  "slot_number": 2,                             #int
  "last_update": 1721401956.123456,     #useless
  "timestamp": "2025-07-19 20:32:36"    #useless
}
"""