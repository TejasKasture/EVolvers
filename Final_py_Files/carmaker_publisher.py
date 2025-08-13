from kuksa_client.grpc import VSSClient, Datapoint
from typing import Dict, List


class KuksaPublisher:
    def __init__(self, host: str = "localhost", port: int = 55555):
        self.host = host
        self.port = port
        self.client = None
        self.is_connected = False

    def connect(self):
        try:
            self.client = VSSClient(host=self.host, port=self.port)
            self.client.connect()
            self.is_connected = True
            print(f"Publisher connected to Kuksa at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect publisher: {e}")
            self.is_connected = False
            return False

    def publish_value(self, vss_path: str, value):
        if not self.is_connected:
            print("Publisher not connected")
            return False
        try:
            datapoint = Datapoint(value=str(value))
            self.client.set_current_values({vss_path: datapoint})
            print(f"Published to {vss_path}: {value}")
            return True
        except Exception as e:
            print(f"Error publishing to {vss_path}: {e}")
            return False

    def publish_all_data_batch(self, charging_request: bool, reachable_stations: List[int],
                               slot_availability: Dict[str, List[int]]):
        if not self.is_connected:
            print("Publisher not connected")
            return False
        try:
            all_datapoints = {}

            # 1. Boolean as int
            all_datapoints["Vehicle.Cabin.HVAC.IsAirConditioningActive"] = Datapoint(value=str(int(charging_request)))

            # 2. String-encoded reachable stations to string-compatible field
            stations_str = ','.join(map(str, reachable_stations))
            all_datapoints["Vehicle.VehicleIdentification.VIN"] = Datapoint(value=stations_str)

            # 3. Slot availability encoded string to a "string-tolerant" field
            availability_segments = []
            for i in range(1, 6):  # CH1 to CH5
                key = f"CH{i}"
                slots = slot_availability.get(key, [0, 0, 0])
                slot_str = ''.join(map(str, slots))
                availability_segments.append(slot_str)
            availability_str = '_'.join(availability_segments)
            all_datapoints["Vehicle.VehicleIdentification.Model"] = Datapoint(value=availability_str)

            self.client.set_current_values(all_datapoints)
            print(
                f"✅ Published batch: Request={charging_request}, Stations={stations_str}, Availability={availability_str}")
            return True
        except Exception as e:
            print(f"❌ Error publishing batch data: {e}")
            return False

    def data(self, params: list):
        if len(params) < 3:
            print("Insufficient parameters. Need: [charging_request, reachable_stations, slot_availability]")
            return False
        charging_request = params[0]
        reachable_stations = params[1]
        slot_availability = params[2]
        return self.publish_all_data_batch(charging_request, reachable_stations, slot_availability)

    def main(self, params: list):
        """Main method that can be called from external code"""
        if not self.is_connected:
            if not self.connect():
                return False

        if self.is_connected:
            return self.data(params)
        return False

    def disconnect(self):
        if self.client:
            try:
                self.client.disconnect()
                self.is_connected = False
                print("Publisher disconnected")
            except Exception as e:
                print(f"Error disconnecting: {e}")


# Example usage
if __name__ == "__main__":
    publisher = KuksaPublisher()

    example_params = [
        True,
        [1, 2, 3, 4],
        {
            "CH1": [0, 1, 1],
            "CH2": [1, 1, 1],
            "CH3": [1, 0, 1],
            "CH4": [0, 0, 1],
            "CH5": [1, 1, 0]
        }
    ]

    publisher.main(example_params)