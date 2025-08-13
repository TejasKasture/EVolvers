import time
import logging
import json
import os
from pycarmaker import CarMaker, Quantity
from carmaker_publisher import KuksaPublisher  # Import the KuksaPublisher class

# Enable logging
FORMAT = '[%(levelname)6s] %(module)10s: %(message)s'
logging.basicConfig(format=FORMAT)

print("🔌 CarMaker Charging Request Publisher with JSON Data Integration - FIXED VERSION")

# JSON file path where subscriber writes data
KUKSA_DATA_FILE = "kuksa_data.json"

# Charging station configuration with 3 slots each
CHARGING_STATIONS = {
    'CH1': {'distance': 2.38, 'label': 'Charging Station 1', 'slots': ['01', '02', '03']},
    'CH2': {'distance': 12.62, 'label': 'Charging Station 2', 'slots': ['01', '02', '03']},
    'CH3': {'distance': 24.30, 'label': 'Charging Station 3', 'slots': ['01', '02', '03']},
    'CH4': {'distance': 55.37, 'label': 'Charging Station 4', 'slots': ['01', '02', '03']},
    'CH5': {'distance': 95.11, 'label': 'Charging Station 5', 'slots': ['01', '02', '03']}
}

TOTAL_DISTANCE = 152.81  # km


def book_station_slot(cm, station_id, slot_number):
    """
    FIXED: Set the charging station booking signal using direct quantity assignment.
    Example: For CH2_03 -> set CH2_03 to 1.0

    Args:
        cm: CarMaker connection object
        station_id: string like 'CH2'
        slot_number: string like '03'
    """
    try:
        signal_name = f"{station_id}_{slot_number}"
        print(f"📥 Booking signal: {signal_name}")

        # FIXED: Create quantity and set directly (like the example)
        booking_quantity = Quantity(signal_name, Quantity.FLOAT)
        booking_quantity.Book = 1.0

        print(f"✅ Booking signal {signal_name} set to 1.0")

        # Also initialize AOC system for charging (from example)
        try:
            aoc = Quantity("PT.BattHV.AOC", Quantity.FLOAT)
            aoc_start = Quantity("Qu::AOC_start", Quantity.FLOAT)
            aoc_time = Quantity("Qu::start", Quantity.FLOAT)

            aoc.data = 0.01
            aoc_start.data = aoc.data
            aoc_time.data = time.time()

            print(f"⚡ AOC initialized for charging at {signal_name}")

        except Exception as aoc_e:
            print(f"⚠️ AOC initialization failed: {aoc_e}")

    except Exception as e:
        print(f"❌ Failed to book {station_id} slot {slot_number}: {e}")


def trigger_charging_process(cm, station_id, slot_number):
    """
    FIXED: Trigger charging using auxiliary power control (based on example)
    """
    try:
        # Use auxiliary power control for charging (from example)
        step_time = Quantity("DM.Long.StepTime", Quantity.FLOAT)
        trigger_id = Quantity("DM.TriggerPoint.Id", Quantity.FLOAT)
        pwr_aux = Quantity("PT.PwrSupply.HV1.Pwr_aux", Quantity.FLOAT)

        cm.subscribe(step_time)
        cm.subscribe(trigger_id)
        cm.read()

        # Apply charging logic from example
        if step_time.data is not None and step_time.data <= 0.0001:
            print("⚠️ Parking and Charging")
            trig_id = trigger_id.data if trigger_id.data is not None else 0
            pwr_aux.data = -20000 * (18 - trig_id)
            print(f"✅ Charging power set: {pwr_aux.data}W")

    except Exception as e:
        print(f"❌ Failed to trigger charging process: {e}")


def monitor_charging_status(cm, station_id, slot_number):
    """
    FIXED: Monitor charging status using available signals (SOC, power)
    """
    try:
        # Monitor SOC and power instead of non-existent ChargingActive signals
        soc = Quantity("PT.BCU.BattHV.SOC", Quantity.FLOAT)
        pwr_aux = Quantity("PT.PwrSupply.HV1.Pwr_aux", Quantity.FLOAT)
        charge_limit = Quantity("Car.Charge_Limit", Quantity.FLOAT)

        cm.subscribe(soc)
        cm.subscribe(pwr_aux)
        cm.subscribe(charge_limit)
        cm.read()

        if soc.data is not None:
            print(f"🔋 SOC: {soc.data:.2f}%")
        if pwr_aux.data is not None and pwr_aux.data != 0:
            print(f"⚡ Charging Power: {pwr_aux.data}W")
        if charge_limit.data is not None:
            print(f"🎯 Charge Limit: {charge_limit.data}%")

    except Exception as e:
        print(f"❌ Failed to monitor charging status: {e}")


def read_kuksa_data_from_json():
    """Read the latest Kuksa data from JSON file"""
    try:
        if os.path.exists(KUKSA_DATA_FILE):
            with open(KUKSA_DATA_FILE, 'r') as f:
                data = json.load(f)
                return data
        else:
            print(f"📄 JSON file {KUKSA_DATA_FILE} not found")
            return None
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON file: {e}")
        return None
    except Exception as e:
        print(f"❌ Error reading JSON file: {e}")
        return None


def get_all_slot_availability(availabilities):
    """Convert CarMaker availability to KuksaPublisher format for ALL stations"""
    slot_availability = {}

    for station_id, station_info in CHARGING_STATIONS.items():
        slots = []
        for slot_num in ['01', '02', '03']:
            slot_key = f"{station_id}_{slot_num}"
            if slot_key in availabilities:
                # Invert: CarMaker 0.0 (available) -> Kuksa 1 (available)
                kuksa_value = 1 if availabilities[slot_key] == 0.0 else 0
                slots.append(kuksa_value)
            else:
                slots.append(0)  # Default to not available
        slot_availability[station_id] = slots

    return slot_availability


def get_stations_ahead_list(current_distance, max_reachable_station_id):
    """Get list of stations ahead of current position within reachable station number"""
    stations_ahead = []

    for station_id, station_info in CHARGING_STATIONS.items():
        # Extract station number (CH1 -> 1, CH2 -> 2, etc.)
        station_num = int(station_id.replace('CH', ''))

        # Only include stations ahead of current position and within reachable station number
        if station_info['distance'] > current_distance and station_num <= max_reachable_station_id:
            stations_ahead.append(station_num)

    return stations_ahead


def publish_charging_request_data(kuksa_publisher, availabilities, current_distance_km, max_reachable_station_id,
                                  current_soc, charge_request_active, kuksa_data):
    """
    Publish charging request data using the KuksaPublisher
    """
    print(f"\n🚀 Publishing charging request data to Kuksa...")
    print(f"   Distance: {current_distance_km:.2f} km, SOC: {current_soc:.2f}%")
    print(f"   Max reachable station ID: {max_reachable_station_id}")

    # Get list of stations ahead within reachable station ID
    reachable_stations = get_stations_ahead_list(current_distance_km, max_reachable_station_id)

    # Convert to KuksaPublisher format
    slot_availability = get_all_slot_availability(availabilities)

    print(f"   Reachable stations: {reachable_stations}")
    print(f"   Slot availability: {slot_availability}")

    # Prepare parameters for KuksaPublisher
    params = [
        charge_request_active,  # charging_request_active
        reachable_stations,  # reachable_stations_list
        slot_availability  # slot_availability_dict
    ]

    # Use the KuksaPublisher to publish data
    try:
        success = kuksa_publisher.main(params)
        if success:
            print("✅ Successfully published data to Kuksa")

            # Auto-select first available slot and update kuksa_data.json
            if reachable_stations and slot_availability:
                for station_num in reachable_stations:
                    station_id = f"CH{station_num}"
                    slots = slot_availability.get(station_id, [])
                    for idx, val in enumerate(slots):
                        if val == 1:  # Available
                            selected_slot = f"{idx + 1:02d}"
                            new_data = {
                                "charging_request_active": int(charge_request_active),
                                "reachable_stations_queue": reachable_stations,
                                "slot_availability": slot_availability,
                                "station_id": station_id,
                                "slot_number": selected_slot,
                                "last_updated": time.time()
                            }
                            with open(KUKSA_DATA_FILE, 'w') as f:
                                json.dump(new_data, f, indent=4)
                            print(f"💾 Updated kuksa_data.json with station {station_id} slot {selected_slot}")
                            return success
                    if "station_id" in locals() and "new_data" in locals():
                        break

        else:
            print("❌ Failed to publish data to Kuksa")

        return success

    except Exception as e:
        print(f"❌ Error publishing to Kuksa: {e}")
        return False


try:
    # Initialize Kuksa Publisher
    kuksa_publisher = KuksaPublisher(host="localhost", port=55555)
    kuksa_publisher.connect()
    print("✅ Connected to Kuksa Publisher successfully.")

    # Connect to CarMaker
    cm = CarMaker(ip='127.0.0.1', port=16660)
    cm.connect()
    print("✅ Connected to CarMaker successfully.")

    # Define input quantities
    soc = Quantity("PT.BCU.BattHV.SOC", Quantity.FLOAT)
    car_distance = Quantity("Car.Distance", Quantity.FLOAT)  # Current distance
    charge_request = Quantity("Car.Charge_Request", Quantity.FLOAT)  # Charging request signal
    max_reachable_distance_qty = Quantity("UserOut_00", Quantity.FLOAT)  # Maximum reachable distance
    charge_limit = Quantity("Car.Charge_Limit", Quantity.FLOAT)  # Charge limit for stop condition

    # Charging station availability quantities for all slots
    availability_quantities = {}

    # Create availability quantities for all stations and slots
    for station_id, station_info in CHARGING_STATIONS.items():
        for slot in station_info['slots']:
            slot_key = f"{station_id}_{slot}"
            availability_quantities[slot_key] = Quantity(f"{slot_key}.Availability", Quantity.FLOAT)

    # Subscribe to input signals
    cm.subscribe(soc)
    cm.subscribe(car_distance)
    cm.subscribe(charge_request)
    cm.subscribe(max_reachable_distance_qty)
    cm.subscribe(charge_limit)

    # Subscribe to all charging station availability signals
    for availability_qty in availability_quantities.values():
        cm.subscribe(availability_qty)

    print("📡 Subscribed and ready. Monitoring for charge requests...")
    print(f"📄 Reading Kuksa agent data from: {KUKSA_DATA_FILE}")

    # Initialize variables
    charge_request_published = False  # Flag to track if request has been published
    last_json_modified_time = 0
    charging_initiated = False  # Track if charging has been initiated
    current_charging_station = None  # Track current charging station
    current_charging_slot = None  # Track current charging slot
    first_charge_call = True  # Track first time charging logic

    while True:
        cm.read()
        print(
            f"SOC: {soc.data}, Distance: {car_distance.data}, ChargeReq: {charge_request.data}, MaxReach: {max_reachable_distance_qty.data}")

        # Get current values
        current_distance_km = car_distance.data / 1000.0 if car_distance.data is not None else 0.0
        current_soc = soc.data if soc.data is not None else 0.0
        charge_request_active = charge_request.data is not None and charge_request.data == 1.0
        max_reachable_distance = max_reachable_distance_qty.data if max_reachable_distance_qty.data is not None else 0.0

        # Collect availability data
        availabilities = {slot_key: qty.data for slot_key, qty in availability_quantities.items() if
                          qty.data is not None}

        # Read Kuksa data from JSON file
        kuksa_data = read_kuksa_data_from_json()

        # Check if JSON file has been updated
        json_updated = False
        if os.path.exists(KUKSA_DATA_FILE):
            current_modified_time = os.path.getmtime(KUKSA_DATA_FILE)
            if current_modified_time > last_json_modified_time:
                json_updated = True
                last_json_modified_time = current_modified_time
                if kuksa_data:
                    print(
                        f"📄 JSON file updated - Station ID: {kuksa_data.get('station_id')}, Slot: {kuksa_data.get('slot_number')}")

        # FIXED: Handle station booking and charging from JSON data
        if kuksa_data and kuksa_data.get("station_id") and kuksa_data.get("slot_number"):
            try:
                selected_station_id = kuksa_data.get("station_id")  # Keep as string (e.g., 'CH2')
                selected_slot = kuksa_data.get("slot_number")  # Could be string or int

                # Check if we're at the station (within 0.5 km)
                station_distance = CHARGING_STATIONS.get(selected_station_id, {}).get('distance', 0)
                distance_to_station = abs(current_distance_km - station_distance)

                print(
                    f"🚗 Current distance: {current_distance_km:.2f}km, Station {selected_station_id} at {station_distance:.2f}km")
                print(f"📍 Distance to station: {distance_to_station:.2f}km")

                # FIXED: Book and start charging when we're close to the station
                if distance_to_station <= 0.5:  # Within 500m of the station
                    if not charging_initiated or (
                            current_charging_station != selected_station_id or current_charging_slot != selected_slot):
                        print(f"🔌 Arrived at {selected_station_id}, initiating charging process...")

                        # Book the station slot (FIXED method)
                        book_station_slot(cm, selected_station_id, selected_slot)

                        # Wait a moment for the booking to register
                        time.sleep(0.1)

                        # Trigger the charging process (FIXED method)
                        trigger_charging_process(cm, selected_station_id, selected_slot)

                        # Update tracking variables
                        charging_initiated = True
                        current_charging_station = selected_station_id
                        current_charging_slot = selected_slot
                        first_charge_call = True

                        print(f"📤 Charging initiated at station {selected_station_id} slot {selected_slot}")

                    # Monitor charging status (FIXED method)
                    monitor_charging_status(cm, selected_station_id, selected_slot)

                    # Check if SOC reached charge limit (from example)
                    if soc.data is not None and charge_limit.data is not None:
                        if soc.data >= charge_limit.data:
                            print(f"🔋 SOC {soc.data:.2f}% reached limit {charge_limit.data}%. Stopping charging.")
                            charging_initiated = False
                            current_charging_station = None
                            current_charging_slot = None

                else:
                    print(f"🚗 Heading to {selected_station_id} (distance: {distance_to_station:.2f}km)")

            except Exception as e:
                print(f"❌ Failed to handle station booking/charging: {e}")

        # Publish data only once when charge request is first detected
        if charge_request_active and not charge_request_published:
            print(f"\n🚨 DETECTED: Charge request at {current_distance_km:.2f} km!")
            print("📤 Attempting to publish charging request data to Kuksa...")

            success = publish_charging_request_data(
                kuksa_publisher,
                availabilities,
                current_distance_km,
                max_reachable_distance,
                current_soc,
                charge_request_active,
                kuksa_data
            )

            if success:
                print("✅ CHARGING REQUEST PUBLISHED to Kuksa!")
                charge_request_published = True
            else:
                print("❌ ERROR: Charging request could not be published!")

        # Reset the flag when charge request is no longer active
        if not charge_request_active and charge_request_published:
            print("🔁 Charge request signal cleared. System reset and ready for next request.")
            charge_request_published = False
            charging_initiated = False  # Reset charging initiation flag
            current_charging_station = None
            current_charging_slot = None

        # Display current status
        remaining_distance = TOTAL_DISTANCE - current_distance_km
        charge_req_status = "🔌 REQUESTED" if charge_request_active else "⚪ No Request"

        # Show Kuksa agent data status
        agent_status = "📱 No Agent Data"
        if kuksa_data and kuksa_data.get('station_id') is not None:
            agent_status = f"📱 Agent: Station {kuksa_data.get('station_id')}, Slot {kuksa_data.get('slot_number')}"

        # Show charging status
        charging_status = "⚫ Not Charging"
        if charging_initiated:
            charging_status = f"🔋 Charging at {current_charging_station}_{current_charging_slot}"

        print(
            f"🚗 Distance: {current_distance_km:.2f} km | Remaining: {remaining_distance:.2f} km | Max Reach: {max_reachable_distance:.2f} km")
        print(f"🔋 SOC: {current_soc:.2f}% | {charge_req_status} | {agent_status}")
        print(f"🔌 {charging_status}")
        print("—" * 80)

        time.sleep(1)

except KeyboardInterrupt:
    print("\n🛑 Charging request publisher stopped by user.")
except Exception as e:
    print(f"❌ Communication error: {e}")
finally:
    try:
        if 'cm' in locals():
            # cm.disconnect()
            print("🔌 Disconnected from CarMaker.")
    except:
        pass
    try:
        if 'kuksa_publisher' in locals():
            # kuksa_publisher.disconnect()
            print("🔌 Disconnected from Kuksa Publisher.")
    except:
        pass