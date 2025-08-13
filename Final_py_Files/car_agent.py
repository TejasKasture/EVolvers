# car_agent.py

from typing import List
import json
import asyncio
import aiohttp
from uagents import Agent, Context, Model
from uagents.setup import fund_agent_if_low
from agent_publisher import KuksaPublisher


# ---------------------- Message Models ----------------------

class SlotAvailabilityQuery(Model):
    car_id: str
    station_id: str


class SlotAvailabilityResponse(Model):
    car_id: str
    station_id: str
    available_slot: int


# ---------------------- Agent Configuration ----------------------

CAR_ID = "EV_007"

# Charging agent endpoint
CHARGING_AGENT_URL = "http://127.0.0.1:8100"

car_agent = Agent(
    name="car_agent",
    port=8000,
    seed="car_agent_seed_unique_123456789",
    endpoint=["http://127.0.0.1:8000/submit"]
)

# Disable almanac registration for local testing
car_agent._almanac_contract = None

# ---------------------- Agent State ----------------------

charging_request_active = False
reachable_stations_queue = []
pending_station_check = None
processing_request = False

# ---------------------- kuksa initiation -------------------------

try:
    publisher = KuksaPublisher()
except Exception as e:
    print(f"⚠️ KuksaPublisher initialization failed: {e}")
    publisher = None


# ---------------------- KUKSA Interface Functions ----------------------

def on_vehicle_data_update():
    """
    Callback function when vehicle data is updated from KUKSA
    """
    global charging_request_active, reachable_stations_queue

    try:
        with open("kuksa_data.json", 'r') as f:
            data = json.load(f)

        charging_request_active = data.get("charging_request_active", False)
        reachable_stations_queue = data.get("reachable_stations_queue", [])

        print(f"📡 Charging Request: {'Active' if charging_request_active else 'Inactive'}")
        print(f"📡 Reachable Stations: {reachable_stations_queue}")

    except FileNotFoundError:
        print(f"⚠️ kuksa_data.json file not found")
    except json.JSONDecodeError:
        print(f"❌ Error parsing kuksa_data.json")
    except Exception as e:
        print(f"❌ Error reading signal values from JSON: {e}")


def publish_booking_result_to_pycarmaker(station_id: int, slot_number: int, success: bool):
    """
    Publish booking result back to PyCarMaker via KUKSA
    """
    try:
        print(f"📡 KUKSA -> PyCarMaker: Booking Result")
        print(f"   Success: {success}")
        print(f"   Station ID: {station_id}")
        print(f"   Slot: {slot_number}")

        # Only publish if we have valid station_id and slot_number for success case
        if success and station_id > 0 and slot_number > 0 and publisher:
            publisher.main(station_id, slot_number)
        elif not success:
            print(f"📡 Booking failed - not publishing to KUKSA")

    except Exception as e:
        print(f"❌ Error publishing booking result to PyCarMaker: {e}")


# ---------------------- Communication Functions ----------------------

async def send_slot_query(station_id: int) -> SlotAvailabilityResponse:
    """
    Send slot availability query to charging agent using HTTP
    """
    try:
        query = SlotAvailabilityQuery(
            car_id=CAR_ID,
            station_id=str(station_id)
        )

        async with aiohttp.ClientSession() as session:
            async with session.post(
                    f"{CHARGING_AGENT_URL}/slot-query",
                    json=query.dict(),
                    headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return SlotAvailabilityResponse(**data)
                else:
                    print(f"❌ HTTP Error {response.status}")
                    return None
    except Exception as e:
        print(f"❌ Error sending slot query: {e}")
        return None


# ---------------------- Agent Behaviors ----------------------

@car_agent.on_interval(5)
async def check_charging_request(ctx: Context):
    """
    Check for charging requests from PyCarMaker via KUKSA
    """
    global pending_station_check, processing_request

    # Skip if already processing a request
    if processing_request:
        return

    # Update data from KUKSA JSON file
    try:
        on_vehicle_data_update()
    except Exception as e:
        ctx.logger.error(f"❌ Error updating vehicle data: {e}")
        return

    # Check if charging request is active
    if not charging_request_active:
        return

    # Check if we have reachable stations
    if not reachable_stations_queue:
        ctx.logger.warning("⚠️ No reachable stations available")
        return

    ctx.logger.info(f"🎯 Processing charging request")
    ctx.logger.info(f"   Reachable stations: {reachable_stations_queue}")

    # Start processing with first reachable station
    processing_request = True
    station_to_check = reachable_stations_queue[0]
    pending_station_check = station_to_check

    # Use station ID directly (they're already numeric)
    station_number = station_to_check

    ctx.logger.info(f"🔍 Sending slot availability query for station {station_number}...")

    # Send query via HTTP
    response = await send_slot_query(station_number)

    if response:
        await handle_slot_response_internal(ctx, response)
    else:
        ctx.logger.error(f"❌ Failed to get response from charging agent")
        processing_request = False


async def handle_slot_response_internal(ctx: Context, msg: SlotAvailabilityResponse):
    """Handle slot availability response from charging agent"""
    global pending_station_check, processing_request, charging_request_active, reachable_stations_queue

    ctx.logger.info(f"📨 Slot response for station {msg.station_id}")
    ctx.logger.info(f"   Available slot: {msg.available_slot}")

    if msg.available_slot != -1:
        # Slot available - booking successful
        station_id_numeric = int(msg.station_id)

        ctx.logger.info(f"✅ Found available slot {msg.available_slot} at station {msg.station_id}")

        # Send result to PyCarMaker via KUKSA
        publish_booking_result_to_pycarmaker(
            station_id=station_id_numeric,
            slot_number=msg.available_slot,
            success=True
        )

        ctx.logger.info(f"📡 Booking result sent to PyCarMaker")
        ctx.logger.info(f"   Station: {station_id_numeric}")
        ctx.logger.info(f"   Slot: {msg.available_slot}")

        # Reset state - request fulfilled
        processing_request = False
        pending_station_check = None
        charging_request_active = False

    else:
        # No slot available - try next station
        ctx.logger.warning(f"🚫 No available slots at station {msg.station_id}")

        try:
            # Find current station index in queue
            current_index = reachable_stations_queue.index(pending_station_check)

            if current_index + 1 < len(reachable_stations_queue):
                # Try next station in queue
                next_station = reachable_stations_queue[current_index + 1]
                pending_station_check = next_station

                ctx.logger.info(f"🔍 Trying next station {next_station}...")

                response = await send_slot_query(next_station)
                if response:
                    await handle_slot_response_internal(ctx, response)
                    return  # Continue processing with next station
                else:
                    ctx.logger.error(f"❌ Failed to get response for station {next_station}")

            # No more stations to try - all stations exhausted
            ctx.logger.error(f"❌ No available slots at any reachable station")
            ctx.logger.info(f"   Tried stations: {reachable_stations_queue}")

            # Send failure result to PyCarMaker
            publish_booking_result_to_pycarmaker(
                station_id=0,  # Use 0 to indicate no station found
                slot_number=-1,
                success=False
            )

            ctx.logger.info(f"📡 Failure result sent to PyCarMaker: No slots available")

            # Reset state - request failed
            processing_request = False
            pending_station_check = None
            charging_request_active = False

        except (ValueError, IndexError) as e:
            ctx.logger.error(f"❌ Error processing station queue: {e}")
            ctx.logger.error(f"   Current station: {pending_station_check}")
            ctx.logger.error(f"   Station queue: {reachable_stations_queue}")

            # Send failure result to PyCarMaker
            publish_booking_result_to_pycarmaker(
                station_id=0,
                slot_number=-1,
                success=False
            )

            # Reset state - error occurred
            processing_request = False
            pending_station_check = None
            charging_request_active = False


@car_agent.on_event("startup")
async def agent_startup(ctx: Context):
    """Initialize agent on startup"""
    ctx.logger.info("🚗 Car Agent initialized")
    ctx.logger.info(f"📍 Car ID: {CAR_ID}")
    ctx.logger.info(f"🔌 Charging Agent: {CHARGING_AGENT_URL}")


# ---------------------- Utility Functions ----------------------

def set_charging_request(request_active: bool, stations: List[str] = None):
    """Set charging request for testing/integration"""
    global charging_request_active, reachable_stations_queue

    charging_request_active = request_active
    print(f"📋 Set Charging Request: {charging_request_active}")

    if stations is not None:
        reachable_stations_queue = stations
        print(f"🎯 Set Reachable Stations: {reachable_stations_queue}")


def get_agent_status() -> dict:
    """Get current agent status"""
    return {
        "car_id": CAR_ID,
        "charging_request_active": charging_request_active,
        "reachable_stations": reachable_stations_queue,
        "pending_station_check": pending_station_check,
        "processing_request": processing_request
    }


# ---------------------- Main Execution ----------------------

if __name__ == "__main__":
    car_agent.run()