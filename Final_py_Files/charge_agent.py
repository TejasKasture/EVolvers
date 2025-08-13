# charging_agent.py

from typing import List, Dict
from uagents import Agent, Context, Model
from uagents.setup import fund_agent_if_low
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import threading
import asyncio


# ---------------------- Message Models ----------------------

class SlotAvailabilityQuery(BaseModel):
    car_id: str
    station_id: str


class SlotAvailabilityResponse(BaseModel):
    car_id: str
    station_id: str
    available_slot: int  # Available slot number (1, 2, or 3) or -1 if none available


# ---------------------- Agent Configuration ----------------------

charging_agent = Agent(
    name="charging_agent",
    port=8100,
    seed="charging_agent_seed_unique_987654321",
    endpoint=["http://127.0.0.1:8100/submit"]
)

# Disable almanac registration for local testing
charging_agent._almanac_contract = None

# ---------------------- FastAPI Setup for HTTP Communication ----------------------

app = FastAPI(title="Charging Agent API")

# ---------------------- Agent State ----------------------

# Current slot availability data from KUKSA
# Format: {station_id: {slot_number: available (bool)}}
slot_availability = {}


# ---------------------- KUKSA Subscriber Functions ----------------------

def get_slot_availability_from_json(json_path="kuksa_signals.json") -> Dict[int, Dict[int, bool]]:
    """
    Reads slot availability from JSON and returns it in this format:
    {
        1: {1: True, 2: False, 3: True},
        2: {1: False, 2: True, 3: True},
        ...
    }
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        availability_raw = data.get("slot_availability", {})
        result = {}

        for ch_id, slots in availability_raw.items():
            station_id = int(ch_id.replace("CH", ""))
            result[station_id] = {
                i + 1: bool(slot) for i, slot in enumerate(slots)
            }

        return result

    except FileNotFoundError:
        print(f"⚠️ kuksa_signals.json file not found, using default availability")
        # Return default availability for testing
        return {
            1: {1: True, 2: False, 3: True},
            2: {1: False, 2: True, 3: True},
            3: {1: True, 2: True, 3: False},
            4: {1: False, 2: False, 3: True},
            5: {1: True, 2: True, 3: True}
        }
    except Exception as e:
        print(f"❌ Error reading slot availability from JSON: {e}")
        return {}


# ---------------------- Slot Management ----------------------

def find_available_slot(station_id: int) -> int:
    """
    Find first available slot in the specified station
    Returns slot number or -1 if none
    """
    availability = get_slot_availability_from_json()

    if station_id not in availability:
        return -1

    for slot_number, is_available in availability[station_id].items():
        if is_available:
            return slot_number

    return -1


# ---------------------- FastAPI Endpoints ----------------------

@app.post("/slot-query", response_model=SlotAvailabilityResponse)
async def handle_slot_query(query: SlotAvailabilityQuery):
    """Handle slot availability query via HTTP"""
    print(f"📨 HTTP Slot query from {query.car_id} for station {query.station_id}")

    try:
        station_id = int(query.station_id)
        available_slot = find_available_slot(station_id)

        response = SlotAvailabilityResponse(
            car_id=query.car_id,
            station_id=query.station_id,
            available_slot=available_slot
        )

        if available_slot != -1:
            print(f"📤 HTTP Response to {query.car_id}: Slot {available_slot} available at station {station_id}")
        else:
            print(f"📤 HTTP Response to {query.car_id}: No slots available at station {station_id}")

        return response

    except Exception as e:
        print(f"❌ Error handling HTTP query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def get_status():
    """Get charging agent status"""
    availability = get_slot_availability_from_json()
    return {
        "agent": "charging_agent",
        "status": "running",
        "slot_availability": availability
    }


# ---------------------- Agent Behaviors (Fallback for uAgent communication) ----------------------

@charging_agent.on_message(model=SlotAvailabilityQuery)
async def handle_slot_availability_query(ctx: Context, sender: str, msg: SlotAvailabilityQuery):
    """Handle slot availability query from car agent via uAgent protocol"""
    ctx.logger.info(f"📨 uAgent Slot query from {msg.car_id} for station {msg.station_id}")

    try:
        station_id = int(msg.station_id)
        available_slot = find_available_slot(station_id)

        response = SlotAvailabilityResponse(
            car_id=msg.car_id,
            station_id=msg.station_id,
            available_slot=available_slot
        )

        await ctx.send(sender, response)

        if available_slot != -1:
            ctx.logger.info(
                f"📤 uAgent Response to {msg.car_id}: Slot {available_slot} available at station {station_id}")
        else:
            ctx.logger.info(f"📤 uAgent Response to {msg.car_id}: No slots available at station {station_id}")

    except Exception as e:
        ctx.logger.error(f"❌ Error handling uAgent query: {e}")

        # Send error response
        error_response = SlotAvailabilityResponse(
            car_id=msg.car_id,
            station_id=msg.station_id,
            available_slot=-1
        )
        await ctx.send(sender, error_response)


@charging_agent.on_event("startup")
async def agent_startup(ctx: Context):
    """Initialize agent on startup"""
    ctx.logger.info("🔌 Charging Agent initialized")
    ctx.logger.info(f"📍 Agent Address: {charging_agent.address}")
    ctx.logger.info("📡 Setting up KUKSA subscription...")
    ctx.logger.info("✅ Ready to respond to slot availability queries")


# ---------------------- FastAPI Server Runner ----------------------

def run_fastapi_server():
    """Run FastAPI server in a separate thread"""
    uvicorn.run(app, host="127.0.0.1", port=8100, log_level="info")


# ---------------------- Main Execution ----------------------

if __name__ == "__main__":
    # Start FastAPI server in a separate thread
    fastapi_thread = threading.Thread(target=run_fastapi_server, daemon=True)
    fastapi_thread.start()

    print("🔌 Starting Charging Agent with HTTP API on port 8100...")
    print("📡 FastAPI server running at http://127.0.0.1:8100")

    # Run the uAgent (this will also handle the keep-alive for the process)
    charging_agent.run()