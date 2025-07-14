import asyncio
import json
import logging
from typing import Dict, List, Optional
from uagents import Agent, Context, Model
from uagents.setup import fund_agent_if_low
from kuksa_client.grpc import VSSClient, Datapoint
import time
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for message passing
class ChargingRequest(Model):
    vehicle_id: str
    current_soc: float
    target_soc: float
    current_location: float  # km from start
    reachable_distance: float  # meters
    priority: int = 1  # 1=normal, 2=urgent, 3=emergency


class ChargingResponse(Model):
    vehicle_id: str
    station_allocated: int
    estimated_wait_time: float  # minutes
    charging_slots_available: bool
    alternative_stations: List[int]


class StationStatus(Model):
    station_id: int
    available_slots: int
    total_slots: int
    current_queue: int
    estimated_charging_time: float
    location_km: float


class ChargeAgent(Agent):
    """
    Fetch.ai Charging Station Management Agent
    Manages charging station availability and allocation
    """

    def __init__(self, name: str, seed: str = ""):
        super().__init__(name=name, seed=seed)

        # Charging station data from Phase 2
        self.charging_stations = {
            1: {
                "label": "CH1",
                "location_km": 2.38,
                "total_slots": 2,
                "available_slots": 2,
                "queue": [],
                "charging_power": 50,  # kW
                "estimated_charging_time": 30  # minutes
            },
            2: {
                "label": "CH2",
                "location_km": 12.62,
                "total_slots": 1,
                "available_slots": 1,
                "queue": [],
                "charging_power": 50,
                "estimated_charging_time": 30
            },
            3: {
                "label": "CH3",
                "location_km": 24.30,
                "total_slots": 3,
                "available_slots": 3,
                "queue": [],
                "charging_power": 50,
                "estimated_charging_time": 30
            },
            4: {
                "label": "CH4",
                "location_km": 55.37,
                "total_slots": 1,
                "available_slots": 0,  # Initially occupied
                "queue": [],
                "charging_power": 50,
                "estimated_charging_time": 30
            },
            5: {
                "label": "CH5",
                "location_km": 95.11,
                "total_slots": 2,
                "available_slots": 2,
                "queue": [],
                "charging_power": 50,
                "estimated_charging_time": 30
            }
        }

        # Current allocations
        self.current_allocations = {}  # vehicle_id -> station_id

        # Kuksa client for communication
        self.kuksa_client = None
        self.kuksa_connected = False

        # Performance metrics
        self.total_requests = 0
        self.successful_allocations = 0
        self.rejected_requests = 0

    async def initialize_kuksa(self):
        """Initialize Kuksa client connection"""
        try:
            self.kuksa_client = VSSClient("localhost", 55555)
            self.kuksa_client.connect()
            self.kuksa_connected = True
            logger.info("Charge Agent connected to Kuksa DataBroker")
        except Exception as e:
            logger.error(f"Failed to connect to Kuksa: {e}")
            self.kuksa_connected = False

    async def publish_station_status(self):
        """Publish station status to Kuksa"""
        if not self.kuksa_connected:
            return

        try:
            # Publish availability data
            availability_data = {}
            for station_id, station in self.charging_stations.items():
                availability_data[f"Vehicle.Charging.Station.{station_id}.AvailableSlots"] = station["available_slots"]
                availability_data[f"Vehicle.Charging.Station.{station_id}.TotalSlots"] = station["total_slots"]
                availability_data[f"Vehicle.Charging.Station.{station_id}.QueueLength"] = len(station["queue"])

            datapoints = {path: Datapoint(value) for path, value in availability_data.items()}
            if self.kuksa_client is not None:
                self.kuksa_client.set_current_values(datapoints)

        except Exception as e:
            logger.error(f"Error publishing station status: {e}")

    def find_optimal_station(self, request: ChargingRequest) -> Optional[int]:
        """Find optimal charging station based on request"""
        current_location = request.current_location
        reachable_distance_km = request.reachable_distance / 1000.0

        # Find reachable stations with available slots
        reachable_stations = []
        for station_id, station in self.charging_stations.items():
            distance_to_station = abs(station["location_km"] - current_location)

            if distance_to_station <= reachable_distance_km:
                if station["available_slots"] > 0:
                    reachable_stations.append({
                        "id": station_id,
                        "distance": distance_to_station,
                        "available_slots": station["available_slots"],
                        "queue_length": len(station["queue"]),
                        "charging_time": station["estimated_charging_time"]
                    })

        if not reachable_stations:
            return None

        # Prioritize based on:
        # 1. Emergency requests - closest station
        # 2. Normal requests - optimal combination of distance and availability
        if request.priority >= 3:  # Emergency
            return min(reachable_stations, key=lambda x: x["distance"])["id"]
        else:
            # Score based on distance, availability, and queue
            def calculate_score(station):
                distance_score = 1.0 / (station["distance"] + 1)  # Closer is better
                availability_score = station["available_slots"] / 3.0  # More slots is better
                queue_score = 1.0 / (station["queue_length"] + 1)  # Shorter queue is better

                return distance_score * 0.4 + availability_score * 0.3 + queue_score * 0.3

            optimal_station = max(reachable_stations, key=calculate_score)
            return optimal_station["id"]

    def allocate_charging_slot(self, vehicle_id: str, station_id: int) -> bool:
        """Allocate a charging slot at the specified station"""
        if station_id not in self.charging_stations:
            return False

        station = self.charging_stations[station_id]

        if station["available_slots"] > 0:
            # Allocate slot
            station["available_slots"] -= 1
            station["queue"].append(vehicle_id)
            self.current_allocations[vehicle_id] = station_id

            logger.info(f"Allocated charging slot at {station['label']} for vehicle {vehicle_id}")
            return True
        else:
            logger.warning(f"No available slots at {station['label']} for vehicle {vehicle_id}")
            return False

    def release_charging_slot(self, vehicle_id: str):
        """Release a charging slot when vehicle finishes charging"""
        if vehicle_id in self.current_allocations:
            station_id = self.current_allocations[vehicle_id]
            station = self.charging_stations[station_id]

            # Release slot
            station["available_slots"] += 1
            if vehicle_id in station["queue"]:
                station["queue"].remove(vehicle_id)

            del self.current_allocations[vehicle_id]

            logger.info(f"Released charging slot at {station['label']} for vehicle {vehicle_id}")

    async def handle_charging_request(self, ctx: Context, sender: str, msg: ChargingRequest):
        """Handle incoming charging requests"""
        self.total_requests += 1
        logger.info(f"Received charging request from {sender}: {msg}")

        # Find optimal station
        optimal_station_id = self.find_optimal_station(msg)

        if optimal_station_id is None:
            # No suitable station found
            response = ChargingResponse(
                vehicle_id=msg.vehicle_id,
                station_allocated=0,
                estimated_wait_time=0.0,
                charging_slots_available=False,
                alternative_stations=[]
            )
            self.rejected_requests += 1
        else:
            # Allocate charging slot
            if self.allocate_charging_slot(msg.vehicle_id, optimal_station_id):
                station = self.charging_stations[optimal_station_id]
                estimated_wait = len(station["queue"]) * station["estimated_charging_time"]

                # Find alternative stations
                alternatives = []
                for station_id, station_data in self.charging_stations.items():
                    if station_id != optimal_station_id and station_data["available_slots"] > 0:
                        distance = abs(station_data["location_km"] - msg.current_location)
                        if distance <= msg.reachable_distance / 1000.0:
                            alternatives.append(station_id)

                response = ChargingResponse(
                    vehicle_id=msg.vehicle_id,
                    station_allocated=optimal_station_id,
                    estimated_wait_time=estimated_wait,
                    charging_slots_available=True,
                    alternative_stations=alternatives
                )
                self.successful_allocations += 1
            else:
                response = ChargingResponse(
                    vehicle_id=msg.vehicle_id,
                    station_allocated=0,
                    estimated_wait_time=0.0,
                    charging_slots_available=False,
                    alternative_stations=[]
                )
                self.rejected_requests += 1

        # Send response
        await ctx.send(sender, response)

        # Update Kuksa with latest station status
        await self.publish_station_status()

    async def simulate_charging_completion(self):
        """Simulate charging completion for allocated vehicles"""
        while True:
            try:
                # Randomly complete some charging sessions
                if self.current_allocations:
                    for vehicle_id in list(self.current_allocations.keys()):
                        # Random chance of completing charging
                        if random.random() < 0.1:  # 10% chance per cycle
                            self.release_charging_slot(vehicle_id)

                # Update station status
                await self.publish_station_status()

                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                logger.error(f"Error in charging simulation: {e}")
                await asyncio.sleep(10)

    def get_station_status(self) -> Dict:
        """Get current status of all charging stations"""
        status = {}
        for station_id, station in self.charging_stations.items():
            status[station_id] = {
                "label": station["label"],
                "location_km": station["location_km"],
                "available_slots": station["available_slots"],
                "total_slots": station["total_slots"],
                "queue_length": len(station["queue"]),
                "utilization": (station["total_slots"] - station["available_slots"]) / station["total_slots"]
            }
        return status

    def get_performance_metrics(self) -> Dict:
        """Get performance metrics"""
        success_rate = self.successful_allocations / self.total_requests if self.total_requests > 0 else 0
        return {
            "total_requests": self.total_requests,
            "successful_allocations": self.successful_allocations,
            "rejected_requests": self.rejected_requests,
            "success_rate": success_rate,
            "current_allocations": len(self.current_allocations)
        }


# Create the charge agent
charge_agent = ChargeAgent("charge_agent", seed="charge_agent_seed_123")


@charge_agent.on_event("startup")
async def startup_handler(ctx: Context):
    """Handle agent startup"""
    logger.info("Charge Agent starting up...")
    await charge_agent.initialize_kuksa()

    # Start background tasks
    asyncio.create_task(charge_agent.simulate_charging_completion())

    # Fund agent if needed
    fund_agent_if_low(str(charge_agent.wallet.address()))


@charge_agent.on_message(ChargingRequest)
async def handle_charging_request(ctx: Context, sender: str, msg: ChargingRequest):
    """Handle charging request messages"""
    await charge_agent.handle_charging_request(ctx, sender, msg)


@charge_agent.on_event("shutdown")
async def shutdown_handler(ctx: Context):
    """Handle agent shutdown"""
    logger.info("Charge Agent shutting down...")
    if charge_agent.kuksa_connected and charge_agent.kuksa_client is not None:
        charge_agent.kuksa_client.disconnect()


class StatusQuery(Model):
    pass

@charge_agent.on_query(StatusQuery)
async def query_status(ctx: Context, _msg: StatusQuery) -> Dict:
    """Query agent status"""
    return {
        "agent_name": "charge_agent",
        "stations": charge_agent.get_station_status(),
        "performance": charge_agent.get_performance_metrics(),
        "kuksa_connected": charge_agent.kuksa_connected
    }


if __name__ == "__main__":
    logger.info("Starting Charge Agent...")
    charge_agent.run()