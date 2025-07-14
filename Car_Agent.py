import asyncio
import json
import logging
from typing import Dict, List, Optional
from uagents import Agent, Context, Model, Bureau
from uagents.setup import fund_agent_if_low
from kuksa_client.grpc import VSSClient, Datapoint
import time
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import models from charge_agent
from Charge_Agent import ChargingRequest, ChargingResponse, StationStatus


class VehicleState(Model):
    vehicle_id: str
    current_soc: float
    target_soc: float
    current_location: float  # km from start
    distance_covered: float  # meters
    reachable_distance: float  # meters
    velocity: float  # m/s
    charging_needed: bool


class ChargingDecision(Model):
    vehicle_id: str
    should_charge: bool
    selected_station: int
    charging_priority: int
    estimated_charging_time: float
    route_feasible: bool


class CarAgent(Agent):
    """
    Fetch.ai Car Decision Agent
    Makes final charging decisions based on FMU recommendations and charge agent responses
    """

    def __init__(self, name: str, seed: str = ""):
        super().__init__(name=name, seed=seed)

        # Vehicle configuration
        self.vehicle_id = str(uuid.uuid4())[:8]
        self.route_distance = 200000.0  # 200km route in meters
        self.battery_capacity = 60.0  # kWh
        self.energy_consumption = 0.15  # kWh/km

        # Current vehicle state
        self.vehicle_state = VehicleState(
            vehicle_id=self.vehicle_id,
            current_soc=1.0,
            target_soc=0.8,
            current_location=0.0,
            distance_covered=0.0,
            reachable_distance=0.0,
            velocity=0.0,
            charging_needed=False
        )

        # Charging decision logic
        self.charging_thresholds = {
            "emergency": 0.08,  # 8% - Emergency charging
            "urgent": 0.15,  # 15% - Urgent charging
            "normal": 0.25,  # 25% - Normal charging
            "optimal": 0.35  # 35% - Optimal charging
        }

        # Communication with other agents
        self.charge_agent_address = None
        self.pending_requests = {}  # request_id -> timestamp
        self.charging_history = []

        # Kuksa client for CarMaker communication
        self.kuksa_client = None
        self.kuksa_connected = False

        # Decision metrics
        self.total_decisions = 0
        self.charging_decisions = 0
        self.successful_charges = 0
        self.route_completion_possible = True

    async def initialize_kuksa(self):
        """Initialize Kuksa client connection"""
        try:
            self.kuksa_client = VSSClient("localhost", 55555)
            self.kuksa_client.connect()
            self.kuksa_connected = True
            logger.info("Car Agent connected to Kuksa DataBroker")
        except Exception as e:
            logger.error(f"Failed to connect to Kuksa: {e}")
            self.kuksa_connected = False

    async def read_vehicle_data_from_kuksa(self):
        """Read vehicle data from Kuksa (from CarMaker via PyCarmaker)"""
        if not self.kuksa_connected or self.kuksa_client is None:
            return

        try:
            # Read vehicle signals
            signals = [
                "Vehicle.Speed",
                "Vehicle.Powertrain.TractionBattery.StateOfCharge.Current",
                "Vehicle.TraveledDistance",
                "Vehicle.Charging.ReachableDistance",
                "Vehicle.Charging.Request"
            ]

            response = self.kuksa_client.get_current_values(signals)

            # Update vehicle state
            if "Vehicle.Speed" in response and response["Vehicle.Speed"].value is not None:
                self.vehicle_state.velocity = float(response["Vehicle.Speed"].value)

            if "Vehicle.Powertrain.TractionBattery.StateOfCharge.Current" in response and response["Vehicle.Powertrain.TractionBattery.StateOfCharge.Current"].value is not None:
                self.vehicle_state.current_soc = float(response["Vehicle.Powertrain.TractionBattery.StateOfCharge.Current"].value) / 100.0

            if "Vehicle.TraveledDistance" in response and response["Vehicle.TraveledDistance"].value is not None:
                self.vehicle_state.distance_covered = float(response["Vehicle.TraveledDistance"].value)
                self.vehicle_state.current_location = float(response["Vehicle.TraveledDistance"].value) / 1000.0

            if "Vehicle.Charging.ReachableDistance" in response and response["Vehicle.Charging.ReachableDistance"].value is not None:
                self.vehicle_state.reachable_distance = float(response["Vehicle.Charging.ReachableDistance"].value)

            if "Vehicle.Charging.Request" in response and response["Vehicle.Charging.Request"].value is not None:
                self.vehicle_state.charging_needed = bool(response["Vehicle.Charging.Request"].value)

        except Exception as e:
            logger.error(f"Error reading vehicle data from Kuksa: {e}")

    async def publish_decision_to_kuksa(self, decision: ChargingDecision):
        """Publish charging decision to Kuksa"""
        if not self.kuksa_connected or self.kuksa_client is None:
            return

        try:
            decision_data = {
                "Vehicle.Charging.Decision.ShouldCharge": decision.should_charge,
                "Vehicle.Charging.Decision.SelectedStation": decision.selected_station,
                "Vehicle.Charging.Decision.Priority": decision.charging_priority,
                "Vehicle.Charging.Decision.EstimatedTime": decision.estimated_charging_time,
                "Vehicle.Charging.Decision.RouteFeasible": decision.route_feasible
            }

            datapoints = {path: Datapoint(value) for path, value in decision_data.items()}
            self.kuksa_client.set_current_values(datapoints)

            logger.info(f"Published charging decision to Kuksa: {decision}")

        except Exception as e:
            logger.error(f"Error publishing decision to Kuksa: {e}")

    def calculate_charging_priority(self) -> int:
        """Calculate charging priority based on current state"""
        soc = self.vehicle_state.current_soc
        remaining_distance = self.route_distance - self.vehicle_state.distance_covered
        reachable_distance = self.vehicle_state.reachable_distance

        # Emergency priority
        if soc <= self.charging_thresholds["emergency"]:
            return 3

        # Urgent priority - cannot complete route
        if reachable_distance < remaining_distance * 1.2:  # 20% safety margin
            return 2

        # Normal priority - proactive charging
        if soc <= self.charging_thresholds["normal"]:
            return 1

        # No charging needed
        return 0

    def calculate_target_soc(self) -> float:
        """Calculate target SoC for charging"""
        remaining_distance_km = (self.route_distance - self.vehicle_state.distance_covered) / 1000.0
        energy_needed = remaining_distance_km * self.energy_consumption

        # Add safety margin
        energy_needed *= 1.3  # 30% safety margin

        # Convert to SoC
        target_soc = energy_needed / self.battery_capacity

        # Ensure reasonable bounds
        target_soc = max(0.5, min(target_soc, 1.0))

        return target_soc

    def assess_route_feasibility(self) -> bool:
        """Assess if route completion is feasible"""
        remaining_distance = self.route_distance - self.vehicle_state.distance_covered

        # Check if reachable distance covers remaining route
        if self.vehicle_state.reachable_distance >= remaining_distance:
            return True

        # Check if charging can make route feasible
        max_energy_after_charge = self.battery_capacity  # Full charge
        max_distance_after_charge = (max_energy_after_charge / self.energy_consumption) * 1000

        return max_distance_after_charge >= remaining_distance

    async def make_charging_decision(self) -> ChargingDecision:
        """Make intelligent charging decision"""
        self.total_decisions += 1

        # Calculate decision parameters
        priority = self.calculate_charging_priority()
        target_soc = self.calculate_target_soc()
        route_feasible = self.assess_route_feasibility()

        # Decide if charging is needed
        should_charge = (
                priority > 0 and
                self.vehicle_state.charging_needed and
                route_feasible
        )

        # Estimate charging time
        if should_charge:
            energy_to_charge = (target_soc - self.vehicle_state.current_soc) * self.battery_capacity
            charging_time = energy_to_charge / 50.0 * 60  # 50kW charging, convert to minutes
        else:
            charging_time = 0.0

        decision = ChargingDecision(
            vehicle_id=self.vehicle_id,
            should_charge=should_charge,
            selected_station=0,  # Will be updated after charge agent response
            charging_priority=priority,
            estimated_charging_time=charging_time,
            route_feasible=route_feasible
        )

        # If charging is needed, request from charge agent
        if should_charge:
            await self.request_charging_slot(decision)

        return decision

    async def request_charging_slot(self, decision: ChargingDecision):
        """Request charging slot from charge agent"""
        if not self.charge_agent_address:
            # Find charge agent (in real implementation, this would be from agent registry)
            logger.warning("Charge agent address not available")
            return

        # Create charging request
        request = ChargingRequest(
            vehicle_id=self.vehicle_id,
            current_soc=self.vehicle_state.current_soc,
            target_soc=self.vehicle_state.target_soc,
            current_location=self.vehicle_state.current_location,
            reachable_distance=self.vehicle_state.reachable_distance,
            priority=decision.charging_priority
        )

        # Send request to charge agent
        try:
            # This would be sent to the charge agent
            logger.info(f"Sending charging request to charge agent: {request}")
            self.pending_requests[request.vehicle_id] = time.time()

            # For simulation, we'll assume a response
            # In real implementation, this would be handled by the message system

        except Exception as e:
            logger.error(f"Error sending charging request: {e}")

    async def handle_charging_response(self, ctx: Context, sender: str, msg: ChargingResponse):
        """Handle response from charge agent"""
        logger.info(f"Received charging response from {sender}: {msg}")

        # Update pending requests
        if msg.vehicle_id in self.pending_requests:
            del self.pending_requests[msg.vehicle_id]

        # Process response
        if msg.charging_slots_available:
            # Update decision with allocated station
            decision = ChargingDecision(
                vehicle_id=msg.vehicle_id,
                should_charge=True,
                selected_station=msg.station_allocated,
                charging_priority=self.calculate_charging_priority(),
                estimated_charging_time=msg.estimated_wait_time,
                route_feasible=True
            )

            self.charging_decisions += 1
            self.successful_charges += 1

            # Add to charging history
            self.charging_history.append({
                "timestamp": time.time(),
                "station": msg.station_allocated,
                "soc_before": self.vehicle_state.current_soc,
                "estimated_wait": msg.estimated_wait_time
            })

            # Publish decision
            await self.publish_decision_to_kuksa(decision)

            logger.info(f"Charging allocated at station {msg.station_allocated}")
        else:
            logger.warning(f"Charging request rejected for vehicle {msg.vehicle_id}")

    async def decision_loop(self):
        """Main decision-making loop"""
        while True:
            try:
                # Read current vehicle state
                await self.read_vehicle_data_from_kuksa()

                # Make charging decision
                decision = await self.make_charging_decision()

                # Publish decision to Kuksa
                await self.publish_decision_to_kuksa(decision)

                # Update route feasibility
                self.route_completion_possible = decision.route_feasible

                # Log decision
                logger.info(f"Decision made: {decision}")

                await asyncio.sleep(1.0)  # Decision loop at 1Hz

            except Exception as e:
                logger.error(f"Error in decision loop: {e}")
                await asyncio.sleep(5.0)

    def get_performance_metrics(self) -> Dict:
        """Get performance metrics"""
        success_rate = self.successful_charges / self.charging_decisions if self.charging_decisions > 0 else 0

        return {
            "vehicle_id": self.vehicle_id,
            "total_decisions": self.total_decisions,
            "charging_decisions": self.charging_decisions,
            "successful_charges": self.successful_charges,
            "success_rate": success_rate,
            "route_completion_possible": self.route_completion_possible,
            "current_soc": self.vehicle_state.current_soc,
            "distance_covered": self.vehicle_state.distance_covered,
            "charging_history": len(self.charging_history)
        }


# Create the car agent
car_agent = CarAgent("car_agent", seed="car_agent_seed_456")


@car_agent.on_event("startup")
async def startup_handler(ctx: Context):
    """Handle agent startup"""
    logger.info(f"Car Agent starting up... Vehicle ID: {car_agent.vehicle_id}")
    await car_agent.initialize_kuksa()

    # Start decision loop
    asyncio.create_task(car_agent.decision_loop())

    # Fund agent if needed
    fund_agent_if_low(str(car_agent.wallet.address()))


@car_agent.on_message(ChargingResponse)
async def handle_charging_response(ctx: Context, sender: str, msg: ChargingResponse):
    """Handle charging response messages"""
    await car_agent.handle_charging_response(ctx, sender, msg)


@car_agent.on_event("shutdown")
async def shutdown_handler(ctx: Context):
    """Handle agent shutdown"""
    logger.info("Car Agent shutting down...")
    if car_agent.kuksa_connected and car_agent.kuksa_client is not None:
        car_agent.kuksa_client.disconnect()


# Status endpoint for monitoring
class StatusQuery(Model):
    pass

@car_agent.on_query(StatusQuery)
async def query_status(ctx: Context, _msg: StatusQuery) -> Dict:
    """Query agent status"""
    return {
        "agent_name": "car_agent",
        "vehicle_state": car_agent.vehicle_state.dict(),
        "performance": car_agent.get_performance_metrics(),
        "kuksa_connected": car_agent.kuksa_connected
    }


if __name__ == "__main__":
    logger.info("Starting Car Agent...")
    car_agent.run()