import asyncio
import logging
from uuid import uuid4
import httpx
import json
from strands import Agent
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest
from tools.restaurant_tools import check_availability, book_restaurant, cancel_booking, set_coordinator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RestaurantBookingCoordinator:
    def __init__(self):
        self.restaurants = {
            "Sushi Maru": "http://localhost:9001",
            "Tokyo Ramen": "http://localhost:9002",
            "Takoyaki Taro": "http://localhost:9003",
        }
        self.timeout = 60
        self.agent = Agent(
            name="Restaurant Booking Coordinator",
            system_prompt="""You are a restaurant booking coordinator that helps users check availability, make reservations, and cancel bookings across multiple restaurants using A2A protocol.

Available restaurants: Sushi Maru, Tokyo Ramen, Takoyaki Taro

You have access to these tools:
- check_availability: Check availability at all restaurants for a given date, time, and party size
- book_restaurant: Book a table at a specific restaurant 
- cancel_booking: Cancel an existing booking

Always use the tools to fulfill user requests. Parse dates in YYYY-MM-DD format and times in 24-hour HH:MM format.

Be helpful and provide clear responses about booking status and availability.""",
            tools=[check_availability, book_restaurant, cancel_booking],
            callback_handler=None,
        )
        
        # Set coordinator instance for tools
        set_coordinator(self)

    async def discover_restaurants(self):
        """Discover available restaurant agents via A2A protocol"""
        available_restaurants = {}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for name, url in self.restaurants.items():
                try:
                    logger.info(f"🔍 Discovering {name} at {url}...")
                    resolver = A2ACardResolver(httpx_client=client, base_url=url)
                    agent_card = await resolver.get_agent_card()

                    # Store the URL and card info, we'll create fresh clients for each request
                    available_restaurants[name] = {"url": url, "card": agent_card}
                    logger.info(f"✅ Discovered {name}: {agent_card.name}")
                except Exception as e:
                    logger.warning(f"❌ Could not connect to {name} at {url}: {e}")

        return available_restaurants

    async def query_restaurant(self, restaurant_info, message):
        """Send a message to a specific restaurant agent via A2A"""
        try:
            # Create a fresh HTTP client for each request
            async with httpx.AsyncClient(timeout=self.timeout) as fresh_client:
                # Get the agent card info but create fresh client
                url = restaurant_info["url"]
                resolver = A2ACardResolver(httpx_client=fresh_client, base_url=url)
                agent_card = await resolver.get_agent_card()
                client = A2AClient(httpx_client=fresh_client, agent_card=agent_card)

                request = SendMessageRequest(
                    id=str(uuid4()),
                    params=MessageSendParams(
                        message={
                            "role": "user",
                            "parts": [{"kind": "text", "text": message}],
                            "messageId": uuid4().hex,
                        }
                    ),
                )

                response = await client.send_message(request)

                # Debug logging for troubleshooting
                logger.debug(f"Response type: {type(response)}")
                logger.debug(f"Response: {response}")

                # Extract message from response
                if hasattr(response, "root") and response.root:
                    result = response.root.result

                    # Check if it's a Task with status message
                    if hasattr(result, "status") and hasattr(result.status, "message"):
                        message = result.status.message
                        if hasattr(message, "parts"):
                            response_parts = []
                            for part in message.parts:
                                if hasattr(part, "root") and hasattr(part.root, "text"):
                                    response_parts.append(part.root.text)
                            if response_parts:
                                return "".join(response_parts)

                    # Check for artifacts
                    if hasattr(result, "artifacts") and result.artifacts:
                        response_parts = []
                        for artifact in result.artifacts:
                            if hasattr(artifact, "parts"):
                                for part in artifact.parts:
                                    if hasattr(part, "root") and hasattr(
                                        part.root, "text"
                                    ):
                                        response_parts.append(part.root.text)
                        if response_parts:
                            return "".join(response_parts)

                    # Check for direct message parts
                    if hasattr(result, "parts"):
                        response_parts = []
                        for part in result.parts:
                            if hasattr(part, "root") and hasattr(part.root, "text"):
                                response_parts.append(part.root.text)
                        if response_parts:
                            return "".join(response_parts)

                return str(response)

        except Exception as e:
            logger.error(f"Error querying restaurant: {e}")
            return f"Error communicating with restaurant: {str(e)}"






async def main():
    print("🍽️" + "=" * 60)
    print("     A2A RESTAURANT BOOKING COORDINATOR")
    print("     Demonstrating Cross-Framework Communication")
    print("=" * 63)
    print()
    print("🔧 Architecture:")
    print("  • Sushi Maru (port 9001) - Strands Agents")
    print("  • Tokyo Ramen (port 9002) - LangGraph + Gemini")
    print("  • Takoyaki Taro (port 9003) - Google ADK + Gemini")
    print("  • This Coordinator - Strands Agent with A2A Client Tools")
    print()

    coordinator = RestaurantBookingCoordinator()

    print("🔍 Testing A2A Discovery...")
    restaurants = await coordinator.discover_restaurants()

    if not restaurants:
        print("\n❌ No restaurant agents found!")
        print("Make sure all restaurant agents are running:")
        print("  Terminal 1: uv run sushi_maru_agent.py")
        print("  Terminal 2: uv run tokyo_ramen_agent.py")
        print("  Terminal 3: uv run takoyaki_taro_agent.py")
        return

    print(f"\n✅ Successfully discovered {len(restaurants)} restaurant agents!")
    for name in restaurants.keys():
        print(f"  • {name}")

    print("\n🎬 DEMO SCENARIOS")
    print("=" * 60)

    # Demo 1: Check availability
    print("\n📋 Demo 1: Agent checking availability at all restaurants")
    availability_result = await coordinator.agent.invoke_async(
        "Check availability for 4 people on 2025-07-25 at 19:00"
    )
    print("\n📊 Results:")
    print(availability_result)

    # Demo 2: Book at specific restaurant
    print("\n\n📋 Demo 2: Agent booking at Sushi Maru")
    booking_result = await coordinator.agent.invoke_async(
        "Book a table at Sushi Maru for 4 people on 2025-07-25 at 20:00 for John Smith"
    )
    print("\n📊 Booking Result:")
    print(booking_result)

    # Interactive mode
    print("\n\n🎯 INTERACTIVE MODE")
    print("=" * 60)
    print("Natural language examples:")
    print("  'Check availability for 4 people on 2025-07-25 at 19:00'")
    print(
        "  'Book a table at Sushi Maru for 2 people on 2025-07-25 at 20:00 for John Smith'"
    )
    print("  'Is Tokyo Ramen available for 6 guests on 2025-07-26 at 18:30?'")
    print(
        "  'Make a reservation at Takoyaki Taro for 3 people on 2025-07-27 at 19:15 under Jane Doe'"
    )
    print("  'Cancel booking at Sushi Maru for John Smith on 2025-07-25 at 20:00'")
    print("  'quit' to exit")
    print()

    while True:
        try:
            user_input = input("👤 Request: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                break

            # Let the agent intelligently choose which tool to use
            print(f"\n🤖 Agent processing: {user_input}")
            response = await coordinator.agent.invoke_async(user_input)
            print(f"\n📊 Response:\n{response}\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")

    print("\n👋 Thanks for testing the A2A Restaurant Booking System!")
    print("🎉 Demo complete - three different frameworks communicating via A2A!")


if __name__ == "__main__":
    print("🚀 A2A Cross-Framework Restaurant Booking Demo")
    print("Make sure all restaurant agents are running first!")
    print()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Demo stopped by user")
