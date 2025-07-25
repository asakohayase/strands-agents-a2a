import asyncio
import logging
from uuid import uuid4
import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest

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

                # CORRECT: Use the new A2A SDK response structure
                # Access via response.root (new SDK pattern)
                if hasattr(response, "root") and response.root:
                    # Check if it's a success response
                    if hasattr(response.root, "result"):
                        result = response.root.result

                        # If result is a Task, we might need to get final result differently
                        if hasattr(result, "artifacts") and result.artifacts:
                            # Extract text from artifacts
                            response_parts = []
                            for artifact in result.artifacts:
                                if hasattr(artifact, "parts"):
                                    for part in artifact.parts:
                                        if hasattr(part, "root") and hasattr(
                                            part.root, "text"
                                        ):
                                            response_parts.append(part.root.text)
                            return (
                                "".join(response_parts)
                                if response_parts
                                else str(result)
                            )

                        # If result is a Message directly
                        elif hasattr(result, "parts"):
                            response_parts = []
                            for part in result.parts:
                                if hasattr(part, "root") and hasattr(part.root, "text"):
                                    response_parts.append(part.root.text)
                                elif hasattr(part, "text"):
                                    response_parts.append(part.text)
                            return (
                                "".join(response_parts)
                                if response_parts
                                else str(result)
                            )

                        # Fallback to string representation
                        else:
                            return str(result)

                    # If no result, try to get error info
                    elif hasattr(response.root, "error"):
                        return f"Agent error: {response.root.error}"
                    else:
                        return str(response.root)

                # Fallback for unexpected response structure
                else:
                    return f"Unexpected response format: {str(response)}"

        except Exception as e:
            logger.error(f"Error querying restaurant: {e}")
            return f"Error communicating with restaurant: {str(e)}"

    async def check_all_availability(self, date, time, party_size):
        """Check availability at all restaurants via A2A"""
        print(f"\n🔍 Checking availability for {party_size} people on {date} at {time}")
        print("-" * 60)

        restaurants = await self.discover_restaurants()

        if not restaurants:
            return "❌ No restaurants are currently available."

        query = f"Check availability for {party_size} people on {date} at {time}"
        results = []

        for name, restaurant_info in restaurants.items():
            print(f"  📞 Querying {name}...")
            response = await self.query_restaurant(restaurant_info, query)
            results.append(f"**{name}**: {response}")
            print(f"  ✅ Response from {name}")

        return "\n".join(results)

    async def book_at_restaurant(
        self, restaurant_name, date, time, party_size, customer_name
    ):
        """Book at a specific restaurant via A2A"""
        print(f"\n📅 Attempting to book at {restaurant_name}")
        print("-" * 60)

        restaurants = await self.discover_restaurants()

        if restaurant_name not in restaurants:
            return f"❌ Restaurant {restaurant_name} is not available."

        query = f"Book a table for {party_size} people on {date} at {time} for {customer_name}"

        print(f"  📞 Sending booking request to {restaurant_name}...")
        response = await self.query_restaurant(restaurants[restaurant_name], query)
        print(f"  ✅ Booking response received")

        return response


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
    print("  • This Coordinator - Manual A2A Client")
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
    print("\n📋 Demo 1: Check availability at all restaurants")
    availability_result = await coordinator.check_all_availability(
        "2025-07-25", "19:00", 4
    )
    print("\n📊 Results:")
    print(availability_result)

    # Demo 2: Book at specific restaurant
    print("\n\n📋 Demo 2: Book at Sushi Maru")
    booking_result = await coordinator.book_at_restaurant(
        "Sushi Maru", "2025-07-25", "20:00", 4, "John Smith"
    )
    print("\n📊 Booking Result:")
    print(booking_result)

    # Interactive mode
    print("\n\n🎯 INTERACTIVE MODE")
    print("=" * 60)
    print("Commands:")
    print("  check YYYY-MM-DD HH:MM party_size")
    print("  book RestaurantName YYYY-MM-DD HH:MM party_size CustomerName")
    print("  quit")
    print()

    while True:
        try:
            user_input = input("👤 Command: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                break

            if user_input.startswith("check "):
                parts = user_input.split(" ", 3)
                if len(parts) == 4:
                    _, date, time, party_size = parts
                    result = await coordinator.check_all_availability(
                        date, time, int(party_size)
                    )
                    print(f"\n📊 Results:\n{result}\n")
                else:
                    print("❌ Usage: check YYYY-MM-DD HH:MM party_size\n")

            elif user_input.startswith("book "):
                parts = user_input.split(" ", 5)
                if len(parts) == 6:
                    _, restaurant, date, time, party_size, customer_name = parts
                    result = await coordinator.book_at_restaurant(
                        restaurant, date, time, int(party_size), customer_name
                    )
                    print(f"\n📊 Booking result:\n{result}\n")
                else:
                    print(
                        "❌ Usage: book RestaurantName YYYY-MM-DD HH:MM party_size CustomerName\n"
                    )
                    # Show available restaurants
                    fresh_restaurants = await coordinator.discover_restaurants()
                    if fresh_restaurants:
                        print(
                            "Available restaurants:",
                            ", ".join(fresh_restaurants.keys()),
                        )

            else:
                print("❌ Unknown command. Use 'check', 'book', or 'quit'\n")

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
