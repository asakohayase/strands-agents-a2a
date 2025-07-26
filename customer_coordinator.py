import asyncio
import logging
from uuid import uuid4
import httpx
import json
from strands import Agent, tool
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
        self.agent = Agent(
            name="Restaurant Booking Coordinator",
            description="Coordinates restaurant bookings and parses user requests",
            tools=[],
            callback_handler=None,
        )

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
                                    if hasattr(part, "root") and hasattr(part.root, "text"):
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

    async def parse_natural_language(self, user_input):
        """Use coordinator's LLM to parse natural language input"""
        prompt = f"""Parse this restaurant booking request and return ONLY a JSON object:
{{
  "intent": "check" or "book" or "cancel",
  "restaurant": one of ["Sushi Maru", "Tokyo Ramen", "Takoyaki Taro"] or null,
  "date": "YYYY-MM-DD" format or null,
  "time": "HH:MM" in 24-hour format (convert AM/PM: 7:00 PM = 19:00, 7:00 AM = 07:00) or null,
  "party_size": number or null,
  "customer_name": string or null
}}

User request: "{user_input}"

IMPORTANT: Convert all times to 24-hour format. Examples:
- "7:00 PM" → "19:00"
- "7:00 AM" → "07:00"
- "12:30 PM" → "12:30"
- "12:30 AM" → "00:30"

Return only valid JSON, no other text."""
        
        try:
            response = await self.agent.invoke_async(prompt)
            # Convert AgentResult to string
            response_text = str(response)
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                parsed = json.loads(json_str)
                
                # Fix time conversion if needed
                if parsed.get('time'):
                    import re
                    # Check if original input had PM/AM that wasn't converted properly
                    if 'pm' in user_input.lower() or 'p.m.' in user_input.lower():
                        time_match = re.search(r'(\d{1,2}):(\d{2})\s*(?:pm|p\.m\.)', user_input.lower())
                        if time_match:
                            hour = int(time_match.group(1))
                            minute = time_match.group(2)
                            if hour != 12:
                                hour += 12
                            parsed['time'] = f"{hour:02d}:{minute}"
                    elif 'am' in user_input.lower() or 'a.m.' in user_input.lower():
                        time_match = re.search(r'(\d{1,2}):(\d{2})\s*(?:am|a\.m\.)', user_input.lower())
                        if time_match:
                            hour = int(time_match.group(1))
                            minute = time_match.group(2)
                            if hour == 12:
                                hour = 0
                            parsed['time'] = f"{hour:02d}:{minute}"
                
                return parsed
        except Exception as e:
            logger.error(f"LLM parsing failed: {e}")
        
        return None


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
    print("Natural language examples:")
    print("  'Check availability for 4 people on 2025-07-25 at 19:00'")
    print("  'Book a table at Sushi Maru for 2 people on 2025-07-25 at 20:00 for John Smith'")
    print("  'Is Tokyo Ramen available for 6 guests on 2025-07-26 at 18:30?'")
    print("  'Make a reservation at Takoyaki Taro for 3 people on 2025-07-27 at 19:15 under Jane Doe'")
    print("  'Cancel booking at Sushi Maru for John Smith on 2025-07-25 at 20:00'")
    print("  'quit' to exit")
    print()
    
    while True:
        try:
            user_input = input("👤 Request: ").strip()
            
            if user_input.lower() in ["quit", "exit", "q"]:
                break
            
            # Parse natural language input
            parsed = await coordinator.parse_natural_language(user_input)
            
            if not parsed:
                print("❌ I didn't understand that. Please try again or type 'quit' to exit.\n")
                continue
            
            if parsed['intent'] == 'check':
                if not all([parsed['date'], parsed['time'], parsed['party_size']]):
                    print("❌ Please specify date (YYYY-MM-DD), time (HH:MM), and party size.\n")
                    continue
                
                if parsed['restaurant']:
                    # Check specific restaurant
                    restaurants = await coordinator.discover_restaurants()
                    if parsed['restaurant'] in restaurants:
                        query = f"Check availability for {parsed['party_size']} people on {parsed['date']} at {parsed['time']}"
                        response = await coordinator.query_restaurant(restaurants[parsed['restaurant']], query)
                        print(f"\n📊 {parsed['restaurant']}: {response}\n")
                    else:
                        print(f"❌ Restaurant {parsed['restaurant']} is not available.\n")
                else:
                    # Check all restaurants
                    result = await coordinator.check_all_availability(
                        parsed['date'], parsed['time'], parsed['party_size']
                    )
                    print(f"\n📊 Results:\n{result}\n")
            
            elif parsed['intent'] == 'book':
                if not all([parsed['restaurant'], parsed['date'], parsed['time'], parsed['party_size'], parsed['customer_name']]):
                    print("❌ Please specify restaurant, date (YYYY-MM-DD), time (HH:MM), party size, and customer name (first and last name).\n")
                    continue
                
                customer_name = parsed['customer_name']
                result = await coordinator.book_at_restaurant(
                    parsed['restaurant'], parsed['date'], parsed['time'], 
                    parsed['party_size'], customer_name
                )
                print(f"\n📊 Booking result:\n{result}\n")
            
            elif parsed['intent'] == 'cancel':
                if not all([parsed['restaurant'], parsed['date'], parsed['time'], parsed['customer_name']]):
                    print("❌ Please specify restaurant, date (YYYY-MM-DD), time (HH:MM), and customer name to cancel.\n")
                    continue
                
                restaurants = await coordinator.discover_restaurants()
                if parsed['restaurant'] in restaurants:
                    query = f"Cancel booking for {parsed['customer_name']} on {parsed['date']} at {parsed['time']}"
                    response = await coordinator.query_restaurant(restaurants[parsed['restaurant']], query)
                    print(f"\n📊 Cancellation result:\n{response}\n")
                else:
                    print(f"❌ Restaurant {parsed['restaurant']} is not available.\n")
        
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
