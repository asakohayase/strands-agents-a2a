from strands import tool
from uuid import uuid4
import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest
import logging

logger = logging.getLogger(__name__)

# Global coordinator instance for tools
coordinator_instance = None

def set_coordinator(coordinator):
    """Set the coordinator instance for tools to use"""
    global coordinator_instance
    coordinator_instance = coordinator

@tool
async def check_availability(date: str, time: str, party_size: int) -> str:
    """Check availability at all restaurants for the given date, time, and party size
    
    Args:
        date: Date in YYYY-MM-DD format (e.g., "2025-07-25")
        time: Time in HH:MM 24-hour format (e.g., "19:00")
        party_size: Number of people for the reservation
    
    Returns:
        Availability status from all restaurants
    """
    print(f"\n🔍 Checking availability for {party_size} people on {date} at {time}")
    print("-" * 60)

    restaurants = await coordinator_instance.discover_restaurants()

    if not restaurants:
        return "❌ No restaurants are currently available."

    query = f"Check availability for {party_size} people on {date} at {time}"
    results = []

    for name, restaurant_info in restaurants.items():
        print(f"  📞 Querying {name}...")
        response = await coordinator_instance.query_restaurant(restaurant_info, query)
        results.append(f"**{name}**: {response}")
        print(f"  ✅ Response from {name}")

    return "\n".join(results)

@tool
async def book_restaurant(restaurant_name: str, date: str, time: str, party_size: int, customer_name: str) -> str:
    """Book a table at a specific restaurant
    
    Args:
        restaurant_name: Name of restaurant ("Sushi Maru", "Tokyo Ramen", or "Takoyaki Taro")
        date: Date in YYYY-MM-DD format (e.g., "2025-07-25")
        time: Time in HH:MM 24-hour format (e.g., "19:00")
        party_size: Number of people for the reservation
        customer_name: Full name for the reservation
    
    Returns:
        Booking confirmation or error message
    """
    print(f"\n📅 Attempting to book at {restaurant_name}")
    print("-" * 60)

    restaurants = await coordinator_instance.discover_restaurants()

    if restaurant_name not in restaurants:
        return f"❌ Restaurant {restaurant_name} is not available."

    query = f"Book a table for {party_size} people on {date} at {time} for {customer_name}"

    print(f"  📞 Sending booking request to {restaurant_name}...")
    response = await coordinator_instance.query_restaurant(restaurants[restaurant_name], query)
    print(f"  ✅ Booking response received")

    return response

@tool
async def cancel_booking(restaurant_name: str, date: str, time: str, customer_name: str) -> str:
    """Cancel a booking at a specific restaurant
    
    Args:
        restaurant_name: Name of restaurant ("Sushi Maru", "Tokyo Ramen", or "Takoyaki Taro")
        date: Date in YYYY-MM-DD format (e.g., "2025-07-25")
        time: Time in HH:MM 24-hour format (e.g., "19:00")
        customer_name: Full name for the reservation to cancel
    
    Returns:
        Cancellation confirmation or error message
    """
    restaurants = await coordinator_instance.discover_restaurants()
    if restaurant_name in restaurants:
        query = f"Cancel booking for {customer_name} on {date} at {time}"
        return await coordinator_instance.query_restaurant(restaurants[restaurant_name], query)
    return f"❌ Restaurant {restaurant_name} is not available."