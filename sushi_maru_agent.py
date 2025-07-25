import sqlite3
import logging
from strands import Agent, tool
from strands.multiagent.a2a import A2AServer

# Set up logging
logging.basicConfig(level=logging.INFO)


@tool
def check_availability(date: str, time: str, party_size: int) -> str:
    """Check if a table is available for the given date, time, and party size

    Args:
        date: Date in YYYY-MM-DD format (e.g., "2025-07-25")
        time: Time in HH:MM format (e.g., "19:00")
        party_size: Number of people for the reservation

    Returns:
        Availability status for the requested date and time
    """
    try:
        conn = sqlite3.connect("sushi_maru.db")
        cursor = conn.execute(
            "SELECT COUNT(*) FROM bookings WHERE date = ? AND time = ? AND status = 'confirmed'",
            (date, time),
        )
        existing_bookings = cursor.fetchone()[0]
        conn.close()

        if existing_bookings == 0:
            return f"Available: Table for {party_size} people on {date} at {time}"
        else:
            return f"Not available: Already booked for {date} at {time}"
    except Exception as e:
        return f"Error checking availability: {str(e)}"


@tool
def book_table(
    date: str, time: str, party_size: int, customer_name: str = "Customer"
) -> str:
    """Book a table for the given date, time, and party size

    Args:
        date: Date in YYYY-MM-DD format (e.g., "2025-07-25")
        time: Time in HH:MM format (e.g., "19:00")
        party_size: Number of people for the reservation
        customer_name: Name for the reservation

    Returns:
        Booking confirmation or error message
    """
    try:
        # First check if available
        conn = sqlite3.connect("sushi_maru.db")
        cursor = conn.execute(
            "SELECT COUNT(*) FROM bookings WHERE date = ? AND time = ? AND status = 'confirmed'",
            (date, time),
        )
        existing_bookings = cursor.fetchone()[0]

        if existing_bookings > 0:
            conn.close()
            return f"Booking failed: Time slot {date} at {time} already taken"

        # Book the table
        conn.execute(
            "INSERT INTO bookings (date, time, party_size, customer_name, status) VALUES (?, ?, ?, ?, 'confirmed')",
            (date, time, party_size, customer_name),
        )
        conn.commit()
        conn.close()

        return f"✅ Booking confirmed at Sushi Maru for {customer_name}, {party_size} people on {date} at {time}"
    except Exception as e:
        return f"Booking error: {str(e)}"


# Create the Sushi Maru agent
strands_agent = Agent(
    name="Sushi Maru Restaurant Agent",
    description="Restaurant booking agent for Sushi Maru, specializing in authentic Japanese sushi",
    tools=[check_availability, book_table],
    callback_handler=None,
)

# Create A2A server
a2a_server = A2AServer(agent=strands_agent)

if __name__ == "__main__":
    print("🍣 Starting Sushi Maru A2A agent on port 9001...")
    print("Agent capabilities:")
    print("- Check table availability")
    print("- Book tables")
    print("- Respond to A2A protocol requests")
    print("\nPress Ctrl+C to stop the server")

    try:
        a2a_server.serve(port=9001)
    except KeyboardInterrupt:
        print("\n🍣 Sushi Maru agent stopped")
