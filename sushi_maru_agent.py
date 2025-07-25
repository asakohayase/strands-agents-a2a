import sqlite3
import logging
from strands import Agent, tool
from strands.multiagent.a2a import A2AServer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_database():
    """Initialize the Sushi Maru database with bookings table."""
    conn = sqlite3.connect("sushi_maru.db")

    # Create bookings table if it doesn't exist
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bookings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            party_size INTEGER NOT NULL,
            customer_name TEXT NOT NULL,
            status TEXT DEFAULT 'confirmed',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # Add some sample data for testing (only if table is empty)
    cursor = conn.execute("SELECT COUNT(*) FROM bookings")
    if cursor.fetchone()[0] == 0:
        sample_bookings = [
            ("2025-07-24", "18:00", 2, "Sample Customer 1", "confirmed"),
            ("2025-07-24", "20:00", 4, "Sample Customer 2", "confirmed"),
        ]
        conn.executemany(
            "INSERT INTO bookings (date, time, party_size, customer_name, status) VALUES (?, ?, ?, ?, ?)",
            sample_bookings,
        )
        logger.info("Added sample booking data")

    conn.commit()
    conn.close()
    logger.info("✅ Sushi Maru database initialized")


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
            return f"🍣 Available: Table for {party_size} people on {date} at {time} at Sushi Maru"
        else:
            return (
                f"❌ Not available: Already booked for {date} at {time} at Sushi Maru"
            )
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
        conn = sqlite3.connect("sushi_maru.db")

        # First check if already booked
        cursor = conn.execute(
            "SELECT COUNT(*) FROM bookings WHERE date = ? AND time = ? AND status = 'confirmed'",
            (date, time),
        )
        existing_bookings = cursor.fetchone()[0]

        if existing_bookings > 0:
            conn.close()
            return f"❌ Booking failed: Time slot {date} at {time} already taken at Sushi Maru"

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


def main():
    """Main function to start the Sushi Maru A2A server."""

    # Initialize database
    init_database()

    # Create the Sushi Maru agent
    strands_agent = Agent(
        name="Sushi Maru Restaurant Agent",
        description="Restaurant booking agent for Sushi Maru, specializing in authentic Japanese sushi",
        tools=[check_availability, book_table],
        callback_handler=None,
    )

    # Create A2A server with correct parameters (according to Strands docs)
    a2a_server = A2AServer(
        agent=strands_agent,
        host="0.0.0.0",  # Host parameter in constructor
        port=9001,  # Port parameter in constructor
        version="0.0.1",  # Optional version
    )

    print("🍣 Starting Sushi Maru A2A agent on port 9001...")
    print("🔍 Agent Card available at: http://localhost:9001/.well-known/agent.json")
    print("📋 Agent capabilities:")
    print("   - Check table availability")
    print("   - Book tables")
    print("   - Respond to A2A protocol requests")
    print("🛠️  Framework: Strands Agents + A2A Protocol")
    print("💾 Database: SQLite (sushi_maru.db)")
    print("\nPress Ctrl+C to stop the server")

    try:
        # Simple serve() call - host and port are set in constructor
        a2a_server.serve()
    except KeyboardInterrupt:
        print("\n🍣 Sushi Maru agent stopped")
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        print(f"\n❌ Failed to start Sushi Maru agent: {e}")


if __name__ == "__main__":
    main()
