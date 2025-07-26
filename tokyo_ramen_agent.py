import logging
import sqlite3
import random
from collections.abc import AsyncIterable
from datetime import date, timedelta
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

# A2A SDK imports
from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Part,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils.errors import ServerError
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

import uvicorn

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

memory = MemorySaver()


def generate_restaurant_schedule() -> dict[str, list[str]]:
    """Generate Tokyo Ramen's available booking slots for the next 7 days."""
    schedule = {}
    today = date.today()

    for i in range(7):
        current_date = today + timedelta(days=i)
        date_str = current_date.strftime("%Y-%m-%d")

        # Restaurant hours: 11 AM to 10 PM, with some slots randomly unavailable
        possible_times = [f"{h:02}:00" for h in range(11, 23)]  # 11 AM to 10 PM
        # Remove 2-4 random slots to simulate existing bookings
        available_times = [t for t in possible_times if random.random() > 0.3]

        schedule[date_str] = sorted(available_times)

    return schedule


RESTAURANT_SCHEDULE = generate_restaurant_schedule()


class AvailabilityInput(BaseModel):
    """Input schema for checking table availability."""

    date: str = Field(..., description="Date in YYYY-MM-DD format (e.g., '2025-07-25')")
    time: str = Field(..., description="Time in HH:MM format (e.g., '19:00')")
    party_size: int = Field(..., description="Number of people for the reservation")


class BookingInput(BaseModel):
    """Input schema for booking a table."""

    date: str = Field(..., description="Date in YYYY-MM-DD format (e.g., '2025-07-25')")
    time: str = Field(..., description="Time in HH:MM format (e.g., '19:00')")
    party_size: int = Field(..., description="Number of people for the reservation")
    customer_name: str = Field(..., description="Name for the reservation")


@tool(args_schema=AvailabilityInput)
def check_availability(date: str, time: str, party_size: int) -> str:
    """Check if a table is available for the given date, time, and party size."""
    try:
        conn = sqlite3.connect("tokyo_ramen.db")
        cursor = conn.cursor()

        # Check for existing bookings at the same time
        cursor.execute(
            """
            SELECT COUNT(*) FROM bookings 
            WHERE date = ? AND time = ? AND status = 'confirmed'
        """,
            (date, time),
        )

        existing_bookings = cursor.fetchone()[0]
        conn.close()

        # Simple availability logic (max 3 tables at same time)
        if existing_bookings >= 3:
            return f"Sorry, no tables available on {date} at {time} for {party_size} people. Tokyo Ramen is fully booked!"
        else:
            return f"Great news! We have tables available on {date} at {time} for {party_size} people at Tokyo Ramen!"

    except Exception as e:
        return f"Error checking availability: {str(e)}"


@tool(args_schema=BookingInput)
def book_table(date: str, time: str, party_size: int, customer_name: str) -> str:
    """Book a table for the given date, time, and party size."""
    try:
        conn = sqlite3.connect("tokyo_ramen.db")
        cursor = conn.cursor()

        # Check availability first
        cursor.execute(
            """
            SELECT COUNT(*) FROM bookings 
            WHERE date = ? AND time = ? AND status = 'confirmed'
        """,
            (date, time),
        )

        existing_bookings = cursor.fetchone()[0]

        if existing_bookings >= 3:
            conn.close()
            return f"Sorry, no tables available on {date} at {time}. Tokyo Ramen is fully booked!"

        # Create booking
        cursor.execute(
            """
            INSERT INTO bookings (date, time, party_size, customer_name, status)
            VALUES (?, ?, ?, ?, 'confirmed')
        """,
            (date, time, party_size, customer_name),
        )

        booking_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return f"Booking confirmed! Reservation #{booking_id} for {customer_name} on {date} at {time} for {party_size} people at Tokyo Ramen."

    except Exception as e:
        return f"Error making booking: {str(e)}"


class ResponseFormat(BaseModel):
    """Response format for Tokyo Ramen agent."""

    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str


class TokyoRamenAgent:
    """Tokyo Ramen Agent - LangGraph-based restaurant booking assistant."""

    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    SYSTEM_INSTRUCTION = (
        "You are Tokyo Ramen restaurant booking assistant. "
        "Help customers check table availability and make reservations for our fast casual Japanese ramen restaurant. "
        "Use the 'check_availability' tool to check if tables are available for specific dates and times. "
        "Use the 'book_table' tool to make reservations (always ask for customer name first). "
        "You will be provided with the current date to help you understand relative queries like 'tomorrow' or 'next week'. "
        "Always be polite and helpful. If asked about anything other than restaurant bookings, "
        "politely state that you can only help with table reservations. "
        "Set response status to input_required if you need more information from the customer. "
        "Set response status to error if there's an error processing the request. "
        "Set response status to completed when the request is successfully handled."
    )

    def __init__(self):
        self.model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        self.tools = [check_availability, book_table]

        self.graph = create_react_agent(
            self.model,
            tools=self.tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=ResponseFormat,
        )

    def invoke(self, query: str, context_id: str) -> dict[str, Any]:
        """Synchronous invoke method."""
        config: RunnableConfig = {"configurable": {"thread_id": context_id}}
        today_str = f"Today's date is {date.today().strftime('%Y-%m-%d')}."
        augmented_query = f"{today_str}\n\nCustomer request: {query}"

        self.graph.invoke({"messages": [("user", augmented_query)]}, config)
        return self.get_agent_response(config)

    async def stream(
        self, query: str, context_id: str
    ) -> AsyncIterable[dict[str, Any]]:
        """
        Stream responses from the LangGraph agent.

        Yields dict with:
        - is_task_complete: bool
        - require_user_input: bool
        - content: str
        """
        today_str = f"Today's date is {date.today().strftime('%Y-%m-%d')}."
        augmented_query = f"{today_str}\n\nCustomer request: {query}"
        inputs = {"messages": [("user", augmented_query)]}
        config: RunnableConfig = {"configurable": {"thread_id": context_id}}

        # Stream through LangGraph execution
        for item in self.graph.stream(inputs, config, stream_mode="values"):
            message = item["messages"][-1]

            if (
                isinstance(message, AIMessage)
                and message.tool_calls
                and len(message.tool_calls) > 0
            ):
                # Agent is making tool calls
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Checking availability and processing your request...",
                }
            elif isinstance(message, ToolMessage):
                # Tool execution completed
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Processing booking information...",
                }

        # Yield final response
        yield self.get_agent_response(config)

    def get_agent_response(self, config: RunnableConfig) -> dict[str, Any]:
        """Extract final response from LangGraph state."""
        current_state = self.graph.get_state(config)
        structured_response = current_state.values.get("structured_response")

        if structured_response and isinstance(structured_response, ResponseFormat):
            if structured_response.status == "input_required":
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": structured_response.message,
                }
            if structured_response.status == "error":
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": structured_response.message,
                }
            if structured_response.status == "completed":
                return {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": structured_response.message,
                }

        # Fallback response
        return {
            "is_task_complete": False,
            "require_user_input": True,
            "content": (
                "I'm sorry, I couldn't process your request properly. "
                "Please try asking about table availability or making a reservation."
            ),
        }


class TokyoRamenAgentExecutor(AgentExecutor):
    """
    A2A AgentExecutor for Tokyo Ramen LangGraph agent.

    Bridges LangGraph agent with A2A protocol by:
    1. Converting A2A RequestContext to LangGraph query
    2. Managing A2A task lifecycle (submit → work → complete)
    3. Streaming LangGraph responses as A2A task updates
    4. Handling different task states (working, input_required, completed)
    """

    def __init__(self):
        self.agent = TokyoRamenAgent()

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """
        Main A2A execution method.

        Args:
            context: A2A RequestContext with task_id, context_id, and message
            event_queue: A2A EventQueue for sending task updates
        """
        # Validate required context
        if not context.task_id or not context.context_id:
            logger.error(
                f"Missing context fields - task_id: {context.task_id}, context_id: {context.context_id}"
            )
            raise ValueError("RequestContext must have task_id and context_id")
        if not context.message:
            logger.error("RequestContext missing message")
            raise ValueError("RequestContext must have a message")

        logger.info(
            f"Processing Tokyo Ramen request - task: {context.task_id}, context: {context.context_id}"
        )

        # Create A2A task updater
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)

        # Initialize task if needed
        if not context.current_task:
            await updater.submit()

        # Start processing
        await updater.start_work()

        # Extract user query from A2A message parts
        query = context.get_user_input()
        logger.debug(f"User query: {query}")

        try:
            # Stream responses from LangGraph agent
            async for item in self.agent.stream(query, context.context_id):
                is_task_complete = item["is_task_complete"]
                require_user_input = item["require_user_input"]
                content = item["content"]

                # Convert to A2A Parts
                parts = [Part(root=TextPart(text=content))]

                if not is_task_complete and not require_user_input:
                    # Agent is working - update status
                    logger.debug("Agent working - updating status")
                    await updater.update_status(
                        TaskState.working,
                        message=updater.new_agent_message(parts),
                    )
                elif require_user_input:
                    # Need more input from user
                    logger.debug("Requiring user input")
                    await updater.update_status(
                        TaskState.input_required,
                        message=updater.new_agent_message(parts),
                    )
                    break
                else:
                    # Task completed successfully
                    logger.info("Task completed successfully")
                    await updater.add_artifact(
                        parts,
                        name="booking_result",
                    )
                    await updater.complete()
                    break

        except Exception as e:
            logger.error(
                f"Error during Tokyo Ramen agent execution: {e}", exc_info=True
            )
            # FIXED: Graceful error handling instead of raising ServerError
            error_parts = [
                Part(root=TextPart(text=f"Error processing request: {str(e)}"))
            ]
            await updater.add_artifact(error_parts, name="error_response")
            await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Handle task cancellation - not supported by this agent."""
        logger.info(f"Cancel requested for task {context.task_id}")
        raise ServerError(
            error=UnsupportedOperationError("Task cancellation not supported")
        )


def create_database():
    """Create the restaurant database if it doesn't exist."""
    conn = sqlite3.connect("tokyo_ramen.db")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bookings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            party_size INTEGER NOT NULL,
            customer_name TEXT,
            status TEXT DEFAULT 'confirmed',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # Add sample bookings
    sample_bookings = [
        ("2025-07-25", "18:00", 2, "John Smith", "confirmed"),
        ("2025-07-25", "20:00", 4, "Alice Johnson", "confirmed"),
    ]

    conn.executemany(
        """
        INSERT OR IGNORE INTO bookings (date, time, party_size, customer_name, status)
        VALUES (?, ?, ?, ?, ?)
    """,
        sample_bookings,
    )

    conn.commit()
    conn.close()
    logger.info("Database initialized")


def main():
    """Main function to start the Tokyo Ramen A2A server."""

    # Create database
    create_database()

    # Define agent skills for A2A discovery
    check_skill = AgentSkill(
        id="check_availability",
        name="Check Table Availability",
        description="Check if tables are available for a given date and time",
        tags=["booking", "availability"],
        examples=["Check availability for Friday 7 PM for 4 people"],
    )

    book_skill = AgentSkill(
        id="book_table",
        name="Book Table",
        description="Book a table for customers",
        tags=["booking", "reservation"],
        examples=["Book a table for 4 people on Friday at 7 PM"],
    )

    # Create agent card for A2A protocol discovery
    agent_card = AgentCard(
        name="Tokyo Ramen Restaurant Agent",
        description="Restaurant booking agent for Tokyo Ramen, fast casual Japanese ramen",
        url="http://localhost:9002",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=True),  # Support streaming
        skills=[check_skill, book_skill],
    )

    # Create request handler with LangGraph executor
    request_handler = DefaultRequestHandler(
        agent_executor=TokyoRamenAgentExecutor(), task_store=InMemoryTaskStore()
    )

    # Create A2A server application
    app = A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)

    print("🍜 Starting Tokyo Ramen A2A Server on http://localhost:9002")
    print("🔍 Agent Card available at: http://localhost:9002/.well-known/agent.json")
    print("📋 Skills: check_availability, book_table")
    print("🛠️  Framework: LangGraph + A2A Protocol")
    print("💾 Database: SQLite (tokyo_ramen.db)")

    # Start the server
    uvicorn.run(app.build(), host="0.0.0.0", port=9002)


if __name__ == "__main__":
    main()
