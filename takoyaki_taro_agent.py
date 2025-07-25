import logging
import sqlite3
from typing import List, AsyncGenerator

from google.adk import Runner
from google.adk.agents import LlmAgent
from google.adk.events import Event
from google.adk.sessions import InMemorySessionService
from google.genai import types

# A2A SDK imports
from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    TaskState,
    TextPart,
    Part,
    FilePart,
    FileWithBytes,
    FileWithUri,
    UnsupportedOperationError,
)
from a2a.utils.errors import ServerError
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

import uvicorn

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def check_availability(date: str, time: str, party_size: int) -> str:
    """
    Check if a table is available for the given date, time, and party size.

    Args:
        date: Date in YYYY-MM-DD format (e.g., "2025-07-25")
        time: Time in HH:MM format (e.g., "19:00")
        party_size: Number of people for the reservation

    Returns:
        Availability status for the requested date and time
    """
    try:
        conn = sqlite3.connect("takoyaki_taro.db")
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

        # Simple availability logic (max 2 tables at same time)
        if existing_bookings >= 2:
            return f"Sorry, no tables available on {date} at {time} for {party_size} people. Our takoyaki is popular!"
        else:
            return f"Excellent! We have tables available on {date} at {time} for {party_size} people. Perfect for enjoying our famous takoyaki!"

    except Exception as e:
        return f"Error checking availability: {str(e)}"


def book_table(date: str, time: str, party_size: int, customer_name: str) -> str:
    """
    Book a table for the given date, time, and party size.

    Args:
        date: Date in YYYY-MM-DD format (e.g., "2025-07-25")
        time: Time in HH:MM format (e.g., "19:00")
        party_size: Number of people for the reservation
        customer_name: Name for the reservation

    Returns:
        Booking confirmation or error message
    """
    try:
        conn = sqlite3.connect("takoyaki_taro.db")
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

        if existing_bookings >= 2:
            conn.close()
            return f"Sorry, no tables available on {date} at {time}. Our takoyaki is very popular!"

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

        return f"Booking confirmed! Reservation #{booking_id} for {customer_name} on {date} at {time} for {party_size} people at Takoyaki Taro. We can't wait to serve you our delicious takoyaki!"

    except Exception as e:
        return f"Error making booking: {str(e)}"


def create_adk_agent() -> LlmAgent:
    """Create the ADK agent for Takoyaki Taro restaurant"""
    return LlmAgent(
        model="gemini-1.5-flash",
        name="Takoyaki_Taro_Agent",
        instruction="""
            **Role:** You are Takoyaki Taro restaurant booking assistant. 
            Help customers check availability and book tables for our casual Japanese street food restaurant.
            Always be friendly and enthusiastic about our delicious takoyaki!
            
            **Core Directives:**
            *   **Check Availability:** Use the `check_availability` tool to determine 
                    if tables are available for a requested date, time, and party size.
            *   **Book Tables:** Use the `book_table` tool to make reservations.
                    Always ask for customer name when booking a table.
            *   **Polite and Enthusiastic:** Be friendly and excited about takoyaki!
            *   **Stick to Your Role:** Focus on restaurant bookings and takoyaki enthusiasm.
        """,
        tools=[check_availability, book_table],
    )


class TakoyakiTaroAgentExecutor(AgentExecutor):
    """
    An AgentExecutor that runs Takoyaki Taro's ADK-based Agent.

    This class bridges the A2A protocol with Google ADK framework:
    1. Receives A2A RequestContext from client
    2. Converts A2A Parts to ADK format
    3. Runs ADK agent with session management
    4. Converts ADK responses back to A2A format
    5. Updates task status through A2A TaskUpdater
    """

    def __init__(self, runner: Runner):
        """
        Initialize with ADK Runner

        Args:
            runner: Google ADK Runner instance that manages the agent and sessions
        """
        self.runner = runner
        self._running_sessions = {}

    def _run_agent(
        self, session_id: str, new_message: types.Content
    ) -> AsyncGenerator[Event, None]:
        """
        Run the ADK agent asynchronously and yield events

        Args:
            session_id: Unique session identifier for conversation continuity
            new_message: User message in ADK format

        Returns:
            AsyncGenerator of ADK Events (streaming responses)
        """
        return self.runner.run_async(
            session_id=session_id, user_id="takoyaki_agent", new_message=new_message
        )

    async def _process_request(
        self,
        new_message: types.Content,
        session_id: str,
        task_updater: TaskUpdater,
    ) -> None:
        """
        Process the ADK agent request and manage task updates

        Args:
            new_message: ADK-formatted user message
            session_id: Session ID for conversation continuity
            task_updater: A2A TaskUpdater for managing task lifecycle
        """
        # Ensure session exists (create if needed)
        session_obj = await self._upsert_session(session_id)
        session_id = session_obj.id

        # Run ADK agent and process streaming events
        async for event in self._run_agent(session_id, new_message):
            if event.is_final_response():
                # Final response - convert to A2A format and complete task
                parts = convert_genai_parts_to_a2a(
                    event.content.parts if event.content and event.content.parts else []
                )
                logger.debug(
                    f"Final response with {len(parts)} parts: {[p.root.text if isinstance(p.root, TextPart) else 'non-text' for p in parts]}"
                )
                task_updater.add_artifact(parts)
                task_updater.complete()
                break
            elif not event.get_function_calls():
                # Intermediate response - update task status
                logger.debug("Intermediate response - updating task status")
                task_updater.update_status(
                    TaskState.working,
                    message=task_updater.new_agent_message(
                        convert_genai_parts_to_a2a(
                            event.content.parts
                            if event.content and event.content.parts
                            else []
                        ),
                    ),
                )
            else:
                # Function call event - skip (internal ADK processing)
                logger.debug("Function call event - skipping")

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ):
        """
        Main A2A execution method - called by A2A framework

        Args:
            context: A2A RequestContext containing task_id, context_id, and message
            event_queue: A2A EventQueue for sending task updates back to client
        """
        # Validate required context fields
        if not context.task_id or not context.context_id:
            logger.error(
                f"Missing required context fields - task_id: {context.task_id}, context_id: {context.context_id}"
            )
            raise ValueError("RequestContext must have task_id and context_id")
        if not context.message:
            logger.error("RequestContext missing message")
            raise ValueError("RequestContext must have a message")

        logger.info(
            f"Starting execution for task {context.task_id} in context {context.context_id}"
        )

        # Create TaskUpdater for managing A2A task lifecycle
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)

        # Initialize task if not already created
        if not context.current_task:
            logger.debug("Submitting new task")
            updater.submit()

        # Mark task as actively being worked on
        updater.start_work()

        try:
            # Convert A2A message parts to ADK format and process
            adk_content = types.UserContent(
                parts=convert_a2a_parts_to_genai(context.message.parts),
            )

            logger.debug(
                f"Converted {len(context.message.parts)} A2A parts to {len(adk_content.parts)} ADK parts"
            )

            await self._process_request(
                adk_content,
                context.context_id,  # Use context_id as session_id for conversation continuity
                updater,
            )

        except Exception as e:
            logger.error(f"Error during execution: {e}", exc_info=True)
            updater.fail(f"Agent execution error: {str(e)}")

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        """
        Handle task cancellation requests

        Args:
            context: A2A RequestContext
            event_queue: A2A EventQueue for sending cancellation status
        """
        logger.info(f"Cancellation requested for task {context.task_id}")

        # A2A protocol requires this to be implemented, but ADK doesn't support cancellation
        # So we raise an UnsupportedOperationError as per A2A spec
        raise ServerError(
            error=UnsupportedOperationError(
                "Task cancellation not supported by ADK agent"
            )
        )

    async def _upsert_session(self, session_id: str):
        """
        Get existing session or create new one for conversation continuity

        Args:
            session_id: Unique session identifier

        Returns:
            ADK Session object

        Raises:
            RuntimeError: If session creation/retrieval fails
        """
        # Try to get existing session
        session = await self.runner.session_service.get_session(
            app_name=self.runner.app_name,
            user_id="takoyaki_agent",
            session_id=session_id,
        )

        # Create new session if doesn't exist
        if session is None:
            logger.debug(f"Creating new session: {session_id}")
            session = await self.runner.session_service.create_session(
                app_name=self.runner.app_name,
                user_id="takoyaki_agent",
                session_id=session_id,
            )
        else:
            logger.debug(f"Using existing session: {session_id}")

        # Validate session creation/retrieval
        if session is None:
            raise RuntimeError(f"Failed to get or create session: {session_id}")

        return session


def convert_a2a_parts_to_genai(parts: List[Part]) -> List[types.Part]:
    """
    Convert A2A Part types to Google GenAI Part types

    This enables cross-framework communication by translating A2A's standardized
    Part format into ADK's expected format.

    Args:
        parts: List of A2A Part objects

    Returns:
        List of Google GenAI Part objects
    """
    return [convert_a2a_part_to_genai(part) for part in parts]


def convert_a2a_part_to_genai(part: Part) -> types.Part:
    """
    Convert single A2A Part to Google GenAI Part

    Handles different content types:
    - TextPart: Plain text content
    - FilePart with URI: Referenced file
    - FilePart with bytes: Inline file data

    Args:
        part: A2A Part object

    Returns:
        Google GenAI Part object

    Raises:
        ValueError: If part type is unsupported
    """
    root = part.root

    if isinstance(root, TextPart):
        return types.Part(text=root.text)

    if isinstance(root, FilePart):
        if isinstance(root.file, FileWithUri):
            return types.Part(
                file_data=types.FileData(
                    file_uri=root.file.uri, mime_type=root.file.mimeType
                )
            )
        if isinstance(root.file, FileWithBytes):
            return types.Part(
                inline_data=types.Blob(
                    data=root.file.bytes.encode("utf-8"),
                    mime_type=root.file.mimeType or "application/octet-stream",
                )
            )
        raise ValueError(f"Unsupported file type: {type(root.file)}")

    raise ValueError(f"Unsupported part type: {type(part)}")


def convert_genai_parts_to_a2a(parts: List[types.Part]) -> List[Part]:
    """
    Convert Google GenAI Parts back to A2A Parts

    Args:
        parts: List of Google GenAI Part objects

    Returns:
        List of A2A Part objects (filters out empty parts)
    """
    return [
        convert_genai_part_to_a2a(part)
        for part in parts
        if (part.text or part.file_data or part.inline_data)
    ]


def convert_genai_part_to_a2a(part: types.Part) -> Part:
    """
    Convert single Google GenAI Part to A2A Part

    Args:
        part: Google GenAI Part object

    Returns:
        A2A Part object

    Raises:
        ValueError: If part type is unsupported or data is missing
    """
    if part.text:
        return Part(root=TextPart(text=part.text))

    if part.file_data:
        if not part.file_data.file_uri:
            raise ValueError("File URI is missing")
        return Part(
            root=FilePart(
                file=FileWithUri(
                    uri=part.file_data.file_uri,
                    mimeType=part.file_data.mime_type,
                )
            )
        )

    if part.inline_data:
        if not part.inline_data.data:
            raise ValueError("Inline data is missing")
        return Part(
            root=FilePart(
                file=FileWithBytes(
                    bytes=part.inline_data.data.decode("utf-8"),
                    mimeType=part.inline_data.mime_type,
                )
            )
        )

    raise ValueError(f"Unsupported part type: {part}")


def create_database():
    """Create the restaurant database if it doesn't exist"""
    conn = sqlite3.connect("takoyaki_taro.db")
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


def main():
    """Main function to start the ADK A2A server"""

    # Create database
    create_database()

    # Create ADK agent and runner with session service
    adk_agent = create_adk_agent()
    session_service = InMemorySessionService()
    runner = Runner(
        agent=adk_agent, app_name="takoyaki_taro", session_service=session_service
    )

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
        name="Takoyaki Taro Restaurant Agent",
        description="Restaurant booking agent for Takoyaki Taro, casual Japanese street food dining",
        url="http://localhost:9003",
        version="1.0.0",
        defaultInputModes=["text"],
        defaultOutputModes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[check_skill, book_skill],
    )

    # Create request handler with ADK executor
    request_handler = DefaultRequestHandler(
        agent_executor=TakoyakiTaroAgentExecutor(runner), task_store=InMemoryTaskStore()
    )

    # Create A2A server application
    app = A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)

    print("🐙 Starting Takoyaki Taro A2A Server on http://localhost:9003")
    print("🔍 Agent Card available at: http://localhost:9003/.well-known/agent.json")
    print("📋 Skills: check_availability, book_table")
    print("🛠️  Framework: Google ADK + A2A Protocol")

    # Start the server
    uvicorn.run(app.build(), host="0.0.0.0", port=9003)


if __name__ == "__main__":
    main()
