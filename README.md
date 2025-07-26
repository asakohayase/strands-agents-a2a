# **A2A Restaurant Booking Coordinator** 🍽️

### Strands Agents SDK Demo: Agent-to-Agent (A2A) Communication

A demonstration project showcasing **Agent-to-Agent (A2A) protocol** through a cross-framework restaurant booking system. This project uses the Strands Agents SDK as the coordinator while communicating with agents built on different frameworks via the standardized A2A protocol. This project implements a coordinator agent that communicates with restaurant agents built on different frameworks (Strands, LangGraph, Google ADK) using the standardized A2A protocol.

## **1. Agent Architecture Diagram**

The system demonstrates true cross-framework agent communication:
- **Coordinator Agent**: Strands-powered with intelligent tool selection
- **Sushi Maru**: Strands Agents
- **Tokyo Ramen**: LangGraph + Gemini
- **Takoyaki Taro**: Google ADK + Gemini

## **2. A2A Protocol Overview**

### **2-1. Architecture & Key Concepts**
Step1: Agent Discovery

<img width="823" height="391" alt="Image" src="https://github.com/user-attachments/assets/0e11fef0-7561-4c45-887b-e9130ccd912e" />

Step2: Task Execution

<img width="829" height="400" alt="Image" src="https://github.com/user-attachments/assets/c6f52818-2700-4d19-b2ac-f4e43b29d3e7" />

**Agent Cards**: Self-describing metadata that agents expose at `/.well-known/agent.json`:
```python
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
```

**Agent Executor**: The runtime component that:
- Processes incoming A2A messages
- Routes requests to appropriate agent skills/tools
- Handles message serialization/deserialization
- Manages agent lifecycle and state

**Communication Transport**: A2A uses **HTTP/HTTPS** as the transport layer:
- RESTful endpoints for agent discovery (`/.well-known/agent.json`)
- POST requests for message exchange
- Standard HTTP status codes for error handling
- JSON payload format for cross-platform compatibility

**Agent Discovery**: Agents discover each other through:
- Known endpoint URLs (as in this demo)
- Service discovery mechanisms
- Agent card metadata exchange
- Network topology awareness

### **2-2. Message Protocol**

Standardized request/response format across all frameworks:
```python
request = SendMessageRequest(
    id=str(uuid4()),
    params=MessageSendParams(
        message={
            "role": "user",
            "parts": [{"kind": "text", "text": "Check availability for 4 people"}],
            "messageId": uuid4().hex,
        }
    ),
)
```

### **2-3. A2A Protocol Implementation**

**A2A Server (Strands Implementation)**:
```python
from strands.multiagent.a2a import A2AServer

# Strands agents can be exposed via A2A protocol
strands_agent = Agent(
    name="Restaurant Agent",
    tools=[check_availability, book_table]
)

# A2AServer automatically:
# 1. Generates agent card from agent metadata
# 2. Exposes /.well-known/agent.json endpoint
# 3. Handles A2A message protocol conversion
# 4. Routes messages to agent tools
a2a_server = A2AServer(
    agent=strands_agent,
    host="0.0.0.0", 
    port=9001
)
```

**A2A Client (Standard Implementation)**:
```python
from a2a.client import A2ACardResolver, A2AClient

# 1. Discover agent capabilities
resolver = A2ACardResolver(httpx_client=client, base_url=url)
agent_card = await resolver.get_agent_card()

# 2. Create client with discovered capabilities  
a2a_client = A2AClient(httpx_client=client, agent_card=agent_card)

# 3. Send standardized messages
response = await a2a_client.send_message(request)
```

## **3. Core Implementation**

### **3-1. Restaurant Booking Coordinator (A2A Client)**
The main coordinator agent uses A2A protocol to communicate with restaurant agents:

```python
from strands import Agent
from tools.restaurant_tools import check_availability, book_restaurant, cancel_booking

class RestaurantBookingCoordinator:
    def __init__(self):
        self.restaurants = {
            "Sushi Maru": "http://localhost:9001",
            "Tokyo Ramen": "http://localhost:9002", 
            "Takoyaki Taro": "http://localhost:9003",
        }
        self.agent = Agent(
            name="Restaurant Booking Coordinator",
            system_prompt="""You are a restaurant booking coordinator that helps users check availability, make reservations, and cancel bookings across multiple restaurants using A2A protocol...""",
            tools=[check_availability, book_restaurant, cancel_booking],
        )
```

### **3-2. Restaurant Agents (A2A Servers)**
Each restaurant agent exposes A2A endpoints for cross-framework communication:

```python
from strands import Agent, tool
from strands.multiagent.a2a import A2AServer

@tool
def check_availability(date: str, time: str, party_size: int) -> str:
    """Check if a table is available for the given date, time, and party size"""
    # Database query logic
    return f"Available: Table for {party_size} people on {date} at {time}"

@tool  
def book_table(date: str, time: str, party_size: int, customer_name: str) -> str:
    """Book a table for the given date, time, and party size"""
    # Database booking logic
    return f"Booking confirmed for {customer_name}, {party_size} people on {date} at {time}"

# Create Strands agent
strands_agent = Agent(
    name="Sushi Maru Restaurant Agent",
    description="Restaurant booking agent for Sushi Maru",
    tools=[check_availability, book_table],
)

# Expose via A2A protocol
a2a_server = A2AServer(
    agent=strands_agent,
    host="0.0.0.0",
    port=9001,
    version="0.0.1",
)
```

### **3-3. Tool-Based Implementation**
The coordinator uses Strands tools that contain the actual A2A communication logic:

```python
# In tools/restaurant_tools.py
@tool
async def check_availability(date: str, time: str, party_size: int) -> str:
    """Check availability at all restaurants for the given date, time, and party size"""
    restaurants = await coordinator_instance.discover_restaurants()
    query = f"Check availability for {party_size} people on {date} at {time}"
    results = []
    
    for name, restaurant_info in restaurants.items():
        response = await coordinator_instance.query_restaurant(restaurant_info, query)
        results.append(f"**{name}**: {response}")
    
    return "\n".join(results)

# Agent with system prompt and tools
agent = Agent(
    name="Restaurant Booking Coordinator",
    system_prompt="""You are a restaurant booking coordinator...""",
    tools=[check_availability, book_restaurant, cancel_booking]
)
```

### **3-4. Key A2A Protocol Features Demonstrated**

- **Cross-Framework Communication**: Agents built on different frameworks (Strands, LangGraph, Google ADK) communicate seamlessly
- **Standardized A2A Protocol**: Consistent interface regardless of underlying agent framework
- **Agent Card Retrieval**: Fetch agent capabilities from known endpoints
- **Tool-Based Intelligence**: Strands coordinator agent intelligently selects appropriate tools

## **4. Current Implementation vs Ideal Strands A2A Tools**

### **4-1. Current Approach (Low-Level A2A Client)**
Due to `strands-agents-tools[a2a-client]` not being available in the current package version, we use the low-level A2A client:

```python
# Complex payload construction required
from a2a.client import A2AClient
from a2a.types import MessageSendParams, SendMessageRequest

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
```

### **4-2. Ideal Approach (Strands A2A Tools)**
When `strands_agents_tools.a2a_client` becomes available, the implementation would be much simpler:

```python
# Simple, high-level tools (not currently available)
from strands_agents_tools.a2a_client import A2AClientToolProvider

provider = A2AClientToolProvider(known_agent_urls=[
    "http://localhost:9001",
    "http://localhost:9002", 
    "http://localhost:9003"
])

coordinator = Agent(
    name="Restaurant Booking Coordinator",
    tools=provider.tools,  # Automatic A2A tools generation
)

# Natural usage - no complex payload construction needed
response = await coordinator.invoke_async(
    "Send message to agent at http://localhost:9001: Check availability for 4 people on 2025-07-25 at 19:00"
)
```

**Benefits of Strands A2A Tools (when available):**
- **Automatic tool generation** from agent URLs
- **Simplified payload handling** - no manual SendMessageRequest construction
- **Built-in error handling** and response parsing
- **Native Strands integration** with proper tool descriptions

## **5. Quick Start**

### **Prerequisites**
- Python 3.11+
- uv package manager: `pip install uv`
- AWS credentials for Strands Agents SDK
- Enable the model you want to use (default is Claude Sonnet 4) in AWS Bedrock
- Google AI API key for LangGraph and Google ADK agents

### **Installation**

1. **Clone and setup:**
```bash
git clone https://github.com/asakohayase/strands-agents-a2a.git
cd strands-agents-a2a
```

2. **Create .env file:**
```bash
# AWS credentials for Strands Agents
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key

# Google AI API key for LangGraph and Google ADK agents
GOOGLE_API_KEY=your_google_api_key
```

3. **Install dependencies:**
```bash
uv install
```

4. **Initialize databases:**
```bash
uv run setup_databases.py
```

5. **Start restaurant agents (3 separate terminals):**
```bash
# Terminal 1 - Sushi Maru (Strands)
uv run sushi_maru_agent.py

# Terminal 2 - Tokyo Ramen (LangGraph)  
uv run tokyo_ramen_agent.py

# Terminal 3 - Takoyaki Taro (Google ADK)
uv run takoyaki_taro_agent.py
```

6. **Run coordinator:**
```bash
# Terminal 4
uv run customer_coordinator.py
```

## **6. Usage Examples**

The system supports natural language requests:

```
👤 Request: Check availability for 4 people on 2025-07-25 at 19:00
👤 Request: Book a table at Sushi Maru for 2 people on 2025-07-25 at 20:00 for John Smith
👤 Request: Is Tokyo Ramen available for 6 guests on 2025-07-26 at 18:30?
👤 Request: Cancel booking at Sushi Maru for John Smith on 2025-07-25 at 20:00
```

## **7. Extending the Demo**

Additional features that can be added:
- **Payment Processing** - integrate payment agents via A2A
- **Notification System** - SMS/email confirmation agents
- **Review System** - feedback collection agents
- **Multi-language Support** - translation agents

Other enhancements:
- **Persistent Sessions** - maintain conversation context across requests
- **Agent Load Balancing** - distribute requests across multiple instances
- **Monitoring Dashboard** - track A2A communication metrics

## **8. Links**
- **A2A Protocol Official Documentation:** https://a2a-protocol.org/latest/
- **Strands Agents SDK:** https://strandsagents.com/
- **Strands A2A Implementation:** https://strandsagents.com/1.0.x/documentation/docs/user-guide/concepts/multi-agent/agent-to-agent/
- **Multi-Agent Examples:** https://github.com/strands-agents/
