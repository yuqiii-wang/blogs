# Checkin App Agent Demo

This guide outlines the backend AI logic for an event check-in assistant. The goal is to handle a user query like: **"Hi, I am Alice, may i checkin?"**

## 1. System Architecture

To answer this question, the AI cannot simply hallucinate a response. It needs access to real-time data and policy documents. We will use a **ReAct (Reasoning + Acting)** approach.

### Key Components

1.  **User Context**: The state dictionary holding caller identity and session constraints (e.g., `user_email: "alice@example.com"`).
2.  **Skills**: Modular domain-specific routers. They declare their scope via semantic descriptions, allowing the Agent to dynamically load relevant sub-routines (e.g., Registration vs. Event Check-in).
3.  **Tools**: Executable functions (local Python code or remote endpoints) exposed to the LLM via explicit JSON-schema definitions for fetching structured data or performing mutations.
4.  **RAG / Knowledge Base**: Unstructured, vector-embedded documents (like Event FAQ or Code of Conduct) used for zero-shot semantic retrieval to ground the LLM's answers.
5.  **Agent / Routing Logic**: The core LLM orchestration loop. It iteratively evaluates context, schedules tool executions, and parses structural responses adhering to the ReAct framework.
6.  **MCP Server Integration**: The Model Context Protocol (MCP) provider. A standardized out-of-process component that securely exposes external enterprise capabilities (email queues, SMS gateways, external APIs) to the Agent via JSON-RPC.

## 2. Defining Tools & Skills

The agent needs tools to access information. Instead of connecting to a complex database for this demo, we will embed our "backend knowledge" directly in the Python code as dummy documents. The tools will simply "grep" (search) these documents.

### A. The Knowledge Source (Multi-Document RAG)

In a real-world scenario, you might have different rules for different events (e.g., a "Tech Conference" vs. a "Music Festival"). The AI needs to retrieve the correct policy document first.

> 💡 **Comparison with Traditional IT Dev**
>
> In traditional development, handling real-time updates (like **cancelling a performance due to bad weather**) often requires building a dedicated **Admin Dashboard** or database interface for operations staff.
>
> With an Agentic/RAG approach, this "Admin Backend" is often just the document itself. Operations staff can simply edit the text file (e.g., "Performance X is cancelled due to rain"), and the Agent immediately has access to the new information without any code changes or database migrations.

**Document 1: Tech Conference Policy**

```txt
TECH CONFERENCE 2024 - CHECK-IN PROTOCOL
----------------------------------------
1. ID REQUIREMENT: Government-issued photo ID required.
2. BAG POLICY: Laptops allowed. No large backpacks.
3. LATE ENTRY: Allowed up to 2 hours after start.
4. VIP ACCESS: Requires QR code scan + wristband.
```

**Document 2: Music Festival Policy**

```txt
SUMMER VIBES FESTIVAL - ENTRY RULES
-----------------------------------
1. ID REQUIREMENT: 21+ wristband check for alcohol areas. Ticket valid for entry.
2. BAG POLICY: Clear bags ONLY. No professional cameras.
3. RE-ENTRY: No re-entry allowed after 6 PM.
4. PROHIBITED: No outside food/drink, no umbrellas.
```

**Document 3: Registration Database**

```txt
REGISTRATION DATABASE
---------------------
1. Alice Smith (Tech Conf) - STATUS: CONFIRMED - TICKET: VIP
2. Bob Jones (Music Fest) - STATUS: CONFIRMED - TICKET: GA
```

### B. The Retrieval Tool (Finding the Right Doc via Embeddings)

Instead of hardcoding "if/else" rules, we use **Embeddings** to find the most relevant document. We convert the user's query (e.g., "Can I bring my laptop?") into a vector and compare it against our document vectors to find a match.

> 💡 In traditional code, you might write:  
> `if "laptop" in query.lower().strip(): return "Allowed"`  
> This is fragile. What if the user types "Lap top"? Or input has leading spaces? You'd need rigorous logic control (e.g., regex, ignorecase).  
> Furthermore, managing these rules usually requires a custom **Admin Backend**. With agents, the "Admin Backend" is just the text file itself. Updates are as simple as editing a doc.

```python
from typing import Annotated
import numpy as np

# Mocking an embedding function (in reality, you'd use OpenAI or HuggingFace)
def get_embedding(text):
    # Returns a random vector for demo purposes
    return np.random.rand(512)

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

class KnowledgeBaseTools:
    
    def __init__(self):
        # Index our documents with their embeddings
        self.doc_store = {
            "tech_conf": {"text": doc_tech_conf, "vector": get_embedding("tech conference business rules")},
            "music_fest": {"text": doc_music_fest, "vector": get_embedding("music festival party rules")}
        }

    @tool("get_relevant_policy")
    def get_relevant_policy(self, user_query: Annotated[str, "The user's specific question or event name"]):
        """
        Retrieves the most relevant policy document using Semantic Search (RAG).
        Use this when the user asks about rules but you don't know which document applies.
        """
        query_vector = get_embedding(user_query)
        best_score = -1
        best_doc = None
        
        # RAG Logic: Find the document with the highest similarity to the query
        for doc_id, data in self.doc_store.items():
            score = cosine_similarity(query_vector, data["vector"])
            if score > best_score:
                best_score = score
                best_doc = data["text"]
        
        # If the similarity is too low, we might return nothing (thresholding)
        if best_score < 0.5:
            return "No relevant policy found."
            
        return best_doc

    @tool("check_registration_doc")
    def check_registration_doc(self, query: Annotated[str, "The name or email to look for"]):
        """
        Searches the 'Attendee List' document for a person's record.
        Use this to find status and WHICH EVENT they are registered for.
        """
        # ... (Existing grep logic) ...
        pass
```

### C. Defining Multiple Skills (The "Brain" Modules)

Instead of one giant "Check-In" procedure, we split functionality into specialized **Skills**. The LLM acts as a **Router**: it looks at the "Description" of each skill to decide which one to load and use based on the user's first message.

#### 1. The Receptionist Skill (Registration Management)
Currently loaded when: "User asks about tickets, registration status, fees, or account details."

```markdown
---
name: Registration & Accounts
description: Use this skill when the user asks about tickets, registration status, fees, or account details.
---

# SKILL: Registration & Accounts

## GOAL

Manage attendee records and payment status.

## TOOLS

- `check_registration_doc(query)`: Search for a user's ticket.
- `check_event_capacity(event_name)`: Get current crowd level.
- `update_attendance_status(name, status)`: Update DB.
- `verify_email(email, code)`: Authentication.

## WORKFLOW

1.  **Status Checks**:
    - If asking about crowds/seats -> Call `check_event_capacity`.
    - If asking about personal ticket -> Call `check_registration_doc`.

2.  **Check-In Process**:
    - If user wants to check in -> Verify email -> Update status -> Hand off to Security.
```

#### 2. The Policy Expert Skill (RAG)

Currently loaded when: "User asks about rules, what to bring, code of conduct, or event times."

```markdown
---
name: Event Policy & Rules
description: Use this skill when the user asks about rules, what to bring, code of conduct, or event times.
---

# SKILL: Event Policy & Rules

## GOAL
Answer questions about event regulations using RAG.

## TOOLS
- `get_relevant_policy(query)`: Semantic search of rules.

## WORKFLOW
1. Identify the user's event (ask if unknown).
2. specific user query -> `get_relevant_policy`.
3. Summarize the rule clearly.
```

#### 3. The Security Officer Skill (Gatekeeper)
Currently loaded when: "User is ready to physically enter the venue or validate identity."

```markdown
---
name: Gate Security & Verification
description: Use this skill when the user is ready to physically enter the venue or validate identity.
---

# SKILL: Gate Security & Verification

## GOAL

Enforce physical entry requirements.

## TOOLS

- `verify_id_photo(photo, name)`
- `verify_location(lat, long)`
- `approve_attendance(name)` / `reject_attendance(reason)`

## WORKFLOW

1. Prerequisite: User must be "Checked In" by Registration Skill.
2. Perform ID Check.
3. Perform Location Check.
4. Grant/Deny Entry.
```

### D. The Router Logic (First Round)

When the conversation starts, the system prompt includes an **Index of Skills**. The system dynamically loads the `name` and `description` from the YAML frontmatter of each skill document.

**System Prompt:**
> You are the Event Master AI. You have access to the following skills. 
> carefully analyze the user's request and ACTIVATE the most relevant skill by calling `activate_skill(skill_name)`.
>
> **Available Skills:**
> 1.  **Registration & Accounts**: Use this skill when the user asks about tickets, registration status, fees, or account details.
> 2.  **Event Policy & Rules**: Use this skill when the user asks about rules, what to bring, code of conduct, or event times.
> 3.  **Gate Security & Verification**: Use this skill when the user is ready to physically enter the venue or validate identity.

**Example Conversation Trace:**
*   **User**: "Can I bring my dog?"
*   **AI (Router Match)**: "User is asking about rules -> Activate `policy_skill`."
*   **AI (Policy Skill)**: Calls `get_relevant_policy("pets")` -> Returns "No pets allowed."

---

### E. The Backend Tools (The "Hands")

The Python functions provide the raw capabilities referenced in the Skills above.

> **💡 Dev Tip: Documentation IS Code**
> When you register a tool, the AI framework reads your **Docstrings** and **Type Hints** to teach the LLM how to use it.
> *   The `docstring` tells the model **when** and **why** to use the tool.
> *   The `Annotated` type hints tell the model accurately **what arguments** to generate.

```python
class SecurityTools:
    
    @tool("verify_email")
    def verify_email(self, email: Annotated[str, "User's email address"], code: Annotated[str, "4-digit verification code"]):
        """
        Verifies ownership of an email address by checking a submitted code.
        (Simulated) Returns True if code is '1234'.
        """
        if code == "1234":
            return {"verified": True}
        return {"verified": False, "error": "Invalid code"}

    @tool("verify_id_photo")
    def verify_id_photo(self, photo_data: Annotated[str, "Simulated photo string"], name: str):
        """
        Analyzes a submitted ID photo to see if it matches the registered name.
        (Simulated) always returns True for demo.
        """
        return {"match": True, "confidence": 0.98}

    @tool("verify_location")
    def verify_location(self, lat: float, long: float):
        """
        Verifies if the user's phone is physically at the event venue.
        (Simulated) Checks if coordinates are within the venue geofence.
        """
        EVENT_LAT = 37.7749
        EVENT_LONG = -122.4194
        
        # Simple distance check (simulated)
        if abs(lat - EVENT_LAT) < 0.01 and abs(long - EVENT_LONG) < 0.01:
            return "inside_venue"
        return "outside_venue_perimeter"

    @tool("check_event_capacity")
    def check_event_capacity(self, event_name: str):
        """
        Returns the current crowd level and available seats.
        Use this to answer questions like 'is it crowded?' or 'are there tickets left?'.
        """
        # Simulated data
        return {
            "total_capacity": 500,
            "current_attendees": 450,
            "remaining_seats": 50,
            "status": "CROWDED"
        }

    @tool("update_attendance_status")
    def update_attendance_status(self, name: str, new_status: str):
        """
        Updates the attendance status in the database.
        Use this to mark a user as 'CHECKED_IN' or 'REFUSED_ENTRY'.
        """
        # In a real app, this would execute: UPDATE db SET status=? WHERE name=?
        return f"SUCCESS: Updated status for {name} to {new_status}."
    
    @tool("approve_attendance")
    def approve_attendance(self, name: str):
        """Grants access to the event."""
        return f"ACCESS GRANTED: Welcome, {name}!"

    @tool("reject_attendance")
    def reject_attendance(self, reason: str):
        """Denies access to the event."""
        return f"ACCESS DENIED: {reason}"
```

In this setup:
1.  **User asks**: "Can David Lee come in?"
2.  **LLM calls**: `check_registration_doc("David Lee")`
3.  **Tool returns**: `4. David Lee (david@example.com) - STATUS: CONFIRMED - TICKET: GA [FLAG: DUPLICATE]`
4.  **LLM analyzes**: "He is confirmed but flagged." -> Calls `get_event_policy("DUPLICATE")` (or appropriate RAG call)
5.  **Tool returns**: `3. FLAGS: If a registration is flagged as a 'DUPLICATE', the user may still attend...`
6.  **LLM actions**: Calls `update_attendance_status("David Lee", "CHECKED_IN")` then `approve_attendance("David Lee")`.

---

## 3. The Agent Workflow (Step-by-Step)

When the user asks **"Am I allowed to attend the event?"**, the backend logic follows this flow:

### Step 1: Input Processing & Intent Recognition
The LLM receives the prompt.
*   **System Prompt**: "You are a helpful check-in assistant. You verify eligibility based on registration status and event policies. You act on behalf of the event organizers."
*   **User Input**: "Am I allowed to attend?"
*   **Context**: `{ "current_user": "alice@example.com" }`

### Step 2: Reasoning (The "Thought")
The Agent analyzes the request. It realizes it cannot answer "Yes/No" without data.
*   *Internal Monologue*: "The user wants to know their attendance eligibility. First, I need to check if they have a valid ticket. I have the user's email from the context."

### Step 3: Tool Execution (The "Act")
The Agent selects the appropriate tool.
*   **Action**: Call `check_registration("alice@example.com")`
*   **Observation (Result)**:
    ```json
    {
      "status": "confirmed",
      "ticket_type": "general_admission",
      "flagged_for_review": true,
      "flag_reason": "potential_duplicate"
    }
    ```

### Step 4: Secondary Reasoning (Handling Complexity)
The Agent sees a flag ("potential_duplicate"). It might need to check policy.
*   *Internal Monologue*: "The user has a confirmed ticket, but it's flagged. I should check the event policy regarding duplicate registrations to see if this disqualifies them."

### Step 5: RAG Lookup (Optional but robust)
*   **Action**: Call `search_event_policy("duplicate registration policy")`
*   **Observation**: "Policy Section 4.2: Duplicate registrations solely for the purpose of holding spots are void. Accidental duplicates will be merged at check-in."

## 4. External MCP Server Integration

For production deployments, the Agent should delegate outbound notifications and third-party data lookups to an external MCP (Microservice Communication Platform) server. The MCP server centralizes integrations for email, SMS, real-time traffic, and other external channels so the Agent remains stateless and auditable.

Key responsibilities
- **Email notifications**: transactional emails (checkin confirmations, receipts), templating, retries, DKIM/SPF settings.
- **SMS messages**: OTPs, short notices (use a provider with delivery receipts and rate-limiting).
- **Real-time public traffic**: query traffic APIs (Google/HERE/TomTom) for ETA estimates and venue delays.
- **Webhook delivery**: reliable outgoing webhooks to notify other systems (gate turnstiles, CRM).

Architecture & APIs
- MCP exposes simple HTTP JSON endpoints the Agent can call as tools:
    - `POST /v1/notify/email` {to, subject, template_id, data} -> {id, status}
    - `POST /v1/notify/sms` {to, message, sender_id} -> {id, status}
    - `GET  /v1/traffic?lat={lat}&lon={lon}` -> {status, eta_minutes, incidents: []}
    - `POST /v1/webhook` {target, payload, idempotency_key} -> {id, status}

Security & operational notes
- Authenticate using API keys or mTLS; rotate keys and limit scopes per environment.
- Enforce idempotency for retries (use `idempotency_key` for notifications).
- Record delivery receipts and expose status endpoints so the Agent can check final state.
- Respect PII: avoid logging full personal data in plain text; redact if persisted.

Sample MCP Server using FastMCP
- The most popular framework in Python is the official `mcp` SDK (using `FastMCP`).
- You annotate the Python functions to be exposed as tools with `@mcp.tool()`.

```python
# Server code (mcp_server.py)
# pip install mcp
from mcp.server.fastmcp import FastMCP
import uuid

# Initialize the FastMCP server
mcp = FastMCP("Checkin Notifications")

@mcp.tool()
def send_email(to: str, subject: str, template: str, data: dict) -> dict:
    """Sends an email notification via the MCP server and returns delivery status."""
    return {"id": f"mcp-{uuid.uuid4()}", "status": "queued", "recipient": to}

@mcp.tool()
def send_sms(to: str, message: str) -> dict:
    """Sends an SMS message and returns provider response."""
    return {"id": f"mcp-{uuid.uuid4()}", "status": "queued"}

@mcp.tool()
def get_traffic_info(lat: float, lon: float) -> dict:
    """Queries for ETA and incident data based on coordinates."""
    return {"eta_minutes": 15, "incidents": []}

if __name__ == "__main__":
    # Runs the server using Standard Input/Output (stdio) for smooth Agent integration
    mcp.run()
```

Providers and tooling
- Email: SendGrid, Mailgun, Amazon SES.
- SMS: Twilio, MessageBird, Nexmo.
- Traffic: Google Traffic API, HERE, TomTom.
- Message queue: RabbitMQ, Kafka, or Cloud Pub/Sub for high-throughput dispatch.

Webhook event schema (example)

```json
{
    "event": "checkin_completed",
    "user": {"name": "Alice Smith", "email": "alice@example.com"},
    "ticket": {"type": "VIP", "id": "abc-123"},
    "timestamp": "2024-05-01T15:23:00Z"
}
```

Delivery & observability
- Expose a `/v1/status/{id}` endpoint for delivery state (queued/sent/failed).
- Emit structured logs and metrics (latency, failure-rate, retries) to monitoring (Prometheus/Grafana).

Example: sending a check-in confirmation (sequence)
1. Agent verifies registration and calls `send_email()` tool.
2. Tool calls MCP `/v1/notify/email` and returns `{id: "mcp-123", status: "queued"}`.
3. MCP dispatches to provider and updates delivery state; Agent can poll or subscribe to webhook callbacks for final delivery status.

Adding MCP support keeps the Agent focused on reasoning and policies while a hardened platform handles external channels, compliance, retries, and provider specifics.


## Step 5: Final Synthesis

The Agent combines the tool output and the policy context to generate the answer.
*   **Final Answer**: "Yes, you have a confirmed General Admission ticket. However, your account is flagged as a potential duplicate. According to our policy, this usually just means we'll need to merge your records at the check-in desk, but you are allowed to attend."

> **💡 Dev Tip: The Conversation Loop**
> The "Agent" is actually just a `while` loop that keeps feeding **tool results** back into the LLM as new messages.
> 1. User Message -> 2. LLM Tool Call -> 3. Execute Python Code -> 4. Send Tool Result Message -> 5. LLM Final Answer

Below is the **refactored all-in-one client app** (`agent_client.py`). It combines establishing the standard I/O connection to the MCP Server with running the ReAct "Agent Loop" that intelligently routes tool calls.

```python
import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# (Assume `llm`, `check_registration`, `search_event_policy`, and `execute_tool` are defined locally)
SYSTEM_PROMPT = "You are a helpful check-in assistant. Use tools if necessary."

async def run_agent(user_query, user_context, mcp_session):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"User: {user_context['email']}\nQuery: {user_query}"}
    ]
    
    # 1. Dynamically discover tools from the external MCP server
    mcp_tools_response = await mcp_session.list_tools()
    
    # Translate MCP tool schemas into the format expected by your specific LLM (e.g., OpenAI)
    mcp_tools = [format_mcp_to_llm_schema(t) for t in mcp_tools_response.tools] 
    
    # Combine your python local tools with the remote MCP tools
    all_tools = [check_registration, search_event_policy] + mcp_tools
    
    while True:
        # 2. Ask LLM what to do, passing ALL available tools
        response = await llm.chat(messages, tools=all_tools)
        
        # 3. Check if LLM wants to run a tool
        if response.tool_calls:
            for tool_call in response.tool_calls:
                function_name = tool_call.function.name
                # Parse arguments safely depending on LLM client format
                args = tool_call.function.arguments 
                
                print(f"🤖 Agent is calling {function_name} with {args}...")
                
                # 4. Route Execution: Local vs MCP
                if function_name in [t.name for t in mcp_tools_response.tools]:
                    # Forward tool call directly to the MCP Server
                    mcp_result = await mcp_session.call_tool(function_name, args)
                    result = mcp_result.content
                else:
                    # Execute standard local logic
                    result = execute_tool(function_name, args)
                
                # 5. Feed result back to LLM
                messages.append({
                    "role": "tool",
                    "content": json.dumps(result),
                    "tool_call_id": tool_call.id
                })
        else:
            # 6. No more tools needed, return final answer
            return response.content

async def main():
    # Setup MCP Server connection configuration
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"]
    )

    # Establish an stdio connection, then wrap it in an MCP ClientSession
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as mcp_session:
            # Initialize with the server
            await mcp_session.initialize()
            
            # Execute the Agentic workflow
            query = "Am I allowed to attend? If so, please send an email with my gate pass."
            context = {"email": "alice@example.com"}
            
            final_answer = await run_agent(query, context, mcp_session)
            print(f"\n✅ Final Answer:\n{final_answer}")

if __name__ == "__main__":
    asyncio.run(main())
```
