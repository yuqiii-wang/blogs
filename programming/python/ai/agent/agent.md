# AI Agent Development Beginner Guide

This guide covers fundamental concepts in modern AI application development, focusing on Large Language Models (LLMs) and Agentic workflows.

## Core Concepts

### 1. LLM (Large Language Model)

The foundation of modern AI applications. Models like GPT-4, Claude 3, and Gemini are trained on vast amounts of text data to predict the next token in a sequence.
- **Context Window**: The limit on the amount of text (measured in tokens) the model can process at once (input + output).
- **Tokens**: The basic unit of text for an LLM (roughly 0.75 words in English).
- **Temperature**: A parameter controlling randomness. Low (0.0) is deterministic; high (0.8+) is creative.

### 2. Prompts & Prompt Engineering

The art of crafting inputs to guide the model's output.
- **System Prompt**: The initial instructions that define the AI's persona, constraints, and behavior (e.g., "You act as a senior Python developer").
- **Zero-shot vs. Few-shot**: Asking the model to perform a task with no examples vs. providing a few examples of input/output pairs in the prompt.
- **Chain of Thought (CoT)**: Encouraging the model to "think step-by-step" to improve reasoning for complex tasks.

### 3. Embeddings & Similarity (Emb Sim)

- **Embeddings**: Converting text (or images) into a list of numbers (vectors) that represent semantic meaning. "Dog" and "Puppy" will have vectors that are numerically close.
- **Vector Database**: A specialized database (e.g., Pinecone, Chroma, pgvector) optimized for storing and querying these high-dimensional vectors.
- **Cosine Similarity**: A metric used to measure how similar two embedding vectors are. Used to find the most relevant documents for a user's query.

### 4. RAG (Retrieval-Augmented Generation)

A technique to ground LLM responses in specific, external data that the model wasn't trained on.
1.  **Retrieve**: User query is converted to an embedding ~> database finds relevant chunks of text.
2.  **Augment**: These chunks are added to the prompt as context.
3.  **Generate**: The LLM answers the user's question using the provided context.

### 5. Agents

An AI system that uses an LLM as a "brain" to reason, plan, and execute actions to achieve a goal. unlike a simple chatbot (input -> output), an agent operates in a loop:
1.  **Thought**: Analyze the request.
2.  **Plan**: Decide which tools to use.
3.  **Action**: Call a tool (function).
4.  **Observation**: Read the tool output.
5.  **Repeat**: Until the task is done.

### 6. Tools & Skills

- **Tools (Function Calling)**: Capabilities given to an Agent. These are functions the LLM can "call" by generating structured JSON (e.g., `get_weather(city="London")`). The application runs the code and returns the result to the LLM.
- **Skills**: A higher-level grouping of tools or a specific competency of an agent (e.g., "Web Research Skill" or "Python Coding Skill").

### 7. MCP (Model Context Protocol)

MCP is an open interoperability standard that solves the fragmentation problem in AI tool integration. Originally developed by Anthropic, it replaces bespoke API connectors with a universal protocol based on **JSON-RPC 2.0**. This standardization allows AI assistants (like Claude or IDEs) to connect to any data source—from local PostgreSQL databases to remote services like Google Drive—using a single, unified interface.

#### Protocol Architecture & Data Unification

MCP eliminates the need for "m × n" integrations (connecting *m* models to *n* tools) by defining a standardized architecture with three primary entities:

- **MCP Host**: The application where the AI model operates (e.g., VS Code, Claude Desktop, Cursor). It manages the lifecycle of connections and discovery of servers.
- **MCP Client**: The protocol implementation within the Host that maintains a 1:1 connection with a server. It speaks the "MCP Language" (JSON-RPC) to translate the Host's intent into protocol messages.
- **MCP Server**: A lightweight service that exposes specific capabilities (resources, prompts, tools). It can be a local process or a remote service.

$$
\underbrace{
\underset{\substack{\\\updownarrow\\\\\text{app tools and prompts}}}{\text{MCP Client}}
\underset{\substack{\text{user}\\\text{app}}}{\in} \underset{\substack{\\\updownarrow\\\\\text{(external) LLM}}}{\text{MCP Host}}
\xleftrightarrow[\text{(stdio or SSE)}]{\text{JSON-RPC 2.0}} \text{MCP Server}}_{\text{on the same computer}}
\begin{cases}
  \xleftrightarrow[\text{(HTTP / WS / SQL / etc.)}]{\text{bespoke protocol}} \text{external systems} \\
  \text{exposes: resources, prompts, tools}
\end{cases}
$$

#### MCP Development Work

1. **Developing an MCP Server (The API Adapter)**
   - **Goal**: Expose a bespoke backend (Database, internal logic, third-party service) to any MCP-compatible AI.
   - Define the explicit JSON schemas for the **Tools**, **Prompts**, or **Resources**.

2. **Developing an MCP Host/Client (The AI Application)**
    - Implement the protocol (or use a client SDK) to manage connection lifecycles (spinning up local binaries or connecting to remote endpoints).
    - Query connected servers for their capabilities (`tools/list`).
    - Pass these discovered tool schemas as standard "Function Calling" definitions to underlying LLM.
    - When the LLM generates a tool call, route it through the MCP Client as a JSON-RPC `tools/call` message and return the serialized result back to the LLM context.

#### Example: Cross-Context Debugging (Technical Flow)

Consider a scenario where an AI assistant investigates a production incident. Technically, this involves the Host orchestrating requests across distinct MCP servers using the JSON-RPC 2.0 protocol over stdio streams.

1.  **Reading Logs (Resource Access via `resources/read`)**:
    The Host requests the latest error log content from the Filesystem Server.
    ```json
    // Request (Host -> FS Server)
    {
      "jsonrpc": "2.0",
      "method": "resources/read",
      "params": { "uri": "file:///app/prod.log" },
      "id": 1
    }
    ```

    **LLM Reasoning Phase**: The Host feeds the retrieved log string back into the LLM's context window. The LLM reads the log, spots a critical timestamp or error message (e.g., "deadlock detected at 2023-10-27T10:00:00Z"), and deduces the next troubleshooting step is checking the database state. It correlates this intent with the available tool schemas and outputs a structured request to call the `run_query` tool.
    
    > **Note: What is a Tool Schema?**
    > A tool schema is indeed a JSON object (specifically, standard JSON Schema) provided to the LLM beforehand. For example:
    > ```json
    > {
    >   "name": "run_query",
    >   "description": "Executes a SELECT query on the Postgres database.",
    >   "inputSchema": {
    >     "type": "object",
    >     "properties": {
    >       "sql": { "type": "string", "description": "The SQL query to run." }
    >     },
    >     "required": ["sql"]
    >   }
    > }
    > ```
    > When LLM decides to use a tool, it outputs a specialized JSON structure containing the exact string `"name"` of the tool (e.g., `"run_query"`) and the generated keyword arguments that match the schema.
    > 
    > **Example of LLM Output (Tool Call):**
    > ```json
    > {
    >   "type": "function",
    >   "function": {
    >     "name": "run_query",
    >     "arguments": "{ \"sql\": \"SELECT * FROM locks WHERE created_at > '2023-10-27T10:00:00Z'\" }"
    >   }
    > }
    > ```

2.  **Querying Database (Tool Execution via `tools/call`)**:
    Acting on the LLM's generated tool call, the Host invokes a SQL query on the Postgres Server to find locked rows.
    ```json
    // Request (Host -> Postgres Server)
    {
      "jsonrpc": "2.0",
      "method": "tools/call",
      "params": {
        "name": "run_query",
        "arguments": { "sql": "SELECT * FROM locks WHERE created_at > '2023-10-27T10:00:00Z'" }
      },
      "id": 2
    }
    ```

3.  **Data Unification**:
    Both servers respond with standard JSON objects. The FS Server returns a `contents` blob (text/base64), and the Postgres Server returns a structured `content` list. The AI model receives these normalized inputs, oblivious to the underlying implementation details (e.g., file descriptors or TCP sockets).

#### The `stdio_client`

The `stdio_client` (Standard Input/Output Client) is the default and preferred way for MCP clients to talk to an MCP Server.
The `stdio_client` launches the server script **as a sub-process** (the client automatically spawns a shiny new child process).

The actual client agent/workflow `run_agent(query, context, mcp_session)` is wrapped within the MCP stdio session.

```py
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
```

The Read vs Write:

* Write channel: When the agent wants to call a tool, it writes a JSON-RPC message directly to the server's stdin (Standard Input).
* Read channel: When the server replies with the status of the email text, it prints JSON to its stdout (Standard Output). The agent captures this stream as data, not text on a screen.

Benefits:

* No Port Collisions Nor Network Setup, e.g. "Port 5000 is already in use"
* Zero Orphaned Severs: The connection is bound to the parent process. If client app stops, MCP server terminates the data stream.
* Security: MCP is a on-prem protocol with stdio, no need of network, hence most of malicious online attacks are avoided.

### 8. Fine-tuning

The process of taking a pre-trained base model and training it further on a specific dataset to specialize it for a particular task or tone, rather than relying solely on prompting.

## AI toC Products (Consumer Agents)

AI "toC" (to Consumer) products are applications designed for end-users that leverage agentic workflows to perform tasks autonomously, moving beyond simple chat interfaces.

### 1. Manus (manus.im)

A general-purpose AI agent designed to execute complex workflows. Instead of just giving advice, it can operate a browser to book flights, research topics, or use its creative suite to generate slides and designs. It represents the shift from "chatbots" to "do-bots".

### 2. OpenClaw

An open-source personal AI agent that focuses on local execution and privacy. It connects to personal tools (calendar, email, Slack) to perform actions on the user's behalf. It is known for its "skills" marketplace where users can download new capabilities for their agent.
