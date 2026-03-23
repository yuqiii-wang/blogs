# AI Development Beginner Guide

This guide covers fundamental concepts in modern AI application development, focusing on Large Language Models (LLMs) and Agentic workflows.

## Core Concepts

### LLM (Large Language Model)

The foundation of modern AI applications. Models like GPT-4, Claude 3, and Gemini are trained on vast amounts of text data to predict the next token in a sequence.
- **Context Window**: The limit on the amount of text (measured in tokens) the model can process at once (input + output).
- **Tokens**: The basic unit of text for an LLM (roughly 0.75 words in English).
- **Temperature**: A parameter controlling randomness. Low (0.0) is deterministic; high (0.8+) is creative.

#### Tokenization & Embedding

Text must be transformed into continuous numerical vectors before neural networks can process it. The pipeline maps a sequence of raw text to a continuous representation $X \in \mathbb{R}^{n \times d}$:

1. **Tokenization**: Breaking text into a sequence of discrete tokens $S = [t_1, t_2, \dots, t_n]$ (e.g., using Byte-Pair Encoding).
2. **Indexing**: Mapping each token $t_i$ to an integer ID $id_i \in \{1, \dots, |V|\}$ in vocabulary $V$.
3. **Embedding**: Projecting IDs to dense vectors $x_i \in \mathbb{R}^d$ via an embedding matrix $E \in \mathbb{R}^{|V| \times d}$, where $x_i = E_{id_i}$.

This produces the input matrix $X = [x_1, \dots, x_n]^T \in \mathbb{R}^{n \times d}$. For GPU efficiency, $X$ is stored in half-precision (FP16/BF16). 

The input $X$ is then used in self-attention:

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

The attention scores are then calculated as:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\tau \cdot \sqrt{d_k}}\right)V
$$

From this formulation, we can mathematically understand critical LLM constraints and behaviors:
1. **Quadratic Scaling ($O(n^2)$)**: The term $QK^T$ evaluates the correlation of every token with every other token. Multiplying the $n \times d_k$ matrix $Q$ by the $d_k \times n$ matrix $K^T$ yields an $n \times n$ attention matrix. Building this matrix requires $O(n^2 \cdot d_k)$ compute operations.
2. **Context Limits**: Since the intermediate $n \times n$ attention matrix must be held in VRAM, memory usage also scales at $O(n^2)$. This quadratic explosion dictates the hard upper limit on the context window for a given GPU.
3. **Temperature ($\tau$)**: The scalar $\tau$ adjusts the logits before they pass through the exponential function inside the `softmax`. 
   - When $\tau < 1.0$, the differences between attention scores are amplified: high scores get closer to 1 (deterministic/sharp).
   - When $\tau > 1.0$, the differences are dampened: the distribution of scores approaches a uniform noise (random/creative).


### Prompts & Context

LLM inputs can be broadly considered as prompts.

#### Prompts

The art of crafting inputs to guide the model's output.
- **System Prompt**: The initial instructions that define the AI's persona, constraints, and behavior (e.g., "You act as a senior Python developer").
- **Zero-shot vs. Few-shot**: Asking the model to perform a task with no examples vs. providing a few examples of input/output pairs in the prompt.
- **Chain of Thought (CoT)**: Encouraging the model to "think step-by-step" to improve reasoning for complex tasks.

#### Context

While modern AI frameworks and IDEs present concepts like "tools", "skills", "rules", "memory", and "project indexing" as distinct architectural components, **they are all ultimately just text injected into the context window**. Before a request is sent to the LLM, the application dynamically compiles these elements together:

- **Tools**: Tool schemas are converted into JSON descriptions detailing function names, parameters, and descriptions.
- **Rules & Skills**: These are prepended as strict system instructions defining the AI's persona, constraints, and coding standards.
- **Memory**: Retrieved user preferences, session state, or historical summaries are appended as background context.
- **Project Indexing**: Code snippets, file tree structures, and relevant documentation are retrieved and injected as raw text to ground the LLM in the current workspace.

The LLM then reads this massive block of text to understand its capabilities, constraints, relevant facts, and finally user query for response.
For example, below rules and skills are just context prompts as prefix input to LLM.

<div style="display: flex; justify-content: center;">
      <img src="imgs/ai_ide_prompt_context.png" width="50%" height="70%" alt="ai_ide_prompt_context" />
</div>
</br>


### Embeddings & Similarity (Emb Sim)

- **Embeddings**: Converting text (or images) into a list of numbers (vectors) that represent semantic meaning. "Dog" and "Puppy" will have vectors that are numerically close.
- **Vector Database**: A specialized database (e.g., Pinecone, Chroma, pgvector) optimized for storing and querying these high-dimensional vectors.
- **Cosine Similarity**: A metric used to measure how similar two embedding vectors are. Used to find the most relevant documents for a user's query.

### RAG (Retrieval-Augmented Generation)

A technique to ground LLM responses in specific, external data that the model wasn't trained on.
1.  **Retrieve**: User query is converted to an embedding (typically emb) ~> database finds relevant chunks of text. The search on corpus can be also done by traditional NLP search algo, e.g., BM25.
2.  **Augment**: These chunks are added to the prompt as context.
3.  **Generate**: The LLM answers the user's question using the provided context.

### Agents

An AI system that uses an LLM as a "brain" to reason, plan, and execute actions to achieve a goal. unlike a simple chatbot (input -> output), an agent **operates in a loop**:
1.  **Thought**: Analyze the request.
2.  **Plan**: Decide which tools to use.
3.  **Action**: Call a tool (function).
4.  **Observation**: Read the tool output.
5.  **Repeat**: Until the task is done.

### MCP (Model Context Protocol)

MCP is an open interoperability standard that solves the fragmentation problem in AI tool integration. Originally developed by Anthropic, it replaces bespoke API connectors with a universal protocol based on **JSON-RPC 2.0**. This standardization allows AI assistants (like Claude or IDEs) to connect to any data source—from local PostgreSQL databases to remote services like Google Drive—using a single, unified interface.

#### Protocol Architecture & Data Unification

- **MCP Host**: The application where the AI model operates (e.g., VS Code, Claude Desktop, Cursor). It manages the lifecycle of connections and discovery of servers.
- **MCP Client**: The protocol implementation within the Host that maintains a 1:1 connection with a server. It speaks the "MCP Language" (JSON-RPC) to translate the Host's intent into protocol messages.
- **MCP Server**: A lightweight service that exposes specific capabilities (resources, prompts, tools). It can be a local process or a remote service.

$$
\underset{\substack{\\\updownarrow\\\\\text{app tools and prompts}}}{\text{MCP Client}}
\underset{\substack{\text{user}\\\text{app}}}{\in} \underset{\substack{\\\updownarrow\\\\\text{(external) LLM}}}{\text{MCP Host}}
\xleftrightarrow[\text{(stdio or SSE)}]{\text{JSON-RPC 2.0}} \text{MCP Server}
\begin{cases}
  \xleftrightarrow[\text{(HTTP / WS / SQL / etc.)}]{\text{bespoke protocol}} \text{external systems} \\
  \text{exposes: resources, prompts, tools}
\end{cases}
$$

where SSE stands for *Server-Sent Events*.

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

    In python implementation by `mcp` lib, a mcp-compatible tool schema can be simply defined by labelling `@mcp.tool()` to a func.
    The func name `run_query` and func arg `sql: str`, and description wrapped in `"""..."""` are passed to `@mcp.tool()` to build the tool schema.
    LLM has a pool of tools in which each tool got a description and func title by which LLM can map user query/intent to a tool/func with semantic understanding.

    ```py
    @mcp.tool()
    def run_query(sql: str) -> str:
        """
        Executes a SELECT query on the Postgres database.
        """
        # Create a new connection or use an existing connection pool
        try:
            with psycopg2.connect(DB_CONN_STRING) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    if not sql.strip().upper().startswith("SELECT"):
                        return json.dumps({"error": "Only SELECT queries are allowed."})
                    cursor.execute(sql)
                    records = cursor.fetchall()
                    return json.dumps(records, default=str)
        except psycopg2.Error as e:
            return json.dumps({"error": str(e)})
    ```

2.  **Querying Database (Tool Execution via `tools/call`)**:
    Acting on the LLM's generated tool call, the Host invokes a SQL query on the Postgres Server to find locked rows.
    The `"arguments": { "sql": "..." }` is passed to the tool func `run_query(sql: str)`.

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

### Training and Fine-tuning

The process of training an LLM involves learning the relationships between words by predicting missing or future tokens. For causal language models like GPT, this is primarily **next-token prediction** given a sequence. For masked language models (like BERT), this involves **guessing masked words** within a sentence.

#### The Attention Mechanism and Weights

At the core of modern LLMs is the Transformer architecture, heavily relying on the **Self-Attention** mechanism. For an input sequence representation $X$, the model learns weight matrices $W_Q, W_K, W_V$ to project $X$ into Queries ($Q$), Keys ($K$), and Values ($V$):

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

The attention scores are then calculated as:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

where $d_k$ is the dimension of the keys. During **pre-training**, these massive weight matrices ($W_Q, W_K, W_V$), along with feed-forward network weights and embeddings, are continuously adjusted (updated) through backpropagation to minimize the token prediction error.

#### Fine-Tuning Approaches

Fine-tuning takes a pre-trained base model and trains it further on a specific dataset to specialize it for a particular task, domain, or tone. There are two primary approaches:

- **Full-Parameter Fine-Tuning**: All original weights of the pre-trained model are updated during training. This is highly effective but computationally expensive and requires significant memory.
- **Parameter-Efficient Fine-Tuning (PEFT)**: Only a small subset of parameters (or newly added parameters) are trained, while freezing most original weights. This saves compute and memory.
  - **LoRA (Low-Rank Adaptation)**: Instead of updating a massive original weight matrix $W_0 \in \mathbb{R}^{d \times k}$ (like $W_Q$ or $W_V$), LoRA freezes $W_0$ and injects trainable low-rank matrices $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$, where the rank $r \ll \min(d, k)$. The new calculation becomes:
    $$
    h = XW_0 + XBA
    $$
    Only the small matrices $A$ and $B$ are trained, approximating the ideal weight update $\Delta W = BA$. It drastically reduces the number of trainable parameters, enabling fine-tuning of large models on consumer GPUs with minimal performance trade-offs.
  - **Prefix Tuning / Prompt Tuning**: Another PEFT approach where a small set of trainable, continuous "prefix" tokens are prepended to the input or hidden layers. Only these prefix embeddings are optimized during training while the base model weights remain completely frozen.

#### Knowledge Distillation

Knowledge Distillation is a technique used to transfer the "knowledge" of a massive, complex model (the **Teacher**) to a smaller, more efficient model (the **Student**). Rather than training the student model purely on raw dataset labels, it is trained to mimic the outputs, reasoning patterns, or internal representations of the teacher model.
- **Soft Targets**: The student learns from the comprehensive probability distributions (soft labels) generated by the teacher instead of just the final answer. This preserves the nuanced understanding of the teacher (e.g., recognizing that an answer, while incorrect, might still be semantically close).
- **Efficiency**: This approach creates compact models that run significantly faster and require far less computing power to host (lowering deployment costs), while retaining a performance level that punches way above their parameter count.

## AI toC Products (Consumer Agents)

AI "toC" (to Consumer) products are applications designed for end-users that leverage agentic workflows to perform tasks autonomously, moving beyond simple chat interfaces.

### 1. Manus (manus.im)

A general-purpose AI agent designed to execute complex workflows. Instead of just giving advice, it can operate a browser to book flights, research topics, or use its creative suite to generate slides and designs. It represents the shift from "chatbots" to "do-bots".

### 2. OpenClaw

An open-source personal AI agent that focuses on local execution and privacy. It connects to personal tools (calendar, email, Slack) to perform actions on the user's behalf. It is known for its "skills" marketplace where users can download new capabilities for their agent.
