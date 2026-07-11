# LangGraph: Stateful Multi-Actor LLM Orchestration

## Introduction
LangGraph is an extension of the LangChain framework designed to build stateful, multi-actor applications using Large Language Models (LLMs). By representing application logic as a cyclic graph, LangGraph enables the orchestration of complex agentic workflows, such as multi-agent collaboration, iterative refinement, and human-in-the-loop systems.

## Basic Functionality

LangGraph structures execution as a state machine where:
- **State**: A shared data structure (often a `TypedDict` or Pydantic model) that is passed around and updated by nodes.
- **Nodes**: Python functions or distinct agents that receive the current state, perform operations (e.g., invoking an LLM), and return state updates.
- **Edges**: Conditional or direct pathways that dictate the transitions between nodes based on the current state framework.

## Execution Hierarchy: Thread $\rightarrow$ Node $\rightarrow$ Task

The execution model in LangGraph follows a strict hierarchy to manage concurrency and state isolation.

1. **Thread ($T$)**: Represents a continuous session or a single trace of graph execution. Threads isolate state across different users or sessions. A thread comprises a sequence of checkpoints.
2. **Node ($N$)**: The logical computation unit defined in the graph structue. For a given state $S_t$ at time $t$, the active node $N_i$ computes the state update $\Delta S_t$.
3. **Task ($K$)**: A localized execution instance of a Node. When a Thread reaches a Node, it spawns a Task to process the inputs. If a graph branches (parallel node execution), multiple tasks are spawned concurrently within the same thread.

Formally, the network transition function can be represented as:
$$ S_{t+1} = S_t \oplus \sum_{k \in K(N)} \Delta S_{t,k} $$
where $\oplus$ denotes the discrete state reduction/update operation defined for the specific key in the graph attributes.

## Checkpoint Preservation and Interruption Recovery

LangGraph provides built-in persistence (e.g., SQLite, PostgreSQL) to manage state transitions, enabling robust execution control.

### Checkpoints and State Saving (PostgreSQL)
At each graph superstep (after node executions complete), LangGraph saves a **checkpoint** of the cumulative state. Utilizing persistent storage like PostgreSQL ensures robust durability for production environments.

```python
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

pool = ConnectionPool(conninfo="postgresql://user:pass@localhost:5432/db")
memory = PostgresSaver(pool)
graph = builder.compile(checkpointer=memory)
```

**Checkpoint Binary Structure:**
The underlying serialized checkpoint binary strictly encapsulates:
1. **$V$ (State Payload)**: The exact materialization of the state graph at the current step `step`.
2. **$K$ (Pending Tasks & Writes)**: The queued operations and data pushed to the communication channels for the subsequent superstep.
3. **$M$ (Metadata)**: Session identifiers (e.g., `thread_id`, `checkpoint_ns`, `step`).
4. **$P$ (Parent Checkpoint)**: A linked pointer to the preceding checkpoint configuration, enabling linked-list traversal for tracking history.

### Subgraph State Management
LangGraph handles hierarchical logic via composable **subgraphs**. State management scales organically to these nested components:
- **Namespace Isolation**: Each subgraph executes within an isolated thread context, delineated by a `checkpoint_ns` (checkpoint namespace).
- **Hierarchical Linking**: The subgraph's initial checkpoint intrinsically references its parent node's execution context.
- **State Propagation**: State modifications operate locally within the subgraph's computational scope. Upon termination, the terminal state reduces ($\oplus$) upwards into the parent's state framework, maintaining isolated yet composable state machines.

### Recovery from Interruption
Because the state is persisted after every node closure, graph execution can gracefully halt and resume. When an interruption occurs (e.g., process failure, or explicitly pausing for human-in-the-loop input via `interrupt_before`), the system traps the execution state. To recover, the graph simply re-invokes the thread using the active `config` checkpoint, continuing execution deterministically from the point of interruption.

### Time Travel
Since all historical checkpoints are retained within a Thread, LangGraph supports **Time Travel**. This allows developers to interact with the execution graph across the temporal dimension:
1. **Replay**: Observe or resume execution from a historical checkpoint without modifying the past.
2. **Forking / Modification**: Alter a historical state $S_{t-k}$ and resume execution along a new trajectory, effectively branching the system's history.

**Retrieving Historical Snapshots:**
Snapshots can be targeted via several deterministic methods:
- **By Absolute Identifier**: Directly querying a known `checkpoint_id` for $O(1)$ precision.
- **By Metadata Search**: Filtering the history iterator based on specific steps, source nodes, or custom payload properties.
- **By Chronological Index**: Fetching the thread sequence and slicing positionally.

```python
base_config = {"configurable": {"thread_id": "session-1"}}

# Method 1: Localize by discrete Checkpoint ID
config_by_id = {"configurable": {"thread_id": "session-1", "checkpoint_id": "1ef-uuid-..."}}
state_by_id = graph.get_state(config_by_id)

# Method 2 & 3: Localize by Metadata filtering or Index
history = list(graph.get_state_history(base_config))
target_state = next((s for s in history if s.metadata.get("step") == 2), history[2])
    
# Altering past state and resuming from the established checkpoint
graph.update_state(target_state.config, {"messages": ["Correction"]})
graph.invoke(None, target_state.config)
```

## Checkpoint Underlying Implementation

1. What is stored in PG Checkpoint
4 tables are created by AsyncPostgresSaver:

checkpoints (the "header" row per graph step)
checkpoint_blobs (non-primitive state values)
Each GraphState field (like query_analysis, stats_data, merged_research) is a dict — too large for inline. It goes here, keyed by (thread_id, checkpoint_ns, channel, version). The checkpoint row's channel_versions points to the exact version to join against.

checkpoint_writes (pending task writes — in-flight outputs)
When a LangGraph node is executing but not yet committed to a checkpoint, its partial writes live here under the current checkpoint_id. On SELECT the pending_writes sub-query joins these in.

checkpoint_migrations — migration version tracking.
Concretely for this graph: After query_node runs, there's a checkpoint row where:

channel_values has inlined primitive fields (thread_id, user_id, query)
channel_versions["query_analysis"] points to a blob version
checkpoint_blobs has the query_analysis dict at that version
metadata.source = "loop", metadata.step = 1
versions_seen["query_node"]["query_analysis"] is set, versions_seen["*"] is updated — this is how LangGraph decides _route_to_region runs next
