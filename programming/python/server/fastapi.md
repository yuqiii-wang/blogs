# Fast API

By 2026, FastAPI is the most popular choice of python-native server.

## Uvicorn and Starlette as Backend for FastAPI

There are Uvicorn + Starlette + FastAPI.

Their relationships are

$$
\text{Uvicorn}\in\text{Starlette}\in\text{FastAPI}
$$

* Uvicorn: server engine parsing HTTP request raw bytes to python compatible objects
* Starlette: a lightweight ASGI framework/chassis for routing
* FastAPI: wrapper with assistance functions, e.g., validation per `pydantic` 

For example, when a HTTP request comes in this FastAPI server:

```py
from fastapi import FastAPI

app = FastAPI()

@app.get("/items/{item_id}")
def get_item(item_id: int, q: str | None = None):
    # ... Some user define process logic, e.g., store data to DB
    return {"item_id": item_id, "query_string": q}
```

### 1. Uvicorn (The ASGI Server)

Listens on the Network: Uvicorn is bound to the network socket at `127.0.0.1:8000`. It receives the incoming TCP connection from the client.
It reads the raw text of the HTTP request from the socket.

Uvicorn breaks the raw bytes down into its core components:

```txt
Method: GET
Path: /items/42
Query String: q=hello
Headers: A list of key-value pairs.
```

Uvicorn then creates a standard Python dictionary called the ASGI `scope` such as below

```py
{
    'type': 'http',
    'method': 'GET',
    'path': '/items/42',
    'query_string': b'q=hello', # as bytes
    'headers': [
        (b'host', b'127.0.0.1:8000'),
        (b'accept', b'*/*'),
        ...
    ],
    ... # other server and client info
}
```

While Uvicorn itself is an ASGI server written in Python, its high performance comes from its use of other libraries that are not pure Python.

Reference: https://uvicorn.dev/concepts/event-loop/

The replaced event loop implementations:

* `uvloop` - If `uvloop` is installed, Uvicorn will use it for maximum performance (`uvloop` is built on `libuv`)
* `asyncio` - If `uvloop` is not available, Uvicorn falls back to Python's built-in `asyncio` event loop

Besides, custom event loop libs are

* `rloop` is an experimental `AsyncIO` event loop implemented in Rust on top of the mio crate. It aims to provide high performance through Rust's systems programming capabilities.
* `Winloop` is an alternative library that brings uvloop-like performance to Windows.

### 2. Starlette (Routing)

Starlette is a lightweight ASGI framework/chassis for routing.
It has an thread pool of up to size 40 (due to python GIL, only one hardware CPU could run at once).

Starlette looks at the path (`/items/42`) and the method (`GET`). It checks the list of all available routes that were registered when the application started.

### 3. FastAPI (The Application Logic & Data Layer)

FastAPI on checking Starlette result then:

* Data Validation and Coercion, e.g., in `get_item` the request format is validated such as `item_id` as `int`.
* User defined process logic

Besides, FastAPI provides assistant utils such as `swagger` API docs on `http://<host>:<port>/docs`

## Gunicorn, Uvicorn, and Process and Thread Management

When using Uvicorn as server engine for FastAPI, by default only one hardware CPU could be used.
There needs multi-processing/parallelism to utilize multi-core CPUs.

Best practice:

Recommend running pure uvicorn in production when using container orchestrators such as Kubernetes and set the number of containers (processes) there instead.
Kubernetes has options to automatically scale containers up and down depending on CPU and memory usage, thus minimizing the resource usage.

Reference:

* https://medium.com/@iklobato/mastering-gunicorn-and-uvicorn-the-right-way-to-deploy-fastapi-applications-aaa06849841e
* https://stackoverflow.com/questions/66362199/what-is-the-difference-between-uvicorn-and-gunicornuvicorn
* https://github.com/Kludex/uvicorn/issues/303

### FastAPI with K8S

Quotes from https://fastapi.tiangolo.com/deployment/docker/#https

> If you have a cluster of machines with Kubernetes, Docker Swarm Mode, Nomad, or another similar complex system to manage distributed containers on multiple machines, 
> then you will probably want to handle replication at the cluster level instead of using a process manager (like Uvicorn with workers) in each container.

where **replication of containers** and **load balancing** all at the **cluster level** are taken care of.

### Gunicorn for Process Management

```sh
gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

This command runs Gunicorn, but instead of its default synchronous workers, it uses Uvicorn's ASGI workers, combining Gunicorn's process management with Uvicorn's asynchronous capabilities.

The process states:

1. A single master Gunicorn process starts up.
2. This master process then "forks" itself to create 4 independent worker processes.
3. Each of these four workers is an instance of `uvicorn.workers.UvicornWorker`, effectively running its own Uvicorn server instance.

The Gunicorn master process does not handle requests itself. Instead, it manages the workers and distributes incoming requests among them.
If a worker process crashes, the master will automatically start a new one to replace it, ensuring the stability of your application.

## Practice Guide

### Server Init and Context Management

In FastAPI, the `lifespan` parameter is useful to define startup and shutdown logic for application using an asynchronous context manager.
This is particularly useful for tasks like initializing database connections, setting up resources, or cleaning up on shutdown.

For example, below example launches a FastAPI server with DB context management (before app is launched, DB connection pool is set up in `init_db()`, and closed in `close_db()` once app is shutdown).
The `@asynccontextmanager` decorator from Python's contextlib module transforms an async generator function (one that uses `yield`) into an asynchronous context manager.

```py
@asynccontextmanager
async def lifespan(_: FastAPI):
	await init_db()
	yield
	await close_db()

app = FastAPI(title="My APP API", version="1.0.0", lifespan=lifespan)
```

where before vs after `yield`:

* `yield` pauses the function and returns control to the caller (FastAPI's internal startup process).
* During this pause, FastAPI runs the main application loop, handling requests.
* The function doesn't exit; it waits here until the app shuts down.

The context manager saved effort manually implementing `__aenter__` and `__aexit__` methods.

```py
class LifespanContext:
    async def __aenter__(self):
        await init_db()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await close_db()
```

### `Depends` and `Annotated`

### Websocket

Given a typical websocket FastAPI server

```py
from fastapi import FastAPI, WebSocket

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()          # Step A
    try:
        while True:
            data = await websocket.receive_text()  # Step B
            await websocket.send_text(f"Message text was: {data}") # Step C
    except WebSocketDisconnect:
        print("Client disconnected")
```

## Test with `fastapi.testclient`

`TestClient` works its magic by interacting with FastAPI application directly in memory, completely bypassing the network layer.
Therefore, there is no need to start a separate FastAPI app server (like with uvicorn or hypercorn) when using TestClient.

### TestClient Under the Hood: ASGI (Asynchronous Server Gateway Interface)

ASGI is a standard interface between web servers and Python asynchronous web applications or frameworks. FastAPI is an ASGI application.

When running `uvicorn main:app`, Uvicorn acts as the ASGI server. It listens for real HTTP requests on a network socket (e.g., `localhost:8000`). When a request comes in, Uvicorn translates it into an ASGI message format and passes it to FastAPI app object; response is back to Uvicorn also in the ASGI format.

The `TestClient` (which is built on top of `httpx`) acts as a mock ASGI server.
When made a call, e.g., `client.get("/")`, `TestClient` does not open a network socket.
Instead, it constructs the necessary ASGI message in memory that represents a GET `/` request.
