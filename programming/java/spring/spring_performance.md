# Spring Performance

## Reactive Streams

*Reactive Streams* are specification for **asynchronous** operation to handle data streams in a reactive, **non-blocking** way, especially when the producer (data source) and the consumer (data processor) operate at different speeds.

* WebFlux vs. Spring MVC

WebFlux is a reactive programming framework introduced in Spring 5 as part of the Spring ecosystem.

Spring MVC is based on the traditional Servlet API and uses a blocking I/O model.

WebFlux is designed for non-blocking, reactive programming and is better suited for high-concurrency, low-latency scenarios.

### Reactive Stream Players

#### `Publisher<T>`

In Project Reactor, Mono and Flux are implementations of `Publisher`.

#### `Subscriber<T>`

A `Subscriber` consumes the data emitted by the `Publisher`.

It has four methods:

* `onSubscribe(Subscription s)`: Called when the subscriber subscribes to the publisher.
* `onNext(T item)`: Called when a new data item is emitted.
* `onError(Throwable t)`: Called when an error occurs.
* `onComplete()`: Called when the stream completes successfully.

### Reactor: `Flux` and `Mono`

#### `Mono`

A `Mono` is a Publisher that emits **zero or one** item.

#### `Flux`

A `Flux` is a Publisher that emits **zero or more** item.

In the example below, `Flux.range(1, 100)` is a publisher to output 100 numbers.
This publisher has a callback defined in `.subscribe(...)`.

```java
Flux.range(1, 100) // Emits numbers from 1 to 100
    .subscribe(new Subscriber<Integer>() {
        private Subscription subscription;
        private int count = 0;

        @Override
        public void onSubscribe(Subscription s) {
            this.subscription = s;
            subscription.request(10); // Request the first 10 items
        }

        @Override
        public void onNext(Integer item) {
            System.out.println("Received: " + item);
            count++;
            if (count % 10 == 0) {
                subscription.request(10); // Request the next 10 items
            }
        }

        @Override
        public void onError(Throwable t) {
            System.err.println("Error: " + t.getMessage());
        }

        @Override
        public void onComplete() {
            System.out.println("Stream completed!");
        }
    });
```

### Functional Programming: `RouterFunction` and `HandlerFunction`

In Spring WebFlux, functional programming is an alternative to the traditional annotation-based programming model (e.g., `@Controller`, `@RequestMapping`).

In the code example below, the traditional java Spring accepting requests is replaced with handlers by routes.

```java
@Bean
public RouterFunction<ServerResponse> routes() {
    return RouterFunctions.route(
            RequestPredicates.GET("/hello/{name}"),
            this::handleHello
    ).andRoute(
            RequestPredicates.GET("/goodbye"),
            this::handleGoodbye
    );
}

private Mono<ServerResponse> handleHello(ServerRequest request) {
    String name = request.pathVariable("name");
    return ServerResponse.ok().bodyValue("Hello, " + name + "!");
}

private Mono<ServerResponse> handleGoodbye(ServerRequest request) {
    return ServerResponse.ok().bodyValue("Goodbye!");
}
```

## Java Virtual Threads

Java Virtual Threads (JEP 444, GA in JDK 21) are lightweight threads managed by the JVM rather than the OS, enabling massive concurrency with blocking I/O without the complexity of reactive programming.

### Thread Model Comparison

| | Platform Thread | Virtual Thread | WebFlux (Reactor) |
|---|---|---|---|
| Managed by | OS | JVM | Application (event loop) |
| Blocking I/O | Blocks OS thread | Parks virtual thread | Non-blocking callback |
| Programming model | Imperative | Imperative | Functional / reactive |
| Concurrency unit | ~thousands | ~millions | Event-driven pipeline |
| Spring support | Spring MVC | Spring MVC + `@EnableVirtualThreads` | Spring WebFlux |

### Virtual Threads in Spring MVC

Spring Boot 3.2+ can use virtual threads transparently via a single property:

```yaml
spring:
  threads:
    virtual:
      enabled: true
```

With this, each HTTP request is handled on a virtual thread. Blocking calls (JDBC, `RestTemplate`, file I/O) simply **park** the virtual thread instead of blocking an OS thread, freeing the scheduler to run other virtual threads.

```java
// Looks blocking, but the virtual thread parks — not an OS thread
@GetMapping("/users/{id}")
public User getUser(@PathVariable Long id) {
    return userRepository.findById(id).orElseThrow(); // JDBC blocking call
}
```

### Streaming with Virtual Threads vs. WebFlux

#### WebFlux Reactive Streaming

WebFlux streams data through a `Flux` pipeline. Backpressure is handled via the `Subscription.request(n)` protocol, keeping producers from overwhelming consumers.

```java
// Server-Sent Events stream — fully reactive, non-blocking
@GetMapping(value = "/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
public Flux<String> stream() {
    return Flux.interval(Duration.ofMillis(100))
               .map(i -> "event-" + i)
               .take(50);
}
```

The thread model here: one or two event-loop threads (`Netty`) handle all connections. No thread is ever blocked.

#### Virtual Thread Streaming with `StreamingResponseBody`

Spring MVC with virtual threads can stream large responses by writing directly to an `OutputStream` on a virtual thread:

```java
@GetMapping("/stream")
public ResponseEntity<StreamingResponseBody> stream() {
    StreamingResponseBody body = outputStream -> {
        for (int i = 0; i < 50; i++) {
            outputStream.write(("event-" + i + "\n").getBytes());
            outputStream.flush();
            Thread.sleep(100); // parks virtual thread, not OS thread
        }
    };
    return ResponseEntity.ok()
                         .contentType(MediaType.TEXT_PLAIN)
                         .body(body);
}
```

#### Server-Sent Events with Virtual Threads

```java
@GetMapping(value = "/sse", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
public SseEmitter sse() {
    SseEmitter emitter = new SseEmitter(Long.MAX_VALUE);
    Thread.ofVirtual().start(() -> {
        try {
            for (int i = 0; i < 50; i++) {
                emitter.send(SseEmitter.event().data("event-" + i));
                Thread.sleep(100);
            }
            emitter.complete();
        } catch (Exception e) {
            emitter.completeWithError(e);
        }
    });
    return emitter;
}
```

### Backpressure: The Core Difference

Backpressure is the mechanism by which a **consumer signals capacity** to a producer to prevent buffer overflow.

- **WebFlux / Reactor**: Backpressure is a first-class protocol. `Subscription.request(n)` propagates demand upstream through the entire pipeline. Operators like `onBackpressureBuffer()`, `onBackpressureDrop()`, and `onBackpressureLatest()` give fine-grained control.
- **Virtual Threads**: No built-in backpressure. The imperative code naturally applies backpressure via **blocking** — the producer blocks (parks the virtual thread) on a `BlockingQueue` or `OutputStream.flush()` until the consumer is ready.

```java
// Implicit backpressure via blocking queue on a virtual thread
BlockingQueue<String> queue = new ArrayBlockingQueue<>(16);

Thread.ofVirtual().start(() -> {  // producer
    for (int i = 0; i < 1000; i++) {
        queue.put("item-" + i);   // parks when queue is full
    }
});

Thread.ofVirtual().start(() -> {  // consumer
    while (true) {
        String item = queue.take(); // parks when queue is empty
        process(item);
    }
});
```

### When to Choose Which

| Scenario | Recommendation |
|---|---|
| High-throughput event streaming, SSE, WebSocket at scale | **WebFlux** |
| Complex async pipelines with backpressure operators | **WebFlux** |
| Existing blocking code (JDBC, legacy libs) | **Virtual Threads** |
| Simple REST APIs with moderate concurrency | **Virtual Threads** |
| Mixed blocking + async in same service | **Virtual Threads** (simpler) |
| Integrating with reactive libraries (R2DBC, reactive Redis) | **WebFlux** |

> **Key insight**: Virtual threads make blocking I/O cheap but do not eliminate the need for backpressure design in streaming scenarios. WebFlux provides structural backpressure; virtual threads rely on blocking primitives to achieve the same effect implicitly.
