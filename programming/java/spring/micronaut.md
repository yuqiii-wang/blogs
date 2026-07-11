# Micronaut (As SpringBoot Replacement)

## Introduction

Micronaut is a modern, JVM-based full-stack framework for building microservices and serverless applications. Developed by OCI (formerly Grails team) and released in 2018, its core design goal is **eliminating the runtime overhead** that frameworks like Spring Boot incur through reflection and runtime classpath scanning.

Key design pillars:
- **Compile-time dependency injection** — all DI wiring happens via annotation processors at build time (no reflection at runtime)
- **Ahead-of-Time (AoT) metadata generation** — bean definitions, interceptors, and configuration bindings are generated as bytecode
- **GraalVM Native Image friendly** — compile-time approach makes native compilation straightforward without extensive reflection config

---

## Core Features

| Feature | Description |
|---|---|
| Dependency Injection | Compile-time, zero-reflection DI/AOP via annotation processors |
| HTTP Server/Client | Non-blocking HTTP layer built on Netty |
| Declarative HTTP Client | `@Client` interface-driven HTTP clients (no runtime proxy needed) |
| Configuration | Hierarchical config from YAML/properties/env, bound at compile time |
| Service Discovery | Built-in Consul, Eureka, Kubernetes integration |
| Distributed Tracing | Zipkin/Jaeger via `@NewSpan` |
| Security | JWT, session, OAuth2 via `micronaut-security` |
| Data Access | Micronaut Data — compile-time query generation (analogous to Spring Data) |
| Serverless | AWS Lambda, Azure Functions, GCP Cloud Functions adapters |
| GraalVM Native | Native image support with minimal manual configuration |

---

## Underlying Implementation

### HTTP Server: Netty vs. Tomcat vs. Jetty

Spring Boot supports three embeddable server options; Micronaut is built exclusively on Netty.

| | **Tomcat** (Spring default) | **Jetty** (Spring optional) | **Netty** (Micronaut) |
|---|---|---|---|
| I/O model | Blocking, thread-per-request | Blocking (NIO selector, but still thread-per-request at handler level) | Non-blocking event loop |
| Concurrency model | Thread pool (default 200) | Thread pool (default 200) | ~2× CPU-count event-loop threads + optional worker pool |
| Stack memory per conn | ~512 KB/thread | ~512 KB/thread | Near-zero (no per-request thread) |
| Protocol support | HTTP/1.1, HTTP/2 (TLS) | HTTP/1.1, HTTP/2, WebSocket | HTTP/1.1, HTTP/2, WebSocket, gRPC |
| API surface | Servlet (javax/jakarta) | Servlet (javax/jakarta) | Custom `HttpRequest`/`HttpResponse` |
| Backpressure | Manual | Manual | Reactive Streams native |
| Spring Boot starter | `spring-boot-starter-web` | `spring-boot-starter-jetty` | `spring-boot-starter-webflux` |
| Micronaut support | ✗ | ✗ | ✓ (built-in) |

**Tomcat** is the Spring Boot default — mature, widely deployed, strong ecosystem. Its NIO connector (since Tomcat 6) uses a selector thread to accept connections but still dispatches each request to a dedicated thread from the pool, so the 200-thread × 512 KB stack overhead is unavoidable under load.

**Jetty** offers a slightly lighter footprint than Tomcat (no JSP engine by default, leaner classloader) and historically had faster HTTP/2 support. Its thread model is identical to Tomcat — one thread per active request.

Jetty's idle memory advantage over Tomcat is modest (~10–20 MB) because both retain a full Servlet container, filter chain, and session management infrastructure.

**Netty** (Micronaut) uses an event-loop architecture (`EpollEventLoopGroup` on Linux, `KQueueEventLoopGroup` on macOS, `NioEventLoopGroup` elsewhere). Each event-loop thread handles thousands of connections via OS-level I/O multiplexing, allocating no stack per connection. Micronaut's `HttpServer` wraps Netty's `ServerBootstrap` directly — there is no Servlet layer.

**Why Netty matters for memory:**
- A 200-thread Tomcat/Jetty pool consumes ~100 MB of stack before any heap allocation.
- Netty multiplexes thousands of connections onto a handful of event-loop threads via `epoll`/`kqueue`, eliminating that stack overhead entirely.
- Off-heap Netty buffer pools (`PooledByteBufAllocator`) are shared across connections, not allocated per-request.

**Can Spring Boot use Netty?** Yes — Spring WebFlux + `spring-boot-starter-webflux` replaces Tomcat/Jetty with Reactor Netty. However, the application code must be written in Project Reactor (`Mono`/`Flux`). Micronaut supports both blocking and reactive handlers on the same Netty core transparently, with no programming-model switch required.
 
---

### GraalVM Native Image

**GraalVM** is a polyglot runtime and ahead-of-time compiler from Oracle that can compile JVM bytecode into a **standalone native binary** — no JVM required at runtime.

```
Standard JVM startup sequence
  1. JVM process starts (~50–100 ms)
  2. Load & verify bytecode for each class
  3. JIT compiles hot paths on first execution
  4. Framework bootstraps (scan, reflect, proxy)  ← Spring Boot's slow part
  5. Application ready

GraalVM Native Image startup sequence
  1. OS loads pre-compiled binary (~5–20 ms)
  2. Run pre-initialized heap snapshot
  3. Application ready                            ← no JIT, no scan, no reflection
```

The native image compiler performs **closed-world analysis**: it traces all reachable code at build time and strips everything else. Dynamic features (reflection, proxies, class loading) must be declared in `reflect-config.json` or they are removed.

**Spring Boot + GraalVM:** Spring historically relied on reflection heavily. Spring Boot 3 (2022) introduced a Spring AOT engine that generates reflection configs, but the list is large and maintenance-heavy.

**Micronaut + GraalVM:** Because Micronaut already emits concrete `BeanDefinition` classes at compile time and never reflects on user classes at runtime, the closed-world constraint is satisfied naturally — no `reflect-config.json` needed for the framework layer.

---

### Why Micronaut Has a Short Cold Start Time

Cold start time decomposes into three phases:

$$T_{\text{start}} = T_{\text{JVM\_init}} + T_{\text{DI\_bootstrap}} + T_{\text{server\_bind}}$$

| Phase | Spring Boot | Micronaut | Reason |
|---|---|---|---|
| $T_{\text{JVM\_init}}$ | ~300–500 ms | ~300–500 ms | Same JVM (on JVM mode) |
| $T_{\text{DI\_bootstrap}}$ | **1–5 s** | **10–50 ms** | Spring scans classpath + builds context via reflection; Micronaut loads pre-generated `BeanDefinition` classes directly |
| $T_{\text{server\_bind}}$ | ~200–500 ms | ~50–100 ms | Tomcat initializes servlet context + filters; Netty channel bind is lighter |

The dominant cost in Spring Boot is $T_{\text{DI\_bootstrap}}$. It involves:
1. `ClassPathScanningCandidateComponentProvider` — walks every JAR entry matching component filters
2. `AutoConfigurationImportSelector` — loads and evaluates `spring.factories` conditions
3. `DefaultListableBeanFactory` — resolves dependency graph via reflection, instantiates beans, runs `@PostConstruct`

Micronaut replaces all of step 1–3 with a direct classloader call to pre-generated classes:

```java
// Spring Boot (simplified runtime path)
Set<BeanDefinitionHolder> candidates = scanner.findCandidateComponents("com.example");
for (BeanDefinitionHolder holder : candidates) {
    Class<?> beanClass = Class.forName(holder.getBeanClassName()); // reflection
    // resolve constructor args, check @Conditional, create proxy...
}

// Micronaut (simplified runtime path)
BeanDefinition<UserService> def = new $UserService$Definition(); // generated class
def.inject(context, instance); // direct field/method call, no reflection
```

On **GraalVM native**, $T_{\text{JVM\_init}}$ drops to near-zero because the JVM heap is serialized into the binary as an **image heap snapshot**. Spring's initialized context *can* be snapshotted too (Spring AOT), but the image is larger due to retained metadata. Micronaut's native images are typically 15–30% smaller.

---

## Memory Footprint Reduction

Spring Boot resolves beans, reads annotations, and builds its application context **at runtime** via reflection. This costs both startup time and heap.

Micronaut moves this work to **compile time**:

```
Build Time (javac + annotation processor)
  └─ Generates BeanDefinition classes
  └─ Generates interceptor chains (AOP)
  └─ Generates HTTP route metadata
  └─ Generates configuration binding classes

Runtime
  └─ Loads pre-generated BeanDefinition → no classpath scan
  └─ No reflection-based proxy creation
  └─ No runtime annotation reading
```

Concrete measures of reduction:

| Metric | Spring Boot (typical) | Micronaut (typical) | Source |
|---|---|---|---|
| Startup time (simple REST app, JVM) | 2–6 s | 100–300 ms | [1][2] |
| Startup time (GraalVM native) | 1–3 s | 20–80 ms | [1] |
| Idle RSS / heap (simple REST app) | ~150–250 MB | ~20–60 MB | [2][3] |
| First-request latency (cold) | High (JIT warmup) | Low (pre-generated metadata) | [2] |
| GraalVM reflection config entries | 1,000–5,000+ | ~0 (manual) | [4] |
| Class metadata retained at runtime | Full annotation index | Stripped after AoT | [1] |
| Native image build time | Very long (Spring AOT) | Moderate | [4] |

> **Note on measurement methodology:** Figures above reflect single-service benchmarks under minimal load (no connection pool, single datasource). Memory grows comparably at high concurrency because Netty thread/buffer overhead then dominates over DI metadata overhead.

### Benchmark Studies

**[1] TechEmpower Framework Benchmarks (continuous)**
- URL: https://www.techempower.com/benchmarks/
- Micronaut consistently places in the top tier for JSON serialization and single-query DB rounds; Spring Boot MVC lags ~20–40% in throughput at equivalent concurrency due to per-request reflection overhead in older versions.

**[2] "Micronaut vs Spring Boot: Performance Comparison" — Piotr Mińkowski (2022)**
- URL: https://piotrminkowski.com/2022/11/23/micronaut-vs-spring-boot-performance/
- Setup: identical REST endpoints, PostgreSQL, Docker; measured with k6.
- Results: Micronaut consumed **~55–65% less heap** at idle; startup **8–15× faster** on JVM. Under 500 concurrent users throughput was roughly equivalent (Netty vs Tomcat effect dominated).

**[3] "Comparing Spring Boot Memory Usage" — Spring Blog / Sébastien Deleuze (2023)**
- URL: https://spring.io/blog/2023/10/16/runtime-efficiency-with-spring
- Spring's own AOT (Spring Boot 3.x with GraalVM) closes part of the gap; idle heap drops to ~80–120 MB. Micronaut still leads on JVM-mode idle memory.

**[4] "GraalVM Native Image and Micronaut" — Micronaut Docs**
- URL: https://docs.micronaut.io/latest/guide/#graal
- Spring Boot 3 AOT mode requires `reflect-config.json` entries for dynamic usage; Micronaut's compile-time approach generates no dynamic reflection by default, yielding smaller native binaries (~15–30% smaller RSS).

### Interpreting the Numbers

Memory is measured along two axes:

$$\text{RSS} = \text{JVM overhead} + \text{heap}_{\text{live}} + \text{metaspace} + \text{off-heap (Netty buffers)}$$

Spring Boot's excess at idle is concentrated in **metaspace** (retained class metadata) and **heap** (annotation caches, proxy class objects). Micronaut's AoT shifts those costs to compile-time bytecode (on-disk `.class` files), so at runtime they appear only as normal class bytecode — loaded once, not stored in searchable cache structures.

The heap reduction comes from three sources:
1. **No annotation metadata caches** — Spring caches `AnnotationMetadata` per class/method in `ConcurrentHashMap`s kept alive for the lifetime of the context.
2. **No CGLIB/JDK proxies** — Spring AOP creates proxy subclasses at runtime; Micronaut generates interceptor wrappers at compile time.
3. **No classpath scanning state** — Spring's `ClassPathScanningCandidateComponentProvider` retains file-system index objects; Micronaut discards this after the build.

---

## Annotation & Syntax Comparison

### Bean Stereotypes

| Spring Boot | Micronaut | Description | Code Example |
|---|---|---|---|
| `@Component` | `@Singleton` | Declares a general-purpose bean that Micronaut manages for DI. | `@Singleton class UserService {}` |
| `@Service` | `@Singleton` | Marks a business-logic bean in the service layer. | `@Singleton class OrderService {}` |
| `@Repository` | `@Singleton` | Identifies a persistence-oriented bean or data-access component. | `@Singleton class UserRepository {}` |
| `@Controller` | `@Controller` | Defines an HTTP controller that handles incoming requests. | `@Controller("/users") class UserController {}` |
| `@RestController` | `@Controller` | Maps to a controller that is REST-oriented by default in Micronaut. | `@Controller class UserController {}` |
| `@Configuration` | `@Factory` | Declares a class that produces beans through factory methods. | `@Factory class AppConfig { @Bean UserService service() { ... } }` |
| `@Bean` (on method) | `@Bean` (on method inside `@Factory`) | Creates a bean from a method and usually requires an explicit scope. | `@Bean @Singleton UserService service() { ... }` |

### Dependency Injection

| Spring Boot | Micronaut | Description | Code Example |
|---|---|---|---|
| `@Autowired` | `@Inject` (JSR-330) | Injects a dependency into a bean instance. | `@Inject UserService userService;` |
| `@Qualifier("name")` | `@Named("name")` | Selects a bean by a specific logical name. | `@Named("primary") UserService service;` |
| `@Value("${prop}")` | `@Value("${prop}")` (same) | Binds a configuration value into a field or constructor parameter. | `@Value("${server.port}") int port;` |
| N/A | `@Context` | Marks a bean as part of the application context lifecycle for shared runtime resources; not intended for session/request lifetime. | `@Context class AppContext {}` |
| Constructor injection (implicit) | Constructor injection (preferred, explicit `@Inject` optional) | Passes required collaborators through the constructor for clearer wiring. | `class Service { Service(Repo repo) { ... } }` |

#### Lifecycle Example

Micronaut DI is driven by compile-time generated `BeanDefinition` classes. Lifecycle follows predictable phases and emits events you can observe:

- Bean creation: generated `BeanDefinition` is used to instantiate a bean when its scope requires it (eager `@Context`/startup singletons or on-first-resolve lazy singletons).
- Injection: constructor injection (preferred) or `@Inject` is wired via generated bytecode — no runtime reflection.
- Initialization hooks: use `@PostConstruct` or listen for `BeanInitializedEvent<T>` to run init logic after dependencies are injected.
- Runtime events: observe `StartupEvent` and `ShutdownEvent` with `@EventListener` or implement `ApplicationEventListener` for app-level lifecycle.
- Destruction: use `@PreDestroy` or handle `ShutdownEvent` for graceful cleanup.

Common events and listener example:

```java
@Singleton
public class MyStartupListener implements ApplicationEventListener<StartupEvent> {
  @Override
  public void onApplicationEvent(StartupEvent event) {
    // startup logic
  }
}
```

Scope guidance & use cases (concise):
- `@Singleton` — stateless services, shared clients, connection pools. Must be thread-safe.
- `@Prototype` — per-resolution, short-lived objects holding mutable per-operation state.
- `@RequestScope` — per-HTTP-request context (trace IDs, per-request caches, auth wrappers).
- `@SessionScope` — per-user session state for session-based web apps.
- `@Context` — eagerly-created, long-lived resources (background schedulers, managers).
- `@Factory` + `@Bean` — create third-party or complex objects (DataSource, legacy clients).

Quick examples:

```java
@Singleton
public class UserService { /* stateless */ }

@Prototype
public class Parser { private String buffer; }

@RequestScope
public class RequestContext { String traceId; }

@Context
@Singleton
public class BackgroundScheduler {
  @PostConstruct
  void start() { /* schedule background tasks */ }

  @PreDestroy
  void stop() { /* stop tasks and release resources */ }
}

@Factory
public class DataSourceFactory {
  @Bean @Singleton
  DataSource ds(@Value("${db.url}") String url) { return create(url); }
}
```

Testing & conditional beans:
- Use `@Named`, `@Requires`, and `@Replaces` to provide test doubles or conditional beans.
- Prefer `@Replaces` in tests to swap singletons with mocks.

Thread-safety note: never store request-specific mutable state in `@Singleton` beans — use `@RequestScope` or `@Prototype` instead.

### Scopes

| Spring Boot | Micronaut | Description | Code Example |
|---|---|---|---|
| `@Scope("singleton")` / default | `@Singleton` | Creates one shared bean instance for the application lifetime. | `@Singleton class CacheService {}` |
| `@Scope("prototype")` | `@Prototype` | Creates a new bean instance each time it is resolved. | `@Prototype class RequestBean {}` |
| `@RequestScope` | `@RequestScope` | Limits a bean to a single HTTP request lifecycle. | `@RequestScope class RequestContext {}` |
| `@SessionScope` | `@SessionScope` | Limits a bean to the lifetime of a user session. | `@SessionScope class SessionState {}` |

### HTTP Layer

**Spring Boot (Spring MVC):**
```java
@RestController
@RequestMapping("/users")
public class UserController {

    @GetMapping("/{id}")
    public User getUser(@PathVariable Long id) { ... }

    @PostMapping
    public ResponseEntity<User> create(@RequestBody User user) { ... }
}
```

**Micronaut:**
```java
@Controller("/users")
public class UserController {

    @Get("/{id}")
    public User getUser(Long id) { ... }           // path var bound by name

    @Post
    @Status(HttpStatus.CREATED)
    public User create(@Body User user) { ... }
}
```

Key differences:
- `@GetMapping` / `@PostMapping` → `@Get` / `@Post` / `@Put` / `@Delete`
- `@RequestBody` → `@Body`
- `@PathVariable` → implicit by parameter name match (or explicit `@PathVariable`)
- `@RequestParam` → `@QueryValue`
- `@RequestHeader` → `@Header`
 
---

## Quick-Reference Summary

```
Spring Boot              Micronaut
─────────────────────    ──────────────────────
@Component               @Singleton
@Service                 @Singleton
@Repository              @Singleton
@Configuration           @Factory
@Bean (in @Config)       @Bean (in @Factory)
@Autowired               @Inject
@RequestBody             @Body
@PathVariable            (implicit) / @PathVariable
@RequestParam            @QueryValue
@RequestHeader           @Header
@GetMapping              @Get
@PostMapping             @Post
Runtime reflection DI    Compile-time AoT DI
CGLIB proxies            Generated interceptor classes
~150-250 MB idle heap    ~20-60 MB idle heap
```

---

## Compatibility with the Spring Boot Ecosystem

Micronaut is **not a drop-in replacement** for Spring Boot — annotations and APIs differ. However, it provides deliberate bridges and first-class integrations for most of the Spring ecosystem's de-facto standards.

### micronaut-spring (Spring Annotation Bridge)

The `micronaut-spring` module translates a subset of Spring annotations at compile time into their Micronaut equivalents. This allows libraries written against Spring APIs to run inside a Micronaut application context.

```xml
<!-- build.gradle / pom.xml -->
<dependency>
  <groupId>io.micronaut.spring</groupId>
  <artifactId>micronaut-spring-annotation</artifactId>
  <scope>annotationProcessor</scope>
</dependency>
```

Supported Spring annotations mapped at compile time:

| Spring Annotation | Mapped to |
|---|---|
| `@Component`, `@Service`, `@Repository` | `@Singleton` |
| `@Autowired` | `@Inject` |
| `@Value` | `@Value` (Micronaut) |
| `@Configuration` + `@Bean` | `@Factory` + `@Bean` |
| `@Qualifier` | `@Named` |
| `@EventListener` | `@EventListener` (Micronaut) |
| `@Scheduled` | `@Scheduled` (Micronaut) |
| `@ConfigurationProperties` | `@ConfigurationProperties` (Micronaut) |

> **Limitation:** Spring MVC (`@RestController`, `@GetMapping`, etc.) is **not** bridged. Spring WebFlux is not bridged. The module targets library-level Spring beans, not web layer code.

---

### Data Layer

| Spring Boot | Micronaut Equivalent | Notes |
|---|---|---|
| Spring Data JPA | **Micronaut Data JPA** (`micronaut-data-hibernate-jpa`) | Generates queries at compile time; same repository pattern (`findByX`) |
| Spring Data JDBC | **Micronaut Data JDBC** | Lightweight, no Hibernate; compile-time SQL generation |
| Spring Data R2DBC | **Micronaut Data R2DBC** | Reactive, compile-time |
| Spring Data MongoDB | **Micronaut MongoDB** (`micronaut-mongo-reactive`) | Reactive driver; no compile-time query gen (dynamic) |
| Spring Data Redis | **Micronaut Redis** (Lettuce-backed) | `@RedisCache` for caching |
| Flyway / Liquibase | Same libraries, same config keys | Drop-in; no wrapper needed |
| HikariCP | Same library, auto-configured | `datasources.default.maximum-pool-size` |

Micronaut Data repository interface syntax is intentionally similar to Spring Data:

```java
// Spring Data JPA
public interface UserRepository extends JpaRepository<User, Long> {
    List<User> findByLastName(String lastName);
}

// Micronaut Data JPA  — identical interface style
@Repository
public interface UserRepository extends CrudRepository<User, Long> {
    List<User> findByLastName(String lastName);  // SQL generated at compile time
}
```

The key difference: Spring Data generates queries via `MethodNameParser` at runtime (first call); Micronaut Data emits the SQL as a string constant inside a generated class at build time.

---

### Messaging

| Spring Boot | Micronaut Equivalent |
|---|---|
| Spring Kafka (`@KafkaListener`) | `micronaut-kafka` (`@KafkaListener`) — same annotation name |
| Spring AMQP / RabbitMQ | `micronaut-rabbitmq` (`@RabbitListener`) |
| Spring Cloud Stream | No direct equivalent; use native Kafka/RabbitMQ modules |
| Spring JMS | No official module; use raw Jakarta JMS |

---

### Observability & Operations

| Spring Boot | Micronaut Equivalent |
|---|---|
| Spring Boot Actuator (`/actuator/health`) | `micronaut-management` (`/health`, `/metrics`, `/info`) |
| Micrometer (metrics) | **Same library** — `micronaut-micrometer` wraps Micrometer directly |
| Spring Cloud Sleuth / Micrometer Tracing | `micronaut-tracing` (OpenTelemetry / Zipkin / Jaeger) |
| Logback / Log4j2 | Same libraries, same `logback.xml` / `log4j2.xml` |

Micronaut adopts Micrometer natively, so dashboards and alerting built around Spring Boot metrics (Prometheus, Grafana) require no changes.

---

### Security

| Spring Security | Micronaut Security (`micronaut-security`) |
|---|---|
| `@PreAuthorize("hasRole('ADMIN')")` | `@Secured("ADMIN")` or `@Secured(SecurityRule.IS_AUTHENTICATED)` |
| `SecurityFilterChain` (programmatic) | `SecurityConfiguration` bean (programmatic) |
| JWT via `spring-security-oauth2-resource-server` | Built-in JWT validation (`micronaut-security-jwt`) |
| OAuth2 login | `micronaut-security-oauth2` |
| LDAP | `micronaut-security-ldap` |

---

### Cloud & Infrastructure

| Spring Cloud | Micronaut Equivalent |
|---|---|
| Spring Cloud Config | `micronaut-config-client` (Consul KV / Kubernetes ConfigMap / AWS Parameter Store) |
| Spring Cloud Netflix Eureka | `micronaut-discovery-client` (Eureka, Consul built-in) |
| Spring Cloud Gateway | No direct equivalent; use Micronaut HTTP proxy or external gateway |
| Spring Cloud OpenFeign | Built-in `@Client` (no external library needed) |
| Spring Cloud Circuit Breaker | `micronaut-retry` (`@Retryable`, `@CircuitBreaker`) |

---

### What Does Not Migrate

| Spring Feature | Status in Micronaut |
|---|---|
| Spring MVC web annotations (`@RestController`, `@GetMapping`) | **Not bridged** — must rewrite to Micronaut HTTP annotations |
| Spring WebFlux / Project Reactor types | Micronaut supports `Mono`/`Flux` as return types but does not use Reactor internally |
| Spring Batch | No equivalent — use Quartz scheduler or custom job runner |
| Spring Integration | No equivalent |
| Spring Session | No equivalent — implement via Redis/cookie manually |
| `ApplicationContext` programmatic API | Different API (`ApplicationContext` exists but is Micronaut's own) |
| `@Conditional` + AutoConfiguration | Replaced by `@Requires` conditions at compile time |

---

### Migration Feasibility Summary

```
Easy to migrate
  ✓ Business-logic beans (@Service, @Component, @Autowired)
  ✓ Spring Data repositories (interface style identical)
  ✓ Micrometer metrics / Actuator endpoints
  ✓ Kafka / RabbitMQ listeners
  ✓ JWT / OAuth2 security

Requires rework
  △ Web controllers (annotation rename + no ResponseEntity needed)
  △ AOP aspects (rewrite as MethodInterceptor)
  △ Spring Cloud Config / Discovery (config key changes)
  △ @Conditional auto-configuration (rewrite as @Requires)

Not available
  ✗ Spring Batch
  ✗ Spring Integration
  ✗ Spring Session
  ✗ Spring Cloud Gateway
```

