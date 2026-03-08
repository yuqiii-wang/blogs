# Spring Project Starter Development

## The Application Init

```java
@EnableConfigurationProperties
@SpringBootApplication
@ImportResource("classpath:applicationContext.xml")
@EnableScheduling
public class MyApplication {
    public static void main(String[] args) {
        SpringApplication.run(MyApplication.class, args);
    }
}
```

where

* `@SpringBootApplication` A convenience annotation that combines three others:
    * `@Configuration` – marks the class as a source of bean definitions.
    * `@EnableAutoConfiguration` – enables Spring Boot’s auto‑configuration mechanism.
    * `@ComponentScan` – enables component scanning in the package of the annotated class.
* `@ImportResource` - Imports one or more XML configuration files into the application context; SpringBoot v3 recommends using Java‑based configuration, the XML injection is deprecated, but still supported by `@ImportResource`.
* `@EnableScheduling` - It allows the detection of methods annotated with @Scheduled (e.g., `@Scheduled(fixedDelay = 5000)`) so they can be executed periodically.
* `@EnableConfigurationProperties` - Enables support for beans annotated with `@ConfigurationProperties`. These beans are automatically populated with values from property sources (like `application.properties` or `application.yml`).

### About Configuration Properties

By default, Sprint Boot loads `application.yml` from this list of folder

1. `./config/`
2. `./`
3. `src/main/resources/config`
4. `src/main/resources`

Suppose these properties are present in `application.yml`.

```yml
spring:
  application:
    name: MyApp
    version: 1.0
    features:
      - feature-a
      - feature-b
  service:
    mq-name: "MyAppMQName"
```

To bind the configs in Java bean `AppProperties`:

```java
@ConfigurationProperties(prefix = "spring.application")
public class AppProperties {
    private String name;
    private String version;
    private List<String> features = new ArrayList<>();

    // getters and setters (required for binding)
}
```

Then, `AppProperties` is a singleton in the whole spring context.

```java
@RestController
public class InfoController {
    private final AppProperties appProps;

    public InfoController(AppProperties appProps) {
        this.appProps = appProps;
    }

    @GetMapping("/info")
    public String info() {
        return appProps.getName() + " version " + appProps.getVersion();
    }
}
```

## A Loop Run

Here is an example of a service that initializes resources and starts a background task loop when the application is ready.

```java
@Service
public class AppService implements ExceptionListener {

    private static final Logger logger = LoggerFactory.getLogger(AppService.class);

    private final AtomicBoolean running = new AtomicBoolean(true);

    @Autowired
    private JmsTemplate jmsTemplate; // Assuming JmsTemplate based on context

    @Value("${spring.service.mq-name}")
    String mqName;

    @PostConstruct
    public void init() {
         // e.g., db init or other setup after dependency injection
         logger.info("Service initialized with mqName: " + mqName);
    }

    @EventListener(ApplicationReadyEvent.class)
    public void run() {
        // Start the loop in a separate thread to avoid blocking the main thread
        Thread thread = new Thread(new TaskThread());
        thread.start();
    }

    class TaskThread implements Runnable {
        @Override
        public void run() {
            while (running.get()) {
                // ... the actual loop task
                try {
                    Thread.sleep(1000); 
                } catch (InterruptedException e) {
                   Thread.currentThread().interrupt();
                }
            }
        }
    }

    @Override
    public void onException(JMSException exception) {
        logger.error("JMS Exception occurred", exception);
    }
}
```

### Explanations

*   `@Service`:
    *   Registers the class as a Spring Bean (singleton by default).
    *   Indicates that this class holds **business logic**.
    *   Allows it to be injected into other beans (like Controllers) using `@Autowired`.
*   `@PostConstruct`: The method annotated with this runs automatically after the bean is constructed and all dependencies (like `@Autowired` and `@Value`) are injected.
*   `@Value("${...}")`: Injects a property value from `application.yml` or `application.properties` directly into the variable.
*   `@EventListener(ApplicationReadyEvent.class)`: This ensures the `run()` method is triggered only when the Spring Boot application has fully started and is ready to create threads or handle traffic.
*   `AtomicBoolean`: Used instead of a regular `boolean` to ensure thread-safety when the `running` flag is accessed or modified by multiple threads (e.g., stopping the loop from a shutdown hook).

#### `@Value` vs `@Autowired`

| Feature | `@Autowired` | `@Value` |
| :--- | :--- | :--- |
| **Purpose** | Dependency Injection | Configuration Injection |
| **Injects** | Spring Beans | Scalar values (Strings, Numbers, Booleans) |
| **Source** | Spring ApplicationContext | `application.properties`, `.yml`, or system env |

## Rest API

Spring Boot makes it easy to create RESTful web services. Here is a typical controller:

```java
@RestController
@RequestMapping("/api/users")
public class UserController {

    private final UserService userService;

    // Constructor injection is recommended over field injection (@Autowired)
    public UserController(UserService userService) {
        this.userService = userService;
    }

    @GetMapping("/{id}")
    public ResponseEntity<User> getUser(@PathVariable Long id) {
        User user = userService.findById(id);
        return ResponseEntity.ok(user);
    }

    @PostMapping
    public ResponseEntity<User> createUser(@RequestBody User user) {
        User createdUser = userService.save(user);
        return ResponseEntity.status(HttpStatus.CREATED).body(createdUser);
    }

}
```

### Annotation Explanations

*   `@RestController`: This is a convenience annotation that combines `@Controller` and `@ResponseBody`. It indicates that the class is a web controller and that the return value of the methods should be written directly to the HTTP response body (automatically serialized to JSON), rather than resolving to a view (like HTML/JSP).
*   `@RequestMapping("/api/users")`: Defines the base URL path for all endpoints in this class.
*   `@GetMapping` / `@PostMapping`: Shortcuts for `@RequestMapping(method = RequestMethod.GET)` and `RequestMethod.POST`. They map specific HTTP verbs to methods.

### Thread Pool for REST API & Multi-threading

**The Flow**:
1. (Tomcat): Tomcat's connector accepts a TCP connection.
2. (Tomcat): It grabs a worker thread from its pool (e.g., named `http-nio-8080-exec-1`).
3. (Spring): Tomcat calls Spring's `DispatcherServlet.service()` **on that same thread**.
4. (Biz Code): `@RestController` method runs **on that same thread**.
5. (Response): The return value is written to the socket **on that same thread**.

Since Tomcat (the container) is multi-threaded, **Spring Boot is inherently multi-threaded** for each request uses ONE same thread from Tomcat to Spring Boot then back to Tomcat.
In other words, by default multi-threading in Spring Boot refers to multi-request, but one request has ONLY one mapping to thread.

By default Tomcat can handle 200 concurrent requests by the time of 2026.

**Implication for Developers**:
*   Controllers and Services are **Singletons**.
*   Since the *same* instance is used by multiple threads at once, they must be **Stateless**.
*   **Never** use class fields to store request-specific data (like `private User currentUser;`). Always pass data through method arguments.
*   If requests are heavy, a biz-logic dedicated thread pool can be constructed to just serve EACH request.

## UI Rendering (for React as An Example)

In a modern "Separation of Concerns" architecture, Spring Boot usually acts purely as a REST API (JSON provider), while a frontend framework like React (or Angular/Vue) handles the UI rendering.

For production, it is common to **bundle the React build inside the Spring Boot JAR** so that a single artifact serves both the API and the UI.

1.  **Build React**: Run `npm run build`. This generates static files in `build/` (index.html, js, css).
2.  **Copy to Spring**: Move these files to `src/main/resources/static` or `src/main/resources/public` in the Spring Boot project.
3.  **Spring Boot Auto-configuration**: Spring Boot automatically serves static content from `classpath:/static/`.

When a user visits the host, Spring serves `index.html` (the React App). When the React App makes a fetch call to `/api/users`, it hits the Spring controller on the same domain, so no CORS is needed.

### Order of Path Routing to REST API vs UI Rendering

Spring Boot by this priority order to route requests:

1.  **Controller Priority**: Spring first checks if the URL matches a `@RestController` path (e.g., `/api/users`). If found, it executes the method.
2.  **Static Content**: If no controller matches, it searches for the file in `src/main/resources/static/` (e.g., `style.css`).

Modern UI is usually Single Page Application (SPA) that all user interactions are done within the one page by "reactive" UI components.
As a result, Spring Boot just need ONE path to host UI/static contents.
