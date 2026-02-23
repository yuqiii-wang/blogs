# Spring

## Quick Intro: Spring vs Spring Boot vs Spring Cloud

### Spring Framework

* Build on core java for web application
* Provide Inversion of Control (IoC) and Dependency Injection (DI)
* Aspect-Oriented Programming (AOP): a programming paradigm that aims to increase modularity by allowing the separation of cross-cutting concerns. In Spring Framework, it addresses concerns such as logging, transaction management, security, etc., separately from the business logic.
* provides a comprehensive transaction management JDBC, JPA, JTA, etc.
* Spring MVC (Model-View-Controller): built on top of the core Spring framework

### Spring Boot

* Simplifies Spring application development with auto-configuration
* Includes embedded servers (like Tomcat or Jetty)

### Spring Cloud

* Build on Spring Boot
* For distributed systems and microservices, especially in cloud environments
* Characterized by service discovery, circuit breakers, intelligent routing, distributed sessions, etc.

### Spring Version 3.x.x vs 2.x.x

|Feature/Component|Spring Boot 2.x|Spring Boot 3.x|
|:---|:---|:---|
|Java Baseline|Java 8 (supports up to Java 19)|Java 17 (supports Java 21+)|
|Spring Framework|Spring Framework 5.x|Spring Framework 6.x|
|Enterprise Java|Java EE / J2EE (`javax.*`)|Jakarta EE 10 (`jakarta.*`)|

where

* The Java 17

Spring Boot 3 requires **Java 17** as a **minimum**.
Under the hood, it relies on Spring Framework 6, which removes a lot of legacy code and deprecated APIs.

* `javax` vs. `jakarta`

`java` was originally owned by oracle, but in 2017, Oracle decided to step away from enterprise Java and donated the Java EE project to the Eclipse Foundation, then there was a massive legal and trademark shift in the Java ecosystem.
The Eclipse Foundation renamed "Java EE" to "Jakarta EE" and changed the package namespace for all future development to `jakarta.*`.

## Spring Servlet and HTTP Handling

In Spring, the handling of HTTP requests, e.g., by multiple `@Controller` and `@FeignClient` components, the underlying management of these requests is handled by the embedded servlet container (e.g., *Tomcat*, *Jetty*).

### Spring MVC Request Handling

Spring Boot uses Spring MVC to manage HTTP requests.
When a request is made to a Spring Boot application, the following happens:

1. Request Reception

The embedded servlet container (typically Tomcat by default) listens for incoming HTTP requests. When a request is received, the container hands it over to the Spring framework.

The tomcat container's main connector (typically *Http11NioProtocol* or *Http11Protocol*) listens on a specific port (default 8080).

Tomcat manages incoming requests such as by this config in `.properties`.

```properties
server.tomcat.max-threads=200
server.tomcat.connection-timeout=20000
```

2. DispatcherServlet

The `DispatcherServlet` in Spring Boot acts as the central component that receives all HTTP requests. It is responsible for routing requests to the appropriate controller methods. It checks the request URL, HTTP method (GET, POST, etc.), and other factors to determine the matching `@RequestMapping` or `@GetMapping` etc., annotations on your controller methods.

Spring Boot automatically configures the DispatcherServlet when used `@SpringBootApplication` or `@EnableAutoConfiguration`.

3. Controller Resolution

DispatcherServlet uses the HandlerMapping mechanism to map incoming requests to controller methods.
If the request matches the path defined in a controller method's annotation (e.g., `@GetMapping("/users")`), the request is passed to the corresponding method in the controller.

DispatcherServlet returns an HTTP 404 error if path not found.

4. Execution of the Controller Method

The controller method is executed in the same thread that received the HTTP request (by default). If the method is annotated with `@Async` (for asynchronous execution), it will be delegated to a separate thread from the configured thread pool.

5. Response Generation

Once the controller method executes, a response (usually in the form of a *ModelAndView*, e.g., jsp, or an object that Spring Boot automatically serializes to JSON/XML) is generated (for RESTful API). This response is then sent back to the client via the servlet container.

### Multi-Threading

The servlet container (e.g., Tomcat) uses a thread pool to handle incoming requests.
Spring itself is just a request handler dispatcher.

If `@Async` is not specified, Tomcat and Spring share the same thread.

If Tomcat not set, by default, there are

* Max threads: 200 (default)
* Min spare threads: 10

## Spring Context and Inversion of Control (IoC)

When java objects are under spring context, they are called *beans* and their lifecycles and dependencies are managed by spring.

Inversion of Control (IoC) refers to the process where the control of object creation and the management of their dependencies is transferred from the application code to a container or framework.

Spring uses `@Bean` to take control of an object, such as setting its member values and managing object life cycle.

A spring context starts with `@SpringBootApplication`, or in test `@SpringBootTest`.

```java
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class MySpringBootApplication {
    public static void main(String[] args) {
        SpringApplication.run(MySpringBootApplication.class, args);
    }
}
```

Beans can be defined using annotations like `@Component`, `@Service`, `@Repository`, `@Controller`, or `@Bean` methods in @Configuration classes.

### Multiple Bean Definition Resolution

When Spring encounters multiple beans of the same type, it cannot decide which one to inject automatically.
This will result in a `NoUniqueBeanDefinitionException` unless:

* Use `@Primary` to mark one bean as the default.
* Use `Qualifier` to explicitly specify which bean to inject.

### Reflection

In Java, reflection can be used to directly "retrieve"/"construct" a class or the class' member, bypassing `private` scope and just creating an object without having even known this class.

For example, below code shows directly accessing `java.util.ArrayList`'s member function and creating a new class without explicitly invoking the constructor.

```java
import java.lang.reflect.Constructor;

public class ReflectionExample {
    public static void main(String[] args) throws Exception {
        Class<?> clazz = Class.forName("java.util.ArrayList");
        System.out.println("Class name: " + clazz.getName());

        // Create a new instance of ArrayList
        Constructor<?> constructor = clazz.getConstructor();
        Object obj = constructor.newInstance();
        System.out.println("Created object: " + obj);
    }
}
```

In Spring Boot, reflection is used in the context of *dependency injection* (DI), where the framework needs to create and wire together beans.

### Dependency Injection (DI)

Dependency Injection (or sometime called wiring) helps in gluing independent classes together and at the same time keeping them independent (decoupling).

```java
// shouldn't be this
public class TextEditor {
   private SpellChecker spellChecker;
   public TextEditor() {
      spellChecker = new SpellChecker();
   }
}

// instead, should be this, so that regardless of changes of SpellChecker class, 
// there is no need of changes to SpellChecker object implementation code 
public class TextEditor {
   private SpellChecker spellChecker;
   public TextEditor(SpellChecker spellChecker) {
      this.spellChecker = spellChecker;
   }
}
```

where the input arg `SpellChecker spellChecker` should be passed in (likely already instantiated elsewhere) rather than be `new()` every time.

### IoC container

In the Spring framework, the interface ApplicationContext represents the IoC container. The Spring container is responsible for instantiating, configuring and assembling objects known as *beans*, as well as managing their life cycles.

In order to assemble beans, the container uses configuration metadata, which can be in the form of XML configuration or annotations, such as setting up attributes for this bean in `applicationContext.xml`, which is loaded by `ClassPathXmlApplicationContext`.

```java
ApplicationContext context
  = new ClassPathXmlApplicationContext("applicationContext.xml");
```

### Dependency Injection Examples

* bean-Based

If we don't specify a custom name, then the bean name will default to the method name.

```java
@Configuration
public class TextEditor {

   private SpellChecker spellChecker;

   @Bean
   public TextEditor() {
      spellChecker = new SpellChecker();
   }
}
```

#### `@Autowired`

`@Autowired` is used to perform injection

When the Spring context is initialized, it starts creating instances of the beans defined in the configuration (either XML or Java-based configuration).

It scans for classes annotated with `@Component`, `@Service`, `@Repository`, and other stereotype annotations, which are automatically registered as Spring beans.

For each bean, Spring checks for the `@Autowired` annotation on constructors, and tries to resolve and inject all the constructor parameters (also same to field and setter).

Spring uses reflection to set the field values or call the constructor/setter methods with the resolved dependencies.
The ApplicationContext (or BeanFactory) is responsible for managing the lifecycle and scope of beans and resolving dependencies as needed.

* P.S. `@Autowired` is only triggered when in `@SpringApplicationContext` (as well as in `@SpringBootTest`).

## Config

### The Config File `application.properties`/`application.yml`

Spring applications by default load from `application.properties`/`application.yml`.

#### Location of `application.yml`

Spring Boot looks for the `application.yml` file in the following locations (in order of priority):

* `/config` directory inside the project root (highest priority).
* Project root directory.
* `/config` package inside the classpath.
* Classpath root (lowest priority).

You can also specify a custom location for the configuration file using the `spring.config.location` property.

#### Retrieve by `@Value`

Items in `application.properties` are auto mapped in spring framework via `@Value`.

For example, in `application.properties`

```yml
app:
  name: My Spring Boot Application
  version: 1.0.0
```

To use it:

```java
@Value("${app.name}")
private String appName;
```

#### Retrieve by `@ConfigurationProperties`

To use it:

```java
@ConfigurationProperties(prefix = "app")
public class AppConfig {
    private String name;
    private String version;

    // Getters and setters
}
```

#### Retrieve by `Environment`

```java
@Autowired
private Environment env;

public void printAppName() {
    String appName = env.getProperty("app.name");
    System.out.println(appName);
}
```

### `@Profile`

`@Profile` allows user to map beans to different profiles, typically diff envs, e.g., dev, test, and prod.

In `application.properties`, config the env.

```conf
spring.profiles.active=dev
```

In implementation, only used in `dev`.

```java
@Component
@Profile("dev")
public class DevDatasourceConfig { ... }
```

or `dev` is NOT active.

```java
@Component
@Profile("dev")
public class NonDevDatasourceConfig { ... }
```

### `@Configuration`, and `@bean` vs `@Component`

#### `@Configuration` and `@bean`

`@Bean` is used within a `@Configuration` class to explicitly declare a bean.
`@bean` is primitive compared to `@Component`, hence provided fine-grained control over instantiation.

In Spring, instantiated beans have a `singleton` scope by default.
This is problematic, as exampled in below when `clientDao()` is called once in `clientService1()` and once in `clientService2()`, but only one singleton instance is returned.

`@Configuration` comes in rescue that beans under `@Configuration`-annotated `AppConfig` will see instantiations of two beans.

```java
@Configuration
public class AppConfig {

    @Bean
    public ClientService clientService1() {
        ClientServiceImpl clientService = new ClientServiceImpl();
        clientService.setClientDao(clientDao());
        return clientService;
    }

    @Bean
    public ClientService clientService2() {
        ClientServiceImpl clientService = new ClientServiceImpl();
        clientService.setClientDao(clientDao());
        return clientService;
    }

    @Bean
    public ClientDao clientDao() {
        return new ClientDaoImpl();
    }
}
```

#### `@bean` vs `@Component`

* `@Component`

Be automatically detected and managed by Spring.

For application-specific classes such as services, repositories, and controllers.

* `@bean`

Need to configure beans for third-party libraries or have fine-grained control over bean instantiation.

## Spring IOC Loading Process
