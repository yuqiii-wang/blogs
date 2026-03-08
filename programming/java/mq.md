# MQ (Message Queue)

## JMS

Java Message Service (JMS) has below concepts:

### Key Concepts

#### Messages

The data exchanged between systems. JMS defines standard message types, such as:

* `TextMessage` for plain text.
* `ObjectMessage` for serializable Java objects.
* `BytesMessage` for raw byte data.
* `MapMessage` for key-value pairs.
* `StreamMessage` for a stream of primitive types.

#### Destinations

* `Queues` (point-to-point): A message sent to a queue is delivered to one consumer， **one-to-one** mapping by queue name.
* `Topics` (publish-subscribe): A message sent to a topic is delivered to all subscribed consumers, **one-to-all** broadcast.

#### Connection Factories

Used to create connections to the messaging server.

#### Producers and Consumers

* MessageProducer sends messages.
* MessageConsumer receives messages.

#### JMS Providers

Implementations of the JMS API, like `Apache ActiveMQ`, `RabbitMQ`, and `IBM MQ`.

### Producer and Consumer Example: ActiveMQ

* MQ Setup

There should be an MQ broker server be up and running prior to producer and consumer start running.

In `pom.xml`, download the ActiveMQ server.

```xml
<dependency>
    <groupId>org.apache.activemq</groupId>
    <artifactId>activemq-all</artifactId>
    <version>5.17.2</version> <!-- Replace with the latest version -->
</dependency>
```

In `activemq.xml`, config the MQ, e.g., what protocols to enable on what ports.

|Connector|Best For|Key Considerations|
|-|-|-|
|OpenWire|Java clients, high-performance JMS apps|Default for most Java clients|
|AMQP (Advanced Message Queuing Protocol)|Cross-language and cross-platform systems|Requires compatible AMQP clients|
|STOMP (Simple Text Oriented Messaging Protocol)|Lightweight, web apps, non-Java clients|Text-based, simple protocol|

```xml
<transportConnectors>
    <transportConnector name="openwire" uri="tcp://0.0.0.0:61616"/>
    <transportConnector name="amqp" uri="amqp://0.0.0.0:5672"/>
    <transportConnector name="stomp" uri="stomp://0.0.0.0:61613"/>
</transportConnectors>
```

In java `.properties`, define the use of ActiveMQ for MQ JMS.

```properties
# JNDI Configuration for ActiveMQ
java.naming.factory.initial=org.apache.activemq.jndi.ActiveMQInitialContextFactory
java.naming.provider.url=tcp://localhost:61616

# Connection factories
connectionFactoryNames=ConnectionFactory

# Define destinations
queue.myQueue=MyQueue
topic.myTopic=MyTopic
```

* Producer

```java
import javax.jms.*;
import javax.naming.Context;
import javax.naming.InitialContext;
import java.util.Properties;

public class JMSProducer {
    public static void main(String[] args) throws Exception {
        // Obtain JNDI Context
        InitialContext context = new InitialContext();

        // Lookup ConnectionFactory and Queue
        ConnectionFactory connectionFactory = (ConnectionFactory) context.lookup("jms/ConnectionFactory");
        Queue queue = (Queue) context.lookup("jms/MyQueue");

        // Create Connection
        Connection connection = connectionFactory.createConnection();
        connection.start();

        // Create Session
        Session session = connection.createSession(false, Session.AUTO_ACKNOWLEDGE);

        // Create Producer
        MessageProducer producer = session.createProducer(queue);

        // Create and Send Message
        TextMessage message = session.createTextMessage("Hello, JMS!");
        producer.send(message);

        System.out.println("Message sent: " + message.getText());

        // Cleanup
        session.close();
        connection.close();
        context.close();
    }
}
```

* Consumer

```java
import javax.jms.*;
import javax.naming.InitialContext;

public class JMSConsumer {
    public static void main(String[] args) throws Exception {
        // Obtain JNDI Context
        InitialContext context = new InitialContext();

        // Lookup ConnectionFactory and Queue
        ConnectionFactory connectionFactory = (ConnectionFactory) context.lookup("jms/ConnectionFactory");
        Queue queue = (Queue) context.lookup("jms/MyQueue");

        // Create Connection
        Connection connection = connectionFactory.createConnection();
        connection.start();

        // Create Session
        Session session = connection.createSession(false, Session.AUTO_ACKNOWLEDGE);

        // Create Consumer
        MessageConsumer consumer = session.createConsumer(queue);

        // Receive Message
        TextMessage message = (TextMessage) consumer.receive();
        System.out.println("Message received: " + message.getText());

        // Cleanup
        session.close();
        connection.close();
        context.close();
    }
}
```

To use topic mode in replacement of queue, simply do

```java
Topic topic = (Topic) context.lookup("jms/MyTopic");
MessageProducer producer = session.createProducer(topic);
```

```java
Topic topic = (Topic) context.lookup("jms/MyTopic");
MessageConsumer consumer = session.createConsumer(topic);
```

* Cautions

Use transactional sessions for critical operations (ensure atomic operation):

```java
Session session = connection.createSession(true, Session.SESSION_TRANSACTED);
```

Properly close resources:

```java
producer.close();
session.close();
connection.close();
```

### MQ As Daemon Example

This example aims to show MQ producer and consumer as a persistent backend daemon processes.

1. `MQRunnable`

```java
public interface MQRunnable extends Runnable {
    void initialize(); // Initialize resources like connections and sessions
    void shutdown();   // Gracefully shut down resources
}
```

2. `MessagePublisher` and `MessageConsumer` inherits `MQRunnable`

`onMessage` is a callback defined in `MessageListener` triggered on message arrival.

```java
public interface MessagePublisher extends MQRunnable {
    void sendMessage(String message) throws Exception;
    void sendMessage(String message, int deliveryMode) throws Exception;
    void sendMessage(String message, long timeToLive) throws Exception;
}
```

```java
public interface MessageConsumer extends MQRunnable, MessageListener {
    void startListening() throws Exception; // Start consuming messages
    void stopListening() throws Exception; // Stop consuming messages
    void onMessage(javax.jms.Message message); // Override from MessageListener
}
```

3. Implementation

`run()` is the callback triggered when the associated thread is started (e.g., via `Thread.start()`).

```java
import javax.jms.*;

public class SimpleMessagePublisher implements MessagePublisher {
    private final String brokerURL;
    private final String queueName;
    private Connection connection;
    private Session session;
    private MessageProducer producer;

    public SimpleMessagePublisher(String brokerURL, String queueName) {
        this.brokerURL = brokerURL;
        this.queueName = queueName;
    }

    @Override
    public void initialize() {
        try {
            ConnectionFactory connectionFactory = new org.apache.activemq.ActiveMQConnectionFactory(brokerURL);
            connection = connectionFactory.createConnection();
            session = connection.createSession(false, Session.AUTO_ACKNOWLEDGE);
            Destination destination = session.createQueue(queueName);
            producer = session.createProducer(destination);
            producer.setDeliveryMode(DeliveryMode.PERSISTENT);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public void sendMessage(String message) throws Exception {
        sendMessage(message, DeliveryMode.NON_PERSISTENT);
    }

    @Override
    public void sendMessage(String message, int deliveryMode) throws Exception {
        TextMessage textMessage = session.createTextMessage(message);
        producer.setDeliveryMode(deliveryMode);
        producer.send(textMessage);
    }

    @Override
    public void shutdown() {
        try {
            if (producer != null) producer.close();
            if (session != null) session.close();
            if (connection != null) connection.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public void run() {
        try {
            initialize();
            for (int i = 1; i <= 10; i++) {
                sendMessage("Message " + i);
                Thread.sleep(1000); // Delay for demonstration
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            shutdown();
        }
    }
}
```

Message Consumer is

```java
import javax.jms.*;

public class SimpleMessageConsumer implements MessageConsumer {
    private final String brokerURL;
    private final String queueName;
    private Connection connection;
    private Session session;
    private javax.jms.MessageConsumer consumer;

    public SimpleMessageConsumer(String brokerURL, String queueName) {
        this.brokerURL = brokerURL;
        this.queueName = queueName;
    }

    @Override
    public void initialize() {
        try {
            ConnectionFactory connectionFactory = new org.apache.activemq.ActiveMQConnectionFactory(brokerURL);
            connection = connectionFactory.createConnection();
            session = connection.createSession(false, Session.AUTO_ACKNOWLEDGE);
            Destination destination = session.createQueue(queueName);
            consumer = session.createConsumer(destination);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public void startListening() throws Exception {
        consumer.setMessageListener(this); // Register MessageListener
        connection.start();
        System.out.println("Listening for messages...");
    }

    @Override
    public void stopListening() throws Exception {
        if (consumer != null) consumer.close();
    }

    @Override
    public void onMessage(Message message) {
        try {
            if (message instanceof TextMessage) {
                TextMessage textMessage = (TextMessage) message;
                System.out.println("Received: " + textMessage.getText());
            } else {
                System.out.println("Received non-text message");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public void shutdown() {
        try {
            stopListening();
            if (session != null) session.close();
            if (connection != null) connection.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Override
    public void run() {
        try {
            initialize();
            startListening();
            synchronized (this) {
                this.wait(); // Keep the consumer running
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            shutdown();
        }
    }
}
```

4. The Daemon main app

Inside the main app launched two daemon threads `producerThread` and `consumerThread`.

```java
public class MQDaemonApp {
    public static void main(String[] args) {
        String brokerURL = "tcp://localhost:61616";
        String queueName = "myQueue";

        // Create and start producer
        Thread producerThread = new Thread(new SimpleMessagePublisher(brokerURL, queueName));
        producerThread.setDaemon(true);
        producerThread.start();

        // Create and start consumer
        Thread consumerThread = new Thread(new SimpleMessageConsumer(brokerURL, queueName));
        consumerThread.setDaemon(true);
        consumerThread.start();

        // Keep main thread alive
        while (true) {
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }
}
```

## ActiveMQ

