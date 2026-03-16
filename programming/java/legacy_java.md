# Legacy Java

## WebSphere

## IBM MQ and `.bindings`

IBM MQ is an enterprise-grade message-oriented middleware (MOM) that enables secure, reliable, and asynchronous communication between distributed applications via messages and queues. 

In legacy Java and JMS (Java Message Service) ecosystems, `.bindings` files operate as a file system-based Java Naming and Directory Interface (JNDI) provider. They act as a local registry storing serialized administrative objects—such as connection factories and destination targets—allowing applications to look up MQ resources dynamically without relying on a centralized directory server (e.g., LDAP).

### Example JNDI and `.bindings`

Below is an example of how a Java application uses the File System JNDI context provider to load configuration from a `.bindings` file:

```java
import javax.jms.*;
import javax.naming.*;
import java.util.Hashtable;

public class MQBindingsExample {
    public static void main(String[] args) {
        Hashtable<String, String> env = new Hashtable<>();
        
        // Specify the File System JNDI service provider
        env.put(Context.INITIAL_CONTEXT_FACTORY, "com.sun.jndi.fscontext.RefFSContextFactory");
        
        // Set the directory path where the .bindings file is located
        env.put(Context.PROVIDER_URL, "file:/path/to/bindings/directory");

        try {
            // Initialize the JNDI context using the .bindings file
            Context ctx = new InitialContext(env);
            
            // Look up the ConnectionFactory and Destination (Queue/Topic) by their JNDI names
            ConnectionFactory cf = (ConnectionFactory) ctx.lookup("jms/MyConnectionFactory");
            Destination dest = (Destination) ctx.lookup("jms/MyQueue");

            // Standard JMS API usage follows
            try (Connection connection = cf.createConnection();
                 Session session = connection.createSession(false, Session.AUTO_ACKNOWLEDGE)) {
                
                MessageProducer producer = session.createProducer(dest);
                TextMessage message = session.createTextMessage("Hello IBM MQ via .bindings!");
                producer.send(message);
                
                System.out.println("Message sent successfully.");
            }
        } catch (NamingException | JMSException e) {
            e.printStackTrace();
        }
    }
}
```

### Example `.bindings` File Content

A `.bindings` file is typically generated using MQ administration tools (like `JMSAdmin`). The file itself is essentially a human-readable properties file mapping JNDI names to serialized Java objects. 

```properties
# This is a sample .bindings file snippet

# Define the ConnectionFactory
jms/MyConnectionFactory/ClassName=com.ibm.mq.jms.MQQueueConnectionFactory
jms/MyConnectionFactory/FactoryName=com.ibm.mq.jms.MQQueueConnectionFactoryFactory
jms/MyConnectionFactory/RefAddr/0/Content=QM1
jms/MyConnectionFactory/RefAddr/0/Type=QUEUE_MANAGER
jms/MyConnectionFactory/RefAddr/1/Content=1414
jms/MyConnectionFactory/RefAddr/1/Type=PORT
jms/MyConnectionFactory/RefAddr/2/Content=mq.example.com
jms/MyConnectionFactory/RefAddr/2/Type=HOSTNAME
jms/MyConnectionFactory/RefAddr/3/Content=SYSTEM.DEF.SVRCONN
jms/MyConnectionFactory/RefAddr/3/Type=CHANNEL
jms/MyConnectionFactory/RefAddr/4/Content=1
jms/MyConnectionFactory/RefAddr/4/Type=TRANSPORT

# Define the Queue (Destination)
jms/MyQueue/ClassName=com.ibm.mq.jms.MQQueue
jms/MyQueue/FactoryName=com.ibm.mq.jms.MQQueueFactory
jms/MyQueue/RefAddr/0/Content=DEV.QUEUE.1
jms/MyQueue/RefAddr/0/Type=QU
jms/MyQueue/RefAddr/1/Content=QM1
jms/MyQueue/RefAddr/1/Type=QMGR
```

#### Understanding `Content` and `Type` in `.bindings`

In the `.bindings` format, the entries represent serialized `javax.naming.Reference` objects. These objects are built using multiple `RefAddr` (Reference Address) properties, grouped by a numerical index (e.g., `/0/`, `/1/`).

* **`Type`**: Specifies the property key or configuration parameter being set. For a ConnectionFactory, this might be `HOSTNAME`, `PORT`, `CHANNEL`, or `QUEUE_MANAGER`. For a Queue, it might be `QU` (Queue Name).
* **`Content`**: Specifies the actual value assigned to that parameter. For example, the `Content` for a `PORT` type would be `1414`, and for a `CHANNEL` type, it would be `SYSTEM.DEF.SVRCONN`.

Together, `Type` and `Content` define the key-value pairs needed by the MQ JMS provider to instantiate the target MQ objects when the application performs a JNDI lookup.
