# Some Advanced JAVA Topics

* Core Java vs Advanced Java

Core Java: Java Fundamentals, OOP, Collections, JDBC (Basic Database Connections)

Advanced Java: Client-Server architecture (Spring + Hibernate), Data Science (Hadoop, cloud-native)

## Callable vs Runnable

Classes implemented `Callable` and `Runnable` interfaces are supposed to be executed by another thread.

* A callable interface is more about getting a return result, that throws the checked exception.
* A runnable interface is more about using a thread running a snippet of code, does not return a result and cannot throw a checked exception.

```java
@FunctionalInterface
public interface Runnable {
    /**
     * When an object implementing interface <code>Runnable</code> is used
     * to create a thread, starting the thread causes the object's
     * <code>run</code> method to be called in that separately executing
     * thread.
     * <p>
     * The general contract of the method <code>run</code> is that it may
     * take any action whatsoever.
     *
     * @see     java.lang.Thread#run()
     */
    public abstract void run();
}
@FunctionalInterface
public interface Callable<V> {
    /**
     * Computes a result, or throws an exception
     *
     * @return computed result
     * @throws Exception if unable to compute a result
     */
    V call() throws Exception;
}
```

For example,

```java
package yuqiexample;

// RunnableDemo class implementing
// Runnable interface
public class RunnableDemo implements Runnable {
    public static void main(String[] args) {
        RunnableDemo ob1 = new RunnableDemo();
        Thread thread = new Thread(ob1);
        thread.start(); // execute `run();`
        System.out.println("Output for code outside the thread");
    }
    public void run() {
        System.out.println("Output for the part running inside the thread ");
    }
}
```

Both are applicable in `ExecutorService` for multi-threading execution via
`ExecutorService.submit(Runnable task)` and `ExecutorService.submit(Callable<T>task)`

## Interview Questions

* Question: Integer equal comparison

Explained: Integer objects with a value between -127 and 127 are cached and return same instance (same addr), others need additional instantiation hence having different addrs.

```java
class D {
   public static void main(String args[]) {
      Integer b2 = 1;
      Integer b3 = 1;
      // print True
      System.out.println(b2 == b3);

      b2 = 128;
      b3 = 128;
      // print False
      System.out.println(b2 == b3);
   }
}
```

In Java, `==` is used for reference addr/object id comparison, not for value.
For value comparison, need to use primitive type, e.g., `double`/`int`, instead of `Double`/`Integer`,
or with method such as `Double::doubleValue()`.

|Comparison|Description|Result for `a` and `b`|
|:---|:---|:---|
|`a == b`|Compares object references.|`false`|
|`a != b`|Compares object references.|`true`|
|`a.equals(b)`|Compares the primitive double values.|`true`|
|`Double.compare(a, b)`|Compares the primitive double values, returning `0` if they are equal.|`0`|

P.S. in Java 1.6, Integer calls `valueOf` when assigning an integer.

```java
public static Integer valueOf(int i) {
   if(i >= -128 && i <= IntegerCache.high)
      return IntegerCache.cache[i + 128];
   else
      return new Integer(i);
}
```

* `interface` vs `extends`

`interface`: define a contract with abstract methods that implementing/inherited classes must provide with `@override`.
There can by multiple `interface` inheritance.
Analogy of `interface` in c++ is pure abstract class.

`extends`: support single inheritance (a class can extend only one superclass).
Analogy of `extends` in c++ is derived class whose base class have some `virtual` functions and some implemented normal functions.

* Package purposes

It only serves as a path by which a compiler can easily find the right definitions.

Namespace management

* Filename is often the contained class name

One filename should only have one class.

* Type Casting

We cast the Dog type to the Animal type. Because Animal is the supertype of Dog, this casting is called **upcasting**.
Note that the actual object type does not change because of casting. The Dog object is still a Dog object. Only the reference type gets changed. 

Here `Animal` is `Dog`'s super class. When `anim.eat();`, it actually calls `dog.eat()`.

```java
Dog dog = new Dog();
Animal anim = (Animal) dog;
anim.eat();
```

Here, we cast the Animal type to the Cat type. As Cat is subclass of Animal, this casting is called **downcasting**.

```java
Animal anim = new Cat();
Cat cat = (Cat) anim;
```

Usage of downward casting, since it is more frequently used than upward casting.

Here, in the `teach()` method, we check if there is an instance of a Dog object passed in, downcast it to the Dog type and invoke its specific method, `bark()`.

```java
public class AnimalTrainer {
    public void teach(Animal anim) {
        // do animal-things
        anim.move();
        anim.eat();
 
        // if there's a dog, tell it barks
        if (anim instanceof Dog) {
            Dog dog = (Dog) anim;
            dog.bark();
        }
    }
}
```

1. Casting does not change the actual object type. Only the reference type gets changed.
2. Upcasting is always safe and never fails.
3. Downcasting can risk throwing a ClassCastException, so the instanceof operator is used to check type before casting.

* Inner Class

```java
public class C
{
   class D{ void f3(){} }
   
    D f4()
    {
        D d = new D();
        return d;
    }

    public static void main(String[] args)
    {
      // C must be instantiated before instantiate C.D
        C c = new C(); 
        C.D d = c.f4();
        d.f3();
         // D d=new D();//error!
    }
}

// Multiple class inheritance example by inner class
public class S extends C.D {} 
```

* Java Bean Concept

In computing based on the Java Platform, `JavaBeans` are classes that encapsulate many objects into a single object (the bean).

The JavaBeans functionality is provided by a set of classes and interfaces in the java.beans package. Methods include info/description for this bean.

## Java Naming and Directory Interface (JNDI)

Java Naming and Directory Interface (JNDI) gives naming convention and directory functionality.
It is useful such as in MQ and LDAP.

For example, a typical ActiveMQ `jndi.properties` file looks like

```properties
java.naming.factory.initial=org.apache.activemq.jndi.ActiveMQInitialContextFactory
java.naming.provider.url=tcp://localhost:61616

connectionFactoryNames=ConnectionFactory
queue.jms/MyQueue=MyQueue
topic.jms/MyTopic=MyTopic
```

Consider this config `queue.jms/MyQueue=MyQueue`, the `"jms/MyQueue"` is hereby set up and used in the below `JMSExample`, that the message delegation is handled by ActiveMQ run on `tcp://localhost:61616`.

```java
import javax.naming.Context;
import javax.naming.InitialContext;
import javax.jms.*;

public class JMSExample {
    public static void main(String[] args) throws Exception {
    // Create Initial Context
    Context context = new InitialContext();

    // Look up ConnectionFactory
    ConnectionFactory connectionFactory = (ConnectionFactory) context.lookup("ConnectionFactory");

    // Look up Destination (Queue or Topic)
    Queue queue = (Queue) context.lookup("jms/MyQueue");

    // Create Connection, Session, and MessageProducer
    Connection connection = connectionFactory.createConnection();
    connection.start();

    ...
    }
}
```

## Java NIO (Non Blocking IO) and EPoll

Traditionally, one thread manages one request/response.
This is wasteful since threads might get blocked by I/O operation.

Java NIO is the wrapper of Linux EPoll.

```java
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.ByteBuffer;
import java.nio.channels.SelectionKey;
import java.nio.channels.Selector;
import java.nio.channels.ServerSocketChannel;
import java.nio.channels.SocketChannel;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

public class NioServer {

    public static void main(String[] args) throws IOException {
      
        // NIO serverSocketChannel
        ServerSocketChannel serverSocketChannel = ServerSocketChannel.open();
        serverSocketChannel.bind(new InetSocketAddress(19002));

        // set non-blocking mode
        serverSocketChannel.configureBlocking(false);

        // launch epoll
        Selector selector = Selector.open();
        serverSocketChannel.register(selector, SelectionKey.OP_ACCEPT);

        while (true) {
            selector.select();
            Set<SelectionKey> selectionKeys = selector.selectedKeys();
            Iterator<SelectionKey> selectionKeyIterator = selectionKeys.iterator();
            while (selectionKeyIterator.hasNext()) {
                SelectionKey selectionKey = selectionKeyIterator.next();
                // onConnect
                if (selectionKey.isAcceptable()) {
                    ServerSocketChannel serverSocket= (ServerSocketChannel) selectionKey.channel();
                    SocketChannel socketChannel=serverSocket.accept();
                    socketChannel.configureBlocking(false);
                    socketChannel.register(selector,SelectionKey.OP_READ);
                    System.out.println("Connection established.");
                } else if (selectionKey.isReadable()) {
                    // onMessage
                    SocketChannel socketChannel= (SocketChannel) selectionKey.channel();
                    ByteBuffer byteBuffer=ByteBuffer.allocate(128);
                    int len=socketChannel.read(byteBuffer);
                    if (len>0){
                        System.out.println("Msg from client: " + new String(byteBuffer.array()));
                    } else if (len==-1){
                        System.out.println("Client disconnected: " + socketChannel.isConnected());
                        socketChannel.close();
                    }
               }
                selectionKeyIterator.remove();
            }
        }
    }
}
```

## Generics

Similar to template in cpp

```java
 public class GenericMethodTest {
   // generic method printArray
   public static < E > void printArray( E[] inputArray ) {
      // Display array elements
      for(E element : inputArray) {
         System.out.printf("%s ", element);
      }
      System.out.println();
   }

   public static void main(String args[]) {
      // Create arrays of Integer, Double and Character
      Integer[] intArray = { 1, 2, 3, 4, 5 };
      Double[] doubleArray = { 1.1, 2.2, 3.3, 4.4 };
      Character[] charArray = { 'H', 'E', 'L', 'L', 'O' };

      System.out.println("Array integerArray contains:");
      printArray(intArray);   // pass an Integer array

      System.out.println("\nArray doubleArray contains:");
      printArray(doubleArray);   // pass a Double array

      System.out.println("\nArray characterArray contains:");
      printArray(charArray);   // pass a Character array
   }
}
```

```java
public class MaximumTest {
   // determines the largest of three Comparable objects
   
   public static <T extends Comparable<T>> T maximum(T x, T y, T z) {
      T max = x;   // assume x is initially the largest
      
      if(y.compareTo(max) > 0) {
         max = y;   // y is the largest so far
      }
      
      if(z.compareTo(max) > 0) {
         max = z;   // z is the largest now                 
      }
      return max;   // returns the largest object   
   }
   
   public static void main(String args[]) {
      System.out.printf("Max of %d, %d and %d is %d\n\n", 
         3, 4, 5, maximum( 3, 4, 5 ));

      System.out.printf("Max of %.1f,%.1f and %.1f is %.1f\n\n",
         6.6, 8.8, 7.7, maximum( 6.6, 8.8, 7.7 ));

      System.out.printf("Max of %s, %s and %s is %s\n","pear",
         "apple", "orange", maximum("pear", "apple", "orange"));
   }
}
```

## JDBC (Java Database Connectivity) and ORM (Object-Relational Mapping)

* JDBC is a driver managed raw SQL parse engine.

```java
Connection conn = DriverManager.getConnection(
     "jdbc:somejdbcvendor:other data needed by some jdbc vendor",
     "myLogin",
     "myPassword");
try {
    Statement stmt = conn.createStatement();
    stmt.executeUpdate("INSERT INTO MyTable(name) VALUES ('my name')");
} finally {
    try { 
        conn.close();
    } catch (Throwable e) {
        logger.warn("Could not close JDBC Connection", e);
    }
}
```

Driver can be ODBC or native DB vendor APIs.

* ORM is the use of Java class/object by OOP (Object Oriented Programming) philosophy to manage DB records/rows via `setField()`/`getField()` to represent SQL `UPDATE`/`SELECT` operations; create/delete object to represent `INSERT`/`DELETE`.

For example,

```java
public class Employee {
   private int id;
   private String first_name; 
   private String last_name;   
   private int salary;  

   public Employee() {}
   public Employee(String fname, String lname, int salary) {
      this.first_name = fname;
      this.last_name = lname;
      this.salary = salary;
   }
   
   public int getId() {             return id; }
   public String getFirstName() {   return first_name; }
   public String getLastName() {    return last_name; }
   public int getSalary() {         return salary; }

   public void setFirstName(String _first_name) {   this._first_name = first_name; }
   public void setLastName(String _last_name) {     this._last_name = last_name; }
   public void setSalary(int _salary) {             this.salary = _salary; }
}
```

```sql
create table EMPLOYEE (
   id INT NOT NULL auto_increment,
   first_name VARCHAR(20) default NULL,
   last_name  VARCHAR(20) default NULL,
   salary     INT  default NULL,
   PRIMARY KEY (id)
);
```

The Java object to SQL mapping is through middleware.
One popular middleware is *Hibernate*.

## Java Naming and Directory Interface (JNDI)


