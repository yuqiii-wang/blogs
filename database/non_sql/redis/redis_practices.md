# Redis

Redis is high-performance, RAM-running, key-value cache server.

* Guaranteed that each retrieval of data entry is $O(1)$, since each retrieval is by hash map.

* Each key can only associated with one data type, for example, Redis throws err when prev value for a key is a FLOAT, while using a STR method updating the value for the key. Consider remove the key first.

## Quick Start

By Docker-Compose:
```yml
version: '3.7'

services:

  redis:
    image: redis
    restart: always
    environment:
      REDIS_USER: root
      REDIS_PASSWORD: P@ssw0rdRoot
    volumes:
      - redisdata:/home/yuqi/Desktop/QuantAlgo/data/redis
    ports:
      - 6379:6379

volumes:
  redisdata:
    external: true
```

By default, redis allows login by password such as `redis-cli -h 127.0.0.1 -a P@ssw0rdRoot`

Once login, use `ACL SETUSER user ON >P@ssw0rdUser` to set up a new user, 
then `ACL SETUSER user ~*` to grant access privileges,
then `ACL LIST` to check existing users.

Alternatively, run `CONFIG SET requirepass "P@ssw0rd"`
then `AUTH P@ssw0rd` to access redis db.

## Config

Usually on file `redis-stable/redis.conf`.

### Persistence
Redis Persistence is about making data persistent on disk from RAM. By default, Redis DB stores persistent data on `dump.rdb`. Data runs on RAM, and are persisted dictated by `redis.conf`. By default, it says 100 changes since last 300 secs.

### Replications
Replication is a method by which other servers receive a continuously updated copy of the data as it’s being written, so that the replicas can service read queries. Redis uses master-to-slave arch.

Redis does not support master-to-master communications.

### Transactions
Transaction in general refers to operations with consistent results, such as MySql starting with `BEGIN` followed by multiple read/writes. Redis uses `MULTI / EXEC` to express transactions.

## DataTypes

### String

* Byte string values, such as
```bash
APPEND keynameStr hello # create a string "hello" associated/referenced with the keyname
APPEND keynameStr world # now this key is associated/referenced with "helloworld"
```

* Integer values, such as
```bash
INCR keynameInt # increase the value stored/referenced for the keyname by 1
INCRBY keynameInt 20 # increase the value stored/referenced for the keyname by 20
```

* Floating-point values
```bash
INCRBYFLOAT keynameFloat 0.1 # set/referenced keyname to 0.1
```

### List

Same as a list ... operates by `push`, `pop`, and `range`. 

Items of a list can be of different types.

```bash
RPUSH keynameList 1 # Left end push an item into the list named keynameList
LPUSH keynameList ahhh # Right end push a string
```

### Set

Set is used to store unique items. Items can be of different data types.

```bash
SADD keynameSet 1 # Add int 1 to a set called "keynameSet"
SADD keynameSet ahhh # Add another item of a string type
SISMEMBER keynameSet ahhh # Check if ahhh exists in the set
```

### Hash

set groups of key-value pairs in a single higher-level Redis key.
```bash
HMSET person1 name John age 31 gender male # person1 has many fields
```

### Sorted Set
Sorted set (`ZSET`) is a set whose members are sorted. It can be used as index to find keys that are associated with more data.

## Aggregates

Aggregation needs external redis module (this module also provides many advanced query features):
```bash
redis-server --loadmodule /path/to/module/src/redisearch.so
```

Build a pipeline of operations that transform the results by zero or more steps of:
* Group and Reduce: grouping by fields in the results, and applying reducer functions on each group.
* Sort: sort the results based on one or more fields.
* Apply Transformations: Apply mathematical and string functions on fields in the pipeline, optionally creating new fields or replacing existing ones
* Limit: Limit the result, regardless of sorting the result.
* Filter: Filter the results (post-query) based on predicates relating to its values.

### Practices

```bash
# create index
FT.CREATE myIdx ON HASH PREFIX 1 doc: SCHEMA title TEXT WEIGHT 5.0 body TEXT url TEXT

# add one item
hset doc:1 title "hello world" body "lorem ipsum" url "http://redis.io" 

# search by index
FT.SEARCH myIdx "hello world" LIMIT 0 10
```