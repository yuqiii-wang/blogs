# PostgreSQL

## Introduction

PostgreSQL is an advanced, open-source relational database management system (RDBMS) known for its robustness, extensibility, and standards compliance. It supports ACID transactions, complex queries, and full-text search out of the box, making it a versatile choice for applications ranging from transaction processing to data warehousing.

## PostgreSQL vs Other Relational Databases

### PostgreSQL vs MySQL

| Feature | PostgreSQL | MySQL |
|---------|-----------|-------|
| **ACID Compliance** | Full ACID support | Varies by storage engine |
| **Complex Queries** | Advanced query optimizer, window functions, CTEs | Limited to simpler queries |
| **Data Types** | Rich set (arrays, JSON, UUID, ranges) | Basic scalar types |
| **Concurrency** | Multi-version concurrency control (MVCC) | Locking-based |
| **Extension System** | Highly extensible (pgvector, PostGIS, etc.) | Limited extensibility |
| **Full-Text Search** | Native support with trigram indexing | Basic text search |
| **Performance** | Excellent for complex analytics and large datasets | Faster for simple OLTP workloads |
| **Licensing** | Open-source (PostgreSQL License) | Open-source (GPL variant) |

**Advantages:** PostgreSQL excels in complex analytical queries, custom data types, and when extensibility is critical.

## Quick Start

Start with a new database, schema, and table:

```sql
CREATE DATABASE my_database;
CREATE SCHEMA my_schema;
CREATE TABLE my_schema.my_table (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    age INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

Create a new user and grant necessary permissions.

```sql
CREATE USER my_user WITH PASSWORD 'my_secure_password';

-- Grant usage on the schema
GRANT USAGE ON SCHEMA my_schema TO my_user;

-- Grant permissions on the table
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA my_schema TO my_user;

-- Ensure future tables in the schema also get permissions
ALTER DEFAULT PRIVILEGES IN SCHEMA my_schema GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO my_user;
```

## PostgreSQL Extensions

PostgreSQL's extensibility is one of its defining strengths. Extensions add specialized functionality without core database modifications. Key areas include embeddings, knowledge graphs, full-text search, and geospatial data.

### Installing Extensions

Connect to your PostgreSQL instance and execute:

```sql
CREATE EXTENSION IF NOT EXISTS pgvector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS uuid-ossp;
CREATE EXTENSION IF NOT EXISTS graph_lib;
```

### Common Extensions by Category

#### Embeddings & Vector Search
- **`pgvector`**: Vector similarity search for AI embeddings. Essential for semantic search, RAG systems, and machine learning applications. Supports cosine, L2, and inner product distances.
- **`vecs`**: High-level vector database built on pgvector with Rust performance and Python bindings for managing embeddings at scale.

#### Knowledge Graphs & Linked Data
- **`apache_age`** (Apache AGE): Property graph database extension enabling graph queries and relationship analysis. Supports Cypher query language for knowledge graph operations.
- **`duckdb`** (integration): For analytical queries over graph-like hierarchical data structures.
- **RDF extensions**: Support for semantic web and linked data (SPARQL-based querying).

#### Full-Text Search & Text Analysis
- **`pg_trgm`**: Trigram indexing for fast fuzzy matching and full-text search optimization.
- **`unaccent`**: Text normalization removing diacritical marks for language-agnostic search.

#### Geospatial & Geometric Data
- **`PostGIS`**: Industry-standard geospatial extension supporting vector and raster data. Used for mapping, location services, and spatial analysis.

#### Utility & Infrastructure
- **`uuid-ossp`**: UUID (Universally Unique Identifier) generation functions.
- **`hstore`**: Key-value storage for semi-structured data.
- **`jsonb`** (built-in): Binary JSON with full-text search and indexing support.
- **`plpgsql`**: Procedural language for stored procedures (typically pre-installed).
- **`pg_stat_statements`**: Track query performance and execution statistics for optimization.

### Example: Vector Search with pgvector

```sql
-- Create extension
CREATE EXTENSION IF NOT EXISTS pgvector;

-- Create table with vector column
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding VECTOR(1536)  -- e.g., OpenAI embedding dimension
);

-- Create index for efficient search
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops);

-- Semantic similarity search
SELECT id, content, 
       embedding <-> '[0.1, 0.2, ...]'::vector AS distance
FROM documents
ORDER BY embedding <-> '[0.1, 0.2, ...]'::vector
LIMIT 10;
```

### Example: Knowledge Graph with Apache AGE

```sql
-- Create extension
CREATE EXTENSION IF NOT EXISTS age;

-- Create a labeled property graph
SELECT * FROM ag_catalog.create_graph('knowledge_graph');

-- Insert nodes and relationships
SELECT * FROM cypher('knowledge_graph', 
    $$ CREATE (person:Person {name: "Alice"}) RETURN person $$
) AS (person agtype);
```

## PostgreSQL Core Concepts

### Schemas

**Definition:** Schemas are logical namespaces within a database that organize database objects (tables, functions, indexes) into separate groups.

- **`public` schema**: The default schema created with every database. All users have default access unless restricted.
- **Best Practice**: Create dedicated schemas rather than using `public`. Use `CREATE SCHEMA my_schema;` to establish isolated logical domains.

**Advantages:**
- Multi-tenancy support with schema-per-tenant pattern
- Namespace collision avoidance
- Access control at schema level
- Cleaner logical organization

### ACID Compliance

PostgreSQL guarantees ACID properties through:

- **Atomicity**: Transactions are all-or-nothing via WAL (Write-Ahead Logging)
- **Consistency**: Constraints and triggers maintain data integrity
- **Isolation**: MVCC (Multi-Version Concurrency Control) allows concurrent reads without blocking
- **Durability**: Committed data survives system failures

### Advanced Features

- **Window Functions**: Compute aggregates over result sets without full grouping (`ROW_NUMBER()`, `RANK()`, `LAG()`, `LEAD()`)
- **Common Table Expressions (CTEs)**: Recursive and non-recursive WITH clauses for complex query logic
- **JSON/JSONB**: Native support for document-oriented data with full indexing
- **Inheritance**: Table inheritance for polymorphic data models
- **Triggers & Stored Procedures**: Server-side logic via PL/pgSQL

### Performance Optimization

- **Indexes**: B-tree (default), Hash, GiST, GIN, BRIN for different access patterns
- **EXPLAIN ANALYZE**: Query execution planning and cost estimation
- **Partitioning**: Table partitioning for large datasets (range, list, hash)
- **Vacuuming**: Automatic cleanup of dead tuples for storage optimization
