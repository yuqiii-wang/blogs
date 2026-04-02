# Supabase

## Introduction

Supabase is an open-source backend-as-a-service platform that provides a managed PostgreSQL database with API layer, real-time subscriptions, authentication, and storage capabilities. It maintains full PostgreSQL compatibility while adding WebSocket-based real-time updates, auto-generated REST/GraphQL APIs, multi-region replication, and managed infrastructure.

## Key Features Comparison

| Feature | Supabase | Self-hosted PostgreSQL |
|:---|:---|:---|
| **PostgreSQL Base** | Full compatibility | Native |
| **Real-time Subscriptions** | WebSocket-based | Manual configuration |
| **REST/GraphQL APIs** | Auto-generated | Manual configuration |
| **Authentication** | JWT-based, included | Manual configuration |
| **Row-Level Security** | Native PostgreSQL | Native PostgreSQL |
| **File Storage** | S3-compatible included | Manual configuration |
| **Serverless Functions** | Edge functions available | Manual configuration |
| **Connection Pooling** | PgBouncer built-in | Manual configuration |
| **Automated Backups** | Daily, 7+ day PITR | Manual configuration |
| **Multi-region Replication** | Available | Manual configuration |

## Performance & Production Features Comparison

| Feature | Supabase | Self-hosted PostgreSQL |
|:---|:---|:---|
| **Max Concurrent Connections** | 100+ (with pooling) | Unlimited (resource-dependent) |
| **Connection Pooling** | PgBouncer built-in | Manual setup required |
| **Latency (p99)** | <100ms (global) | <5ms (on-premise) |
| **Backup Frequency** | Daily, automated | Manual or custom |
| **Point-in-Time Recovery (PITR)** | 7+ days | Manual snapshots |
| **Backup Storage** | Included | Self-managed |
| **Data Replication** | Multi-region available | Manual setup (Streaming Replication) |
| **Sync Across Environments** | Via Postgres logical replication | Custom tooling |
| **Zero-downtime Backup** | Yes | Requires custom setup |
| **Disaster Recovery RTO** | <1 hour | Custom SLA |
| **Disaster Recovery RPO** | Configurable | Depends on setup |
| **High Availability** | Yes (managed) | Requires setup (Streaming Replication) |
| **Auto-scaling** | Vertical (upgrade plan) | Manual configuration |
| **Failover Time** | Automatic, <1min | Manual or custom tooling |
| **Monitoring & Alerts** | Included, advanced via add-ons | Manual or third-party tools |
| **Pricing Model** | Pay-as-you-go (compute + storage) | Infrastructure cost only |
| **Free Tier** | 500k API calls/month | Free (self-host cost) |

## When to Use Supabase

**Suitable for Supabase:**
- Applications with sub-100ms latency requirements (global)
- Systems with 100-10,000 concurrent connections requiring managed pooling
- Workloads needing 7+ day point-in-time recovery
- Multi-region deployments with automated failover
- Projects requiring authentication, real-time APIs, and storage without additional infrastructure

**Suitable for Self-hosted PostgreSQL:**
- On-premise deployments requiring <5ms p99 latency
- Systems with unlimited concurrent connection needs
- Organizations with dedicated DevOps infrastructure
- Workloads with custom backup and recovery requirements
- Deployments requiring complete control over hardware and configuration

## Engineering Guide

### Supabase: Native Cloud Setup

1. **Create project**: Go to [supabase.com](https://supabase.com), sign up, create new project
2. **Get connection details**: Project settings → Database → Connection string
3. **Connect with client library**:
   ```javascript
   import { createClient } from '@supabase/supabase-js'
   
   const supabase = createClient(
     'https://your-project.supabase.co',
     'your-public-anon-key'
   )
   ```
4. **Enable extensions**: Project dashboard → SQL Editor → Run extension creation commands
5. **Set up Row-Level Security**: Database → RLS policies (required for client access)
6. **Configure authentication**: Authentication → Providers (email, OAuth, etc.)

### Supabase: Docker Local Setup

**docker-compose.yml**:
```yaml
version: '3.8'
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: postgres
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  supabase-api:
    image: supabase/postgres:latest
    depends_on:
      - postgres
    environment:
      POSTGRES_URL: postgres://postgres:postgres@postgres:5432/postgres
    ports:
      - "3000:3000"

volumes:
  postgres_data:
```

**Setup steps**:
```bash
# Start Docker containers
docker-compose up -d

# Connect to Postgres
docker exec -it <container_name> psql -U postgres

# Inside psql, create extensions
CREATE EXTENSION IF NOT EXISTS pgvector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

**Alternative: Supabase CLI for local development**:
```bash
# Install Supabase CLI
npm install -g supabase

# Initialize local project
supabase projects create

# Start local Supabase stack
supabase start

# Connect to local database
psql postgres://postgres:postgres@127.0.0.1:54322/postgres
```

### Connecting Applications

**Node.js/JavaScript**:
```javascript
import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_ANON_KEY
)

// Query data
const { data, error } = await supabase
  .from('documents')
  .select('*')
```

**Python**:
```python
from supabase import create_client, Client

url = "postgresql://user:password@localhost:54322/postgres"
key = "your-jwt-key"

client: Client = create_client(url, key)

result = client.table("documents").select("*").execute()
```

**Raw PostgreSQL connection**:
```bash
psql "postgresql://postgres:postgres@localhost:54322/postgres"
```

