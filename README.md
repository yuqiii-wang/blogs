# Yuqi's Knowledge Blogs

A comprehensive personal knowledge base and blog covering various technical topics, built with MkDocs and deployed to GitHub Pages.

## Live Site

https://yuqiii-wang.github.io/blogs

## Project Structure

```
KnowledgeNotes/
├── database/         # Database-related notes (SQL, NoSQL, clustering)
├── dev_ops/          # DevOps tools and practices (AWS, Docker, K8s)
├── fundamental_computer/ # Computer science fundamentals
├── math/             # Mathematics topics (AI, ML, Statistics, Algebra)
├── mgt_and_design/   # Project management and design concepts
├── network/          # Network-related topics (Blockchain)
├── javascripts/      # JavaScript files for the site
├── .github/          # GitHub Actions workflow
├── prepare_site.py   # Script to prepare site content
├── mkdocs.yml        # MkDocs configuration
└── README.md         # This file
```

## Main Categories

### Database
- SQL (MySQL, PostgreSQL, SQLite)
- NoSQL (MongoDB, Redis, Elasticsearch, LDAP)
- Database clustering and sharding
- Indexing and optimization techniques

### DevOps
- Cloud DevOps (AWS, Docker, Kubernetes, Terraform)
- Linux DevOps (Bash, Linux tools, networking)
- Performance optimization

### Fundamental Computer Science
- Hardware (CPU, GPU, disk, motherboard)
- Operating Systems (Linux, macOS, memory management)
- Computer terminologies

### Mathematics
- AI/ML (Computer Vision, NLP, Deep Learning)
- Linear Algebra (Matrices, Vectors)
- Statistics and Probability
- Finance and Quantitative Trading
- SLAM (Simultaneous Localization and Mapping)

### Management and Design
- UML diagrams
- Project management methodologies
- IT Office Concepts

### Network
- Blockchain technology
- Message queues
- HTTP headers

## Getting Started

### Prerequisites
- Python 3.x
- MkDocs
- Material for MkDocs theme

### Local Development

1. **Set up the environment**:
   ```bash
   export PYTHONIOENCODING=utf-8
   python -m pip install -r requirements.txt
   ```

2. **Prepare the site content**:
   ```bash
   python prepare_site.py
   ```

3. **Start the development server**:
   ```bash
   python -m mkdocs serve --dev-addr 127.0.0.1:8080
   ```

4. **Access the site**:
   Open your browser and navigate to `http://localhost:8080`

## Deployment

The site is automatically deployed to GitHub Pages via GitHub Actions when changes are pushed to the `main` or `master` branch.

## How It Works

1. **Content Organization**: Markdown files are organized in a hierarchical directory structure
2. **Site Preparation**: The `prepare_site.py` script copies relevant files to the `docs_collection` directory
3. **Build Process**: MkDocs builds the static site from the prepared content
4. **Deployment**: GitHub Actions handles the deployment to GitHub Pages

## Contributing

This is a personal knowledge base, but feel free to fork and adapt it for your own use.

## License

MIT
