#!/bin/bash

# Docker script to run Jekyll as a static web server
# This script builds and serves your Jekyll site using Docker

set -e  # Exit on any error

# Configuration
CONTAINER_NAME="jekyll-blog"
IMAGE_NAME="jekyll/jekyll:latest"
HOST_PORT="4000"
CONTAINER_PORT="4000"
SITE_DIR="/srv/jekyll"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Jekyll with Docker...${NC}"

# Stop and remove existing container if it exists
if docker ps -a --format 'table {{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${YELLOW}Stopping existing container...${NC}"
    docker stop ${CONTAINER_NAME} 2>/dev/null || true
    docker rm ${CONTAINER_NAME} 2>/dev/null || true
fi

# Get the current directory (should be the blog root)
BLOG_DIR="$(pwd)"

echo -e "${GREEN}Blog directory: ${BLOG_DIR}${NC}"
echo -e "${GREEN}Starting Jekyll server on http://localhost:${HOST_PORT}${NC}"

# Run Jekyll in Docker container
docker run --name ${CONTAINER_NAME} \
    --rm \
    -p ${HOST_PORT}:${CONTAINER_PORT} \
    -v "${BLOG_DIR}:${SITE_DIR}" \
    -e JEKYLL_ENV=development \
    ${IMAGE_NAME} \
    jekyll serve \
        --host 0.0.0.0 \
        --port ${CONTAINER_PORT} \
        --livereload \
        --incremental \
        --drafts \
        --force_polling

echo -e "${RED}Jekyll server stopped.${NC}"