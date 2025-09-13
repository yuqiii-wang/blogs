#!/bin/bash

# Docker script to build and run Jekyll using custom Dockerfile
# This builds a custom image with your specific Gemfile dependencies

set -e  # Exit on any error

# Configuration
CONTAINER_NAME="jekyll-blog-custom"
IMAGE_NAME="jekyll-blog"
HOST_PORT="4000"
CONTAINER_PORT="4000"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building and running Jekyll with custom Docker image...${NC}"

# Stop and remove existing container if it exists
if docker ps -a --format 'table {{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${YELLOW}Stopping existing container...${NC}"
    docker stop ${CONTAINER_NAME} 2>/dev/null || true
    docker rm ${CONTAINER_NAME} 2>/dev/null || true
fi

# Build the Docker image
echo -e "${BLUE}Building Docker image...${NC}"
docker build -t ${IMAGE_NAME} .

# Run the container
echo -e "${GREEN}Starting Jekyll server on http://localhost:${HOST_PORT}${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"

docker run --name ${CONTAINER_NAME} \
    --rm \
    -it \
    -p ${HOST_PORT}:${CONTAINER_PORT} \
    -v "$(pwd):/srv/jekyll" \
    -v "$(pwd)/_site:/srv/jekyll/_site" \
    -e JEKYLL_ENV=development \
    ${IMAGE_NAME}

echo -e "${RED}Jekyll server stopped.${NC}"