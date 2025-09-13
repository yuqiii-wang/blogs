FROM ruby:3.1-alpine

# Install dependencies
RUN apk add --no-cache \
    build-base \
    gcc \
    cmake \
    git \
    nodejs \
    npm

# Set working directory
WORKDIR /srv/jekyll

# Copy Gemfile and install gems
COPY Gemfile* ./
RUN bundle install

# Copy the rest of the site
COPY . .

# Expose port
EXPOSE 4000

# Default command
CMD ["bundle", "exec", "jekyll", "serve", "--host", "0.0.0.0", "--port", "4000", "--livereload", "--incremental", "--drafts", "--force_polling"]