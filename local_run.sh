#!/bin/bash

# Jekyll Local Development Setup and Run Script
# This script sets up and runs a Jekyll static website locally

echo "🚀 Starting Jekyll local development setup..."

# Function to install Ruby on Windows
install_ruby_windows() {
    echo "🔧 Installing Ruby for Windows..."
    
    # Check if we're running on Windows
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
        # Check if winget is available (Windows Package Manager)
        if command -v winget &> /dev/null; then
            echo "📦 Using winget to install Ruby..."
            winget install RubyInstallerTeam.Ruby.3.3 --accept-source-agreements --accept-package-agreements
        # Check if chocolatey is available
        elif command -v choco &> /dev/null; then
            echo "📦 Using Chocolatey to install Ruby..."
            choco install ruby -y
        # Check if scoop is available
        elif command -v scoop &> /dev/null; then
            echo "� Using Scoop to install Ruby..."
            scoop install ruby
        else
            echo "❌ No package manager found (winget, chocolatey, or scoop)."
            echo "Please install Ruby manually from: https://rubyinstaller.org/"
            echo ""
            echo "Or install a package manager:"
            echo "- winget (built into Windows 10/11)"
            echo "- chocolatey: https://chocolatey.org/install"
            echo "- scoop: https://scoop.sh/"
            exit 1
        fi
        
        # Refresh PATH to include newly installed Ruby
        echo "🔄 Refreshing environment variables..."
        source ~/.bashrc 2>/dev/null || true
        export PATH="/c/Ruby33-x64/bin:$PATH"
        
    else
        echo "❌ This auto-install is designed for Windows. Please install Ruby manually."
        exit 1
    fi
}

# Check if Ruby is installed
if ! command -v ruby &> /dev/null; then
    echo "❌ Ruby is not found in PATH."
    
    # Check common Ruby installation paths on Windows
    RUBY_PATHS=(
        "/c/Ruby33-x64/bin"
        "/c/Ruby32-x64/bin"
        "/c/Ruby31-x64/bin"
        "/c/Ruby30-x64/bin"
        "/c/tools/ruby33/bin"
        "/c/tools/ruby32/bin"
    )
    
    RUBY_FOUND=""
    for path in "${RUBY_PATHS[@]}"; do
        if [ -f "$path/ruby.exe" ] || [ -f "$path/ruby" ]; then
            RUBY_FOUND="$path"
            echo "✅ Found Ruby installation at: $path"
            break
        fi
    done
    
    if [ -n "$RUBY_FOUND" ]; then
        echo "🔧 Adding Ruby to PATH for this session..."
        export PATH="$RUBY_FOUND:$PATH"
        
        # Verify Ruby is now accessible
        if command -v ruby &> /dev/null; then
            echo "✅ Ruby is now accessible!"
        else
            echo "❌ Failed to add Ruby to PATH. Please add $RUBY_FOUND to your system PATH."
            exit 1
        fi
    else
        echo "❌ Ruby installation not found in common locations."
        read -p "🤔 Would you like to install Ruby automatically? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            install_ruby_windows
            
            # Verify installation
            if ! command -v ruby &> /dev/null; then
                echo "❌ Ruby installation failed or PATH not updated."
                echo "Please restart your terminal and try again, or install Ruby manually."
                echo "Visit: https://rubyinstaller.org/ for manual installation"
                exit 1
            fi
        else
            echo "❌ Ruby installation cancelled."
            echo "Visit: https://rubyinstaller.org/ for manual installation"
            exit 1
        fi
    fi
fi

echo "✅ Ruby successfully detected!"
echo "📋 Ruby version: $(ruby --version)"

# Configure RubyGems mirror for faster downloads in China
echo "🔧 Configuring RubyGems mirror..."
echo "🌐 Setting up Tsinghua University mirror for faster downloads..."

# Check current gem sources
echo "📋 Current gem sources:"
gem sources -l

# Remove default source and add Tsinghua mirror
echo "📦 Configuring RubyGems to use Tsinghua mirror (mirrors.tuna.tsinghua.edu.cn/rubygems/)..."
gem sources --remove https://rubygems.org/ 2>/dev/null || true
gem sources --remove https://gems.ruby-china.com/ 2>/dev/null || true
gem sources --add https://mirrors.tuna.tsinghua.edu.cn/rubygems/ --remove https://rubygems.org/

echo "📋 Updated gem sources:"
gem sources -l

# Also configure bundler to use the Tsinghua mirror
echo "🔧 Configuring Bundler to use Tsinghua mirror..."
bundle config mirror.https://rubygems.org https://mirrors.tuna.tsinghua.edu.cn/rubygems

echo "✅ RubyGems and Bundler configured to use Tsinghua University mirror!"

# Check if Bundler is installed
if ! command -v bundle &> /dev/null; then
    echo "📦 Installing Bundler..."
    echo "🌐 Bundler will be downloaded from: https://mirrors.tuna.tsinghua.edu.cn/rubygems/gems/bundler"
    echo "📋 Running: gem install bundler --verbose"
    gem install bundler --verbose
    
    # Verify bundler installation
    if ! command -v bundle &> /dev/null; then
        echo "❌ Bundler installation failed. Please check your Ruby installation."
        exit 1
    fi
    echo "✅ Bundler installed successfully!"
fi

echo "📋 Bundler version: $(bundle --version)"

# Install Jekyll if not present
if ! command -v jekyll &> /dev/null; then
    echo "📦 Installing Jekyll..."
    echo "🌐 Jekyll will be downloaded from: https://mirrors.tuna.tsinghua.edu.cn/rubygems/gems/jekyll"
    echo "📋 Running: gem install jekyll --verbose"
    gem install jekyll --verbose
    echo "✅ Jekyll installed successfully!"
fi

# Install dependencies
echo "📦 Installing Jekyll dependencies..."
if [ -f "Gemfile" ]; then
    echo "📋 Found Gemfile, installing dependencies with verbose output..."
    echo "🌐 Gems will be downloaded from: https://mirrors.tuna.tsinghua.edu.cn/rubygems/"
    echo "📋 Running: bundle install --verbose"
    bundle install --verbose
    echo "✅ All dependencies installed successfully!"
else
    echo "❌ Gemfile not found. Please ensure you're in the correct directory."
    exit 1
fi

# Add plugins if they're not already in Gemfile
echo "🔧 Ensuring required plugins are installed..."
if ! grep -q "jekyll-katex" Gemfile; then
    echo "📦 Adding jekyll-katex plugin..."
    echo "🌐 jekyll-katex will be downloaded from: https://mirrors.tuna.tsinghua.edu.cn/rubygems/gems/jekyll-katex"
    echo "📋 Running: bundle add jekyll-katex --verbose"
    bundle add jekyll-katex --verbose
    echo "✅ jekyll-katex added successfully!"
fi

if ! grep -q "jekyll-include-cache" Gemfile; then
    echo "📦 Adding jekyll-include-cache plugin..."
    echo "🌐 jekyll-include-cache will be downloaded from: https://mirrors.tuna.tsinghua.edu.cn/rubygems/gems/jekyll-include-cache"
    echo "📋 Running: bundle add jekyll-include-cache --verbose"
    bundle add jekyll-include-cache --verbose
    echo "✅ jekyll-include-cache added successfully!"
fi

# Start Jekyll server
echo ""
echo "=== 🌐 STARTING JEKYLL DEVELOPMENT SERVER ==="
echo "📍 Your site will be available at: http://localhost:4000"
echo "📍 With baseurl, your site will be at: http://localhost:4000/blogs"
echo "📍 Live reload is enabled - changes will auto-refresh the browser"
echo ""
echo "🔧 Server configuration:"
echo "   - Host: 0.0.0.0 (accessible from other devices on network)"
echo "   - Port: 4000"
echo "   - Live reload: enabled"
echo "   - Incremental build: enabled"
echo ""
echo "📋 Running: bundle exec jekyll serve --livereload --host=0.0.0.0 --port=4000 --incremental --verbose"
echo ""
echo "Press Ctrl+C to stop the server"
echo "================================================"
echo ""

# Run Jekyll with live reload and verbose output
bundle exec jekyll serve --livereload --host=0.0.0.0 --port=4000 --incremental --verbose

### Development Notes:
### 
### Image fixes:
### Use: <img src="{{ site.baseurl }}/assets/imgs/..." width="50%" height="50%" alt="..." />
### 
### KaTeX fixes (for math in $...$):
### - Replace "_" with "\_", especially "}_{" with "}\_{"
### - Replace "|" with "\mid"
### 
### Useful Jekyll commands:
### - bundle exec jekyll build    # Build the site
### - bundle exec jekyll clean    # Clean build files
### - bundle exec jekyll doctor   # Check for issues