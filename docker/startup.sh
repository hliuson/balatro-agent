#!/bin/bash

# Startup script for Balatro Agent container
# Installs Claude Code and sets up the repo at runtime

set -e

echo "Starting Balatro Agent container setup..."

# Install Claude Code if not already installed
#if ! which claude > /dev/null 2>&1; then
#    echo "Installing Claude Code..."
#    curl -fsSL https://claude.ai/install.sh | bash
#    export PATH="/root/.local/bin:$PATH"
#fi

# Clone the repo if it doesn't exist
if [ ! -d "/root/balatro-agent" ]; then
    echo "Cloning balatro-agent repository..."
    cd /root
    git clone https://github.com/hliuson/balatro-agent.git
fi

# Set up the botmod
echo "Setting up botmod..."
cd /root/balatro-agent
if [ -f "setup-botmod.sh" ]; then
    chmod +x setup-botmod.sh
    ./setup-botmod.sh
fi

# Set up agent-rl environment
#echo "Setting up agent-rl environment..."
#cd /root/balatro-agent/agent-rl
#if [ -f "pyproject.toml" ]; then
#    uv sync
#fi

echo "Setup complete!"

# Execute the command passed to the container, or default to bash
exec "$@"