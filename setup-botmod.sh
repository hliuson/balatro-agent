#!/bin/bash

# Script to symlink the botmod folder to the appropriate mod directory

# Check if we're in Docker container
if [ -d "/root/.config/love/Mods" ]; then
    MOD_DIR="/root/.config/love/Mods"
elif [ -d "$HOME/.local/share/love/Mods" ]; then
    MOD_DIR="$HOME/.local/share/love/Mods"
else
    echo "Error: Could not find Love2D mods directory"
    echo "Expected locations:"
    echo "  - /root/.config/love/Mods (Docker)"
    echo "  - $HOME/.local/share/love/Mods (Local)"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOTMOD_DIR="$SCRIPT_DIR/botmod"

# Check if botmod directory exists
if [ ! -d "$BOTMOD_DIR" ]; then
    echo "Error: botmod directory not found at $BOTMOD_DIR"
    exit 1
fi

# Create symlink
SYMLINK_TARGET="$MOD_DIR/botmod"

# Remove existing symlink or directory if it exists
if [ -L "$SYMLINK_TARGET" ] || [ -d "$SYMLINK_TARGET" ]; then
    echo "Removing existing botmod at $SYMLINK_TARGET"
    rm -rf "$SYMLINK_TARGET"
fi

# Create the symlink
echo "Creating symlink: $SYMLINK_TARGET -> $BOTMOD_DIR"
ln -s "$BOTMOD_DIR" "$SYMLINK_TARGET"

# Verify the symlink was created
if [ -L "$SYMLINK_TARGET" ]; then
    echo "Successfully created symlink!"
    echo "botmod is now available at: $SYMLINK_TARGET"
else
    echo "Error: Failed to create symlink"
    exit 1
fi