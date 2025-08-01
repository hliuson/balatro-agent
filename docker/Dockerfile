FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    git \
    xvfb \
    curl \
    ca-certificates \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    fuse \
    tmux \
    imagemagick \
    jq \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:/usr/local/cuda-12.9/bin:$PATH"

# Create directories
RUN mkdir -p /app/balatro \
    && mkdir -p /root/.config/love/Mods \
    && mkdir -p /root/.local/share/love

# Download Love2D AppImage
WORKDIR /tmp
RUN wget https://github.com/love2d/love/releases/download/11.5/love-11.5-x86_64.AppImage \
    && chmod +x love-11.5-x86_64.AppImage \
    && ./love-11.5-x86_64.AppImage --appimage-extract \
    && cp -r squashfs-root/* /app/balatro/ \
    && rm -rf /tmp/*

# Download and install lovely-injector for Linux
WORKDIR /tmp
RUN wget https://github.com/ethangreen-dev/lovely-injector/releases/download/v0.8.0/lovely-x86_64-unknown-linux-gnu.tar.gz \
    && tar -xzf lovely-x86_64-unknown-linux-gnu.tar.gz \
    && cp liblovely.so /app/balatro/ \
    && rm -rf /tmp/*

# Copy Balatro game file
COPY balatro.love /app/balatro/bin/Balatro.love

# Install Steamodded
WORKDIR /root/.config/love/Mods
RUN git clone https://github.com/Steamodded/smods.git

# Set up environment
ENV DISPLAY=:99
# Note: LD_PRELOAD is set in entrypoint script to avoid build-time issues

# Expose SSH port
EXPOSE 22

# Copy startup script
COPY startup.sh /usr/local/bin/startup.sh
RUN chmod +x /usr/local/bin/startup.sh

# Set entrypoint
ENTRYPOINT ["/bin/bash", "/usr/local/bin/startup.sh"]

# Default command - keep container alive
WORKDIR /app/balatro-agent
CMD ["bash"]