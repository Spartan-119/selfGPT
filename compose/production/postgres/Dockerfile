FROM docker.io/postgres:15

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    postgresql-server-dev-15 \
    && rm -rf /var/lib/apt/lists/*

# Clone and install pgvector
RUN cd /tmp && \
    git clone --branch v0.6.0 https://github.com/pgvector/pgvector.git && \
    cd pgvector && \
    make && \
    make install

# Copy and set permissions for maintenance scripts
COPY ./compose/production/postgres/maintenance /usr/local/bin/maintenance
RUN chmod +x /usr/local/bin/maintenance/* && \
    mv /usr/local/bin/maintenance/* /usr/local/bin && \
    rmdir /usr/local/bin/maintenance

# Clean up
RUN apt-get purge -y --auto-remove build-essential git postgresql-server-dev-15
