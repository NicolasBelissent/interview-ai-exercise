# Stage 1: Base image with curl
FROM python:3.11.4-slim-bookworm as curl-stage

# Install curl and Redis server; remove apt cache to reduce image size
RUN apt-get -y update && apt-get -y install curl redis-server && rm -rf /var/lib/apt/lists/*

# Stage 2: Poetry for dependency management
FROM curl-stage as poetry-requirements-stage

WORKDIR /tmp

ENV HOME /root
ENV PATH=${PATH}:$HOME/.local/bin

# Install poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=1.5.1 python3 -

# Export requirements.txt
COPY ./pyproject.toml ./poetry.lock* /tmp/
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes --no-interaction --no-cache --only=main

# Stage 3: Application image
FROM curl-stage

WORKDIR /code

ENV \
    # Prevent Python from buffering stdout and stderr
    PYTHONUNBUFFERED=1 \
    # Prevent Pip from timing out when installing heavy dependencies
    PIP_DEFAULT_TIMEOUT=600 \
    # Prevent Pip from creating a cache directory to reduce image size
    PIP_NO_CACHE_DIR=1

# Install Python dependencies from exported requirements.txt
COPY --from=poetry-requirements-stage /tmp/requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Install langchain package
RUN pip install langchain


# Install Redis Python library
RUN pip install redis

# Copy application files
COPY src ./src

# Copy Redis configuration (optional if you want to customize Redis settings)
# COPY redis.conf /etc/redis/redis.conf

# Start both Redis server and FastAPI using a script
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Expose Redis and FastAPI ports
EXPOSE 6379 80

# Healthcheck
HEALTHCHECK --interval=100s --timeout=1s --retries=3 CMD curl --fail http://localhost/health || exit 1

# Command to run the start script
CMD ["/start.sh"]
