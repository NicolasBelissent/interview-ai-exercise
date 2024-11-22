
markdown
Copy code
# README

## Overview

This project has been adjusted to simplify its usage by running everything directly with `docker-compose` on the command line. The modifications include changes to the Dockerfile and some additional dependencies.

## Changes Made

### 1. Dockerfile Modifications
- Added several packages, including a Redis server.
- Redis is used to store documents locally for lexical search functionality (to compensate for features not supported by Chroma).

### 2. Usage Instructions
To start the API:

```bash
docker-compose up --build