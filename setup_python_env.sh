#!/bin/bash

# Check for Homebrew installation
if ! command -v brew &> /dev/null; then
    echo "Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "Homebrew is already installed."
fi

# Install dependencies for building Python
echo "Installing dependencies for Python build..."
brew update
brew install zlib bzip2 openssl readline sqlite3 xz

# Install pyenv
if ! command -v pyenv &> /dev/null; then
    echo "pyenv not found. Installing pyenv..."
    brew install pyenv
else
    echo "pyenv is already installed."
fi

# Set up pyenv in the shell
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
if command -v pyenv &> /dev/null; then
    eval "$(pyenv init --path)"
    eval "$(pyenv init -)"
fi

# Install Python 3.11.4 using pyenv
echo "Installing Python 3.11.4..."
pyenv install 3.11.4

# Set global Python version to 3.11.4
echo "Setting Python 3.11.4 as the global version..."
pyenv global 3.11.4

# Verify installation and architecture
echo "Verifying Python installation and architecture..."
python3 --version
python3 -c "import platform; print(platform.architecture())"

# Set up a virtual environment (optional)
echo "Creating a virtual environment named 'env'..."
python3 -m venv env
source env/bin/activate
echo "Virtual environment 'env' activated."

# Installation complete
echo "Python 3.11.4 environment for ARM64 is set up!"

