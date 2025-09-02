#!/bin/bash

# Backscatter Experiment Startup Script
# This script starts the HackRF transmitter and Python receiver in separate terminals

echo "Starting backscatter experiment..."

# Start HackRF transmitter in a new terminal
gnome-terminal --title="HackRF Transmitter" -- bash -c "hackrf_transfer -f 915000000 -x 47 -c 127; exec bash"

# Wait a moment for the first terminal to start
sleep 2

# Start Python main.py in another terminal
gnome-terminal --title="Python Receiver" -- bash -c "cd python && python src/main.py; exec bash"

echo "Experiment started! Check the two terminal windows."
echo "HackRF Transmitter: hackrf_transfer -f 915000000 -x 47 -c 127"
echo "Python Receiver: python/src/main.py"