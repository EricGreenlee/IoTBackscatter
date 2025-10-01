#!/bin/bash

# Backscatter Experiment Startup Script
# This script starts the HackRF transmitter and USRP N210 receiver with automatic cleanup

echo "Starting FFT experiment..."

# Create a temporary file to track the HackRF process
HACKRF_PID_FILE="/tmp/hackrf_transmitter.pid"

# Start HackRF transmitter in background and save its PID
echo "Starting HackRF transmitter..."
hackrf_transfer -f 914850000 -x 47 -c 127 -a 1 &
# hackrf_transfer -f 914840800 -x 47 -c 127 -a 1 &
# hackrf_transfer -f 914865800 -x 47 -c 127 -a 1 &
HACKRF_PID=$!
echo $HACKRF_PID > $HACKRF_PID_FILE
echo "HackRF transmitter started with PID: $HACKRF_PID"

# Wait a moment for the transmitter to initialize
sleep 2

# Start USRP N210 receiver in foreground (blocks until complete)
echo "Starting USRP N210 receiver..."
python3 USRP_N210_RX_samples.py --freq 915e6 --rate 2e5 --gain 38 --num_samps 1000000 --plot
# python3 USRP_N210_RX_samples.py --freq 915e6 --rate 1e6 --gain 50 --num_samps 18000000 --plot

# When receiver finishes, stop the HackRF transmitter
if [ -f $HACKRF_PID_FILE ]; then
    HACKRF_PID=$(cat $HACKRF_PID_FILE)
    echo "Stopping HackRF transmitter (PID: $HACKRF_PID)..."
    kill $HACKRF_PID 2>/dev/null
    rm -f $HACKRF_PID_FILE
    echo "HackRF transmitter stopped."
else
    echo "Warning: Could not find HackRF PID file"
fi

echo "Experiment complete!"