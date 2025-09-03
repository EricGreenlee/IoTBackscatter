#!/bin/bash
# Script to configure USRP N210 network without breaking internet connectivity

# Find the USB-to-Ethernet interface (typically starts with enx or eth)
USRP_INTERFACE=$(ip link show | grep -E "enx|eth" | grep -v lo | head -1 | cut -d: -f2 | tr -d ' ')

if [ -z "$USRP_INTERFACE" ]; then
    echo "No USB-to-Ethernet interface found. Please plug in the USRP N210."
    exit 1
fi

echo "Configuring interface: $USRP_INTERFACE"

# Configure static IP for USRP communication
sudo ip addr add 192.168.10.1/24 dev $USRP_INTERFACE
sudo ip link set $USRP_INTERFACE up

# Add route specifically for USRP subnet (don't change default route)
sudo ip route add 192.168.10.0/24 dev $USRP_INTERFACE

echo "USRP network configured:"
echo "  Interface: $USRP_INTERFACE"
echo "  Host IP: 192.168.10.1" 
echo "  USRP IP: 192.168.10.2"
echo "  Internet connectivity preserved on WiFi"

# Test USRP connectivity
echo "Testing USRP connectivity..."
ping -c 3 192.168.10.2