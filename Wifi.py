# =============================================================================
# Wifi.py
# WiFi Connection Helper - runs on ESP32S3 Sense
#
# Author: Lisa Chen
# Course: ENMGT 5400 - Project 4
#
# Description:
#   Provides a simple function to connect the ESP32 to a WiFi network.
#   Used by the socket server to enable image streaming to laptop.
#
# Note: Use a phone hotspot rather than Cornell WiFi (security restrictions)
# =============================================================================

import network
import time

# =============================================================================
# CONFIGURATION - update with your hotspot credentials
# =============================================================================
WIFI_SSID = "iPhone"        # Your phone hotspot name
WIFI_PASSWORD = "your_password_here"  # Your hotspot password

def connect(ssid=WIFI_SSID, password=WIFI_PASSWORD, timeout=15):
    """
    Connect to WiFi and return IP address.
    
    Args:
        ssid: WiFi network name
        password: WiFi password
        timeout: seconds to wait before giving up
    
    Returns:
        IP address string if connected, None if failed
    """
    wlan = network.WLAN(network.STA_IF)
    
    # Reset WiFi module to clear any previous state
    wlan.active(False)
    time.sleep(0.5)
    wlan.active(True)
    time.sleep(0.5)
    
    if wlan.isconnected():
        print("Already connected:", wlan.ifconfig()[0])
        return wlan.ifconfig()[0]
    
    print(f"Connecting to {ssid}...")
    wlan.connect(ssid, password)
    
    # Wait for connection with timeout
    start = time.time()
    while not wlan.isconnected():
        elapsed = time.time() - start
        if elapsed > timeout:
            print(f"Connection failed after {timeout}s")
            print("Check: hotspot is on, credentials are correct, ESP is in range")
            return None
        print(f"  Waiting... {elapsed:.0f}s")
        time.sleep(1)
    
    ip = wlan.ifconfig()[0]
    print(f"Connected! IP: {ip}")
    return ip


def disconnect():
    """Disconnect from WiFi."""
    wlan = network.WLAN(network.STA_IF)
    wlan.disconnect()
    wlan.active(False)
    print("WiFi disconnected")


# Auto-connect when imported as main script
if __name__ == '__main__':
    ip = connect()
    if ip:
        print(f"\nESP32 IP address: {ip}")
        print("Use this IP in your laptop socket client")
    else:
        print("Failed to connect to WiFi")
