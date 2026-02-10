#!/bin/bash
set -e

# Define ADB Host
# On Docker for Mac/Windows, 'host.docker.internal' resolves to the host machine.
# On Linux with --network host, you might just access localhost, but this script assumes Mac/Windows context
ADB_HOST="host.docker.internal"
ADB_PORT="5555"

echo "Attempting to connect to ADB at $ADB_HOST:$ADB_PORT..."

# Try to connect
adb connect "$ADB_HOST:$ADB_PORT"

# Check devices
adb devices

# Execute the passed command
exec "$@"
