#!/bin/bash

# Ensure directories exist
mkdir -p /app/src
mkdir -p /root/.wine/drive_c/mt5

# Initialize Wine prefix if needed (silent)
if [ ! -d "$WINEPREFIX" ]; then
    echo "Initializing Wine prefix..."
    wineboot --init > /dev/null 2>&1
fi

# Start Supervisor
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
