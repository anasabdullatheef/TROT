#!/bin/bash
# Setup cron job for data collector
# Run this script once to install the cron job

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_PATH="$SCRIPT_DIR/venv/bin/python"
COLLECTOR_PATH="$SCRIPT_DIR/collector.py"
LOG_PATH="$SCRIPT_DIR/logs/collector.log"

# Create logs directory
mkdir -p "$SCRIPT_DIR/logs"

# Create cron entry (runs every minute)
CRON_JOB="* * * * * cd $SCRIPT_DIR && $PYTHON_PATH $COLLECTOR_PATH >> $LOG_PATH 2>&1"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "collector.py"; then
    echo "Collector cron job already exists"
else
    # Add to crontab
    (crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -
    echo "✅ Collector cron job installed"
    echo "   Running every minute"
    echo "   Log file: $LOG_PATH"
fi

echo ""
echo "To check status:"
echo "  crontab -l | grep collector"
echo ""
echo "To view logs:"
echo "  tail -f $LOG_PATH"
echo ""
echo "To remove:"
echo "  crontab -l | grep -v collector | crontab -"
