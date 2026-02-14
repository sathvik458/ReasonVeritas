import json
import os
import sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "experiment_log.json")

def update_log(step_name, log_data):

    os.makedirs(LOG_DIR, exist_ok=True)

    # Load existing log safely
    if os.path.exists(LOG_FILE) and os.path.getsize(LOG_FILE) > 0:
        try:
            with open(LOG_FILE, "r") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = {}
    else:
        data = {}

    # Add universal metadata
    log_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_data["python_version"] = sys.version.split()[0]

    data[step_name] = log_data

    with open(LOG_FILE, "w") as f:
        json.dump(data, f, indent=4)

    print(f"{step_name} log saved successfully.")
