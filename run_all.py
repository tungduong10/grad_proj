import subprocess

# Start FastAPI
api = subprocess.Popen([
    "uvicorn", "server.server:app",
    "--host", "0.0.0.0",
    "--port", "8000"
])

# Start Streamlit
ui = subprocess.Popen([
    "streamlit", "run", "app.py"
])

try:
    api.wait()
    ui.wait()
except KeyboardInterrupt:
    api.terminate()
    ui.terminate()