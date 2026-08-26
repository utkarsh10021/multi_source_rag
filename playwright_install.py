import subprocess

subprocess.run(
    ["python", "-m", "playwright", "install", "chromium"],
    check=True
)