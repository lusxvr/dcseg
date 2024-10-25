import subprocess

def run_command(command):
    print(f"Running command: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {command}")