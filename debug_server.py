import debugpy
import subprocess

HOST = "0.0.0.0"
PORT = 42019

print(f"Starting debugpy server on {HOST}:{PORT}...")
debugpy.listen((HOST, PORT))

while True:
    print("\033[1;33m Waiting for debugger to attach...\033[0m")
    debugpy.wait_for_client()
    print("Debugger attached. Running script...")
    
    # Replace 'target_script.py' with your script
    subprocess.run(["./experiment-emb.sh", "configs/itl-transe.sh", "--train", "0", "--debug"])
    print("Script execution finished.")

