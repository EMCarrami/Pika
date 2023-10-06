import argparse
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", type=int, help="gpu id")

args = parser.parse_args()

print("\033c", end="")  # Clear terminal screen
print("GPU Memory Usage Over Time")
while True:
    # Run nvidia-smi to get GPU memory usage
    smi_output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            f"--id={args.gpu_id}",
        ]
    )
    memory_used = int(smi_output.decode().strip())
    bar = "█" * (memory_used // 100)
    print(f"{memory_used} MiB | {bar}")

    time.sleep(1)
