import os
import multiprocessing
import psutil

def check_cpu_resources():
    # 1. Total Logical Cores (includes Hyper-threading)
    logical = os.cpu_count() 
    
    # 2. Total Physical Cores (Actual hardware chips)
    physical = psutil.cpu_count(logical=False)
    
    # 3. Available to CURRENT Process (CRITICAL for DataHub/Docker)
    # This shows what the OS scheduler actually allows you to use right now.
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        # Fallback for systems without sched_getaffinity (like macOS/Windows)
        available = logical

    print(f"--- CPU Resource Report ---")
    print(f"Total Logical Cores:   {logical}")
    print(f"Total Physical Cores:  {physical}")
    print(f"Available to Process:  {available}  <-- USE THIS FOR PARALLEL POOLS")
    print(f"---------------------------")
    
    return available

if __name__ == "__main__":
    usable_cores = check_cpu_resources()