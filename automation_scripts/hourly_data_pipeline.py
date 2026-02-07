import subprocess
import sys
import os

def run_step(script_name):
    # Calculate path: Go up from 'automation_scripts' then into 'scripts'
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target = os.path.join(base_path, "scripts", script_name)

    print(f"\n--- Running: {script_name} ---")
    result = subprocess.run([sys.executable, target], capture_output=False)

    if result.returncode != 0:
        print(f"FAILED: {script_name} exited with error.")
        sys.exit(1)

if __name__ == "__main__":
    run_step("data_extraction.py")
    run_step("feature_engineering.py")
    print("\n[SUCCESS] Data pipeline finished.")