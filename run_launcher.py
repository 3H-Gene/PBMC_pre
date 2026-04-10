import subprocess, sys, os

env = os.environ.copy()
env["PYTHONUTF8"] = "1"
env["PYTHONIOENCODING"] = "utf-8"

log_path = r"d:\Programs\workspace\PBMC\outputs\pipeline_run.log"
os.makedirs(os.path.dirname(log_path), exist_ok=True)

with open(log_path, "w", encoding="utf-8") as log_f:
    proc = subprocess.Popen(
        [sys.executable, "run_pipeline.py"],
        cwd=r"d:\Programs\workspace\PBMC",
        stdout=log_f,
        stderr=log_f,
        env=env,
    )
    proc.wait()

print(f"Pipeline finished. exit={proc.returncode}. Log: {log_path}")
