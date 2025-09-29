import subprocess
import time

def check_job_completed(job_id):
    """
    Returns:
        True  -> job completed successfully
        False -> job failed or cancelled
        None  -> job still running/pending
    """
    # First, check squeue (running or pending jobs)
    result = subprocess.run(
        ["squeue", "-j", str(job_id), "-h", "-o", "%T"],
        capture_output=True,
        text=True
    )
    state = result.stdout.strip()
    if state == "":
        # Not in squeue → maybe finished, check sacct
        result = subprocess.run(
            ["sacct", "-j", str(job_id), "--format=State", "-n", "-P"],
            capture_output=True,
            text=True
        )
        state = result.stdout.strip()
        if state == "":
            # sacct didn't return anything → treat as failed
            return False
        elif "COMPLETED" in state:
            return True
        elif any(x in state for x in ["FAILED", "CANCELLED", "TIMEOUT"]):
            return False
        else:
            # Other states like RUNNING/STARTING → treat as running
            return None
    elif state in ["COMPLETED"]:
        return True
    elif state in ["FAILED", "CANCELLED", "TIMEOUT"]:
        return False
    else:
        return None  # still running

def monitor_jobs(jobs, retry=True):
    """
    jobs: list of dicts, each with keys 'job_id' and 'command'
    retry: if True, will resubmit failed jobs once
    """
    pending_jobs = jobs.copy()
    retries_left = {job["job_id"]: 1 for job in jobs} if retry else {}

    while pending_jobs:
        for job in pending_jobs[:]:
            job_id = job["job_id"]
            status = check_job_completed(job_id)

            if status is True:
                print(f"Job {job_id} completed successfully.")
                pending_jobs.remove(job)
            elif status is False:
                print(f"Job {job_id} failed.")
                if retry and retries_left.get(job_id, 0) > 0:
                    print(f"Retrying job {job_id}...")
                    new_job_id = resubmit_job(job["command"])  # define resubmit_job()
                    job["job_id"] = new_job_id
                    retries_left[new_job_id] = 0
                    pending_jobs.append(job)
                pending_jobs.remove(job)
            # else: still running, do nothing

        time.sleep(30)  # check every 30 seconds

def resubmit_job(command):
    """Resubmit a command via sbatch, return new job_id"""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    output = result.stdout.strip()
    # parse sbatch output like "Submitted batch job 5596282"
    job_id = int(output.split()[-1])
    print(f"Resubmitted job {job_id}")
    return job_id
