#!/usr/bin/env python3
import argparse
from pathlib import Path
import subprocess
import time
import shutil

def split_input_file(input_file, n_splits, split_dir):
    """Split input file into N parts."""
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total_lines = len(lines)
    chunk_size = (total_lines + n_splits - 1) // n_splits

    split_files = []
    for i in range(n_splits):
        chunk = lines[i*chunk_size:(i+1)*chunk_size]
        split_path = split_dir / f"split_{i}.jsonl"
        with open(split_path, "w", encoding="utf-8") as f_out:
            f_out.writelines(chunk)
        split_files.append(split_path)

    return split_files, total_lines

def submit_job(command, slurm_args, log_dir, job_name):
    """Create a temporary Slurm script and submit it."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    out_log = log_dir / f"{job_name}.out"
    err_log = log_dir / f"{job_name}.err"

    slurm_script = log_dir / f"{job_name}.slurm"
    with open(slurm_script, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n")
        for arg in slurm_args:
            f.write(f"#SBATCH {arg}\n")
        f.write(f"#SBATCH --job-name={job_name}\n")
        f.write(f"#SBATCH --output={out_log}\n")
        f.write(f"#SBATCH --error={err_log}\n\n")
        f.write(f"{command}\n")

    result = subprocess.run(["sbatch", str(slurm_script)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to submit job: {result.stderr}")

    job_id = int(result.stdout.strip().split()[-1])
    return job_id, out_log, err_log, slurm_script

def check_job_completed(job_id):
    """
    Check SLURM job status.
    Returns:
        True if COMPLETED,
        False if FAILED or CANCELLED,
        None if still RUNNING/PENDING
    """
    try:
        result = subprocess.run(
            ["sacct", "-j", str(job_id), "--format=State", "--noheader"],
            capture_output=True,
            text=True,
        )
        state_line = result.stdout.strip()
        if not state_line:
            return None
        state = state_line.split()[0]
        if state in ("COMPLETED", "COMPLETING"):
            return True
        elif state in ("FAILED", "CANCELLED", "TIMEOUT"):
            return False
        else:
            return None
    except Exception as e:
        print(f"Error checking job {job_id}: {e}")
        return None

def resubmit_job(command):
    """
    Resubmit the same SLURM command.
    Returns new job ID as int.
    """
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    # Assume sbatch returns: "Submitted batch job 123456"
    for line in result.stdout.splitlines():
        if "Submitted batch job" in line:
            return int(line.strip().split()[-1])
    raise RuntimeError(f"Failed to resubmit job: {command}")

def monitor_jobs(jobs, retry=True, poll_interval=30):
    """
    Monitor a list of (job_id, command) tuples.
    Retries failed jobs once if retry=True.
    """
    pending_jobs = jobs.copy()
    retries_left = {job[0]: 1 for job in jobs} if retry else {}

    while pending_jobs:
        for job in pending_jobs[:]:
            job_id, command = job
            status = check_job_completed(job_id)

            if status is True:
                print(f"[INFO] Job {job_id} completed successfully.")
                pending_jobs.remove(job)
            elif status is False:
                print(f"[WARN] Job {job_id} failed.")
                if retry and retries_left.get(job_id, 0) > 0:
                    print(f"[INFO] Retrying job {job_id}...")
                    new_job_id = resubmit_job(command)
                    print(f"[INFO] New job submitted: {new_job_id}")
                    pending_jobs.append((new_job_id, command))
                    retries_left[new_job_id] = 0
                pending_jobs.remove(job)
            # else: still running

        if pending_jobs:
            time.sleep(poll_interval)

def concatenate_outputs(split_files, final_output):
    """Concatenate outputs from split files."""
    with open(final_output, "w", encoding="utf-8") as fout:
        for f in split_files:
            # assuming output file for each split has same name with _out suffix
            out_file = f.with_name(f"{f.stem}_out.jsonl")
            with open(out_file, "r", encoding="utf-8") as fin:
                shutil.copyfileobj(fin, fout)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--n_splits", type=int, default=4)
    parser.add_argument("--slurm_args", nargs="*", default=[], help="Extra sbatch arguments, e.g. --partition=short")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--split_dir", default="splits")
    parser.add_argument("--python_command", required=True, help="Python command template. Use {split_input} and {split_output}")
    args = parser.parse_args()

    # Split input
    split_files, total_lines = split_input_file(args.input_file, args.n_splits, args.split_dir)

    # Submit jobs
    jobs = []
    for split_file in split_files:
        split_output = split_file.with_name(f"{split_file.stem}_out.jsonl")
        cmd = args.python_command.format(split_input=split_file, split_output=split_output)
        job_id, out_log, err_log, script = submit_job(cmd, args.slurm_args, args.log_dir, split_file.stem)
        print(f"Submitted {split_file} as job {job_id}")
        jobs.append((job_id, out_log, err_log, script))

    # Monitor
    monitor_jobs(jobs)

    # Concatenate outputs
    output_files = [f.with_name(f"{f.stem}_out.jsonl") for f in split_files]
    concatenate_outputs(output_files, args.output_file)
    print(f"All done! Output written to {args.output_file} (expected {total_lines} lines)")

if __name__ == "__main__":
    main()
