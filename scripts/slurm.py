#!/usr/bin/env python3
import os
import argparse
import subprocess
import time
from pathlib import Path

def split_input_file(input_file, n_splits, out_dir):
    """Split input file into N parts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total_lines = len(lines)
    chunk_size = (total_lines + n_splits - 1) // n_splits  # ceiling division
    split_files = []

    for i in range(n_splits):
        chunk = lines[i*chunk_size:(i+1)*chunk_size]
        split_path = out_dir / f"split_{i}.txt"
        with open(split_path, "w", encoding="utf-8") as f_out:
            f_out.writelines(chunk)
        split_files.append(split_path)
    return split_files, total_lines

def submit_slurm_job(split_file, job_script, slurm_args, log_dir):
    """Submit a Slurm job for one split."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    job_name = f"job_{split_file.stem}"
    out_log = log_dir / f"{split_file.stem}.out"
    err_log = log_dir / f"{split_file.stem}.err"

    sbatch_cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--output={out_log}",
        f"--error={err_log}",
    ] + slurm_args + [job_script, str(split_file)]
    
    result = subprocess.run(sbatch_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to submit job: {result.stderr}")
    
    # Parse job ID from sbatch output
    job_id = int(result.stdout.strip().split()[-1])
    return job_id, out_log, err_log

def check_job_status(job_id):
    """Return True if job completed successfully."""
    sacct_cmd = ["sacct", "-j", str(job_id), "--format=State", "--noheader"]
    result = subprocess.run(sacct_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False
    state = result.stdout.strip().split()[0]
    return state in ("COMPLETED",)

def concatenate_outputs(output_files, final_output):
    """Concatenate all job outputs into a single file."""
    with open(final_output, "w", encoding="utf-8") as f_out:
        for fpath in output_files:
            with open(fpath, "r", encoding="utf-8") as f_in:
                f_out.writelines(f_in)

def monitor_jobs(job_infos, max_retries=1, poll_interval=30):
    """Monitor jobs, restart if failed (once)."""
    retries = {job_id: 0 for job_id, _, _ in job_infos}
    remaining_jobs = set(job_id for job_id, _, _ in job_infos)

    while remaining_jobs:
        for job_id, out_log, err_log in job_infos:
            if job_id not in remaining_jobs:
                continue

            if check_job_status(job_id):
                remaining_jobs.remove(job_id)
            else:
                # if failed and retry allowed
                if retries[job_id] < max_retries:
                    print(f"Job {job_id} failed. Retrying...")
                    # re-submit logic here (depends on your job script & args)
                    retries[job_id] += 1
                else:
                    raise RuntimeError(f"Job {job_id} failed after {max_retries} retries.")

        if remaining_jobs:
            print(f"Waiting for {len(remaining_jobs)} jobs...")
            time.sleep(poll_interval)

def validate_output(final_output, expected_lines):
    """Check that output has expected number of lines."""
    with open(final_output, "r", encoding="utf-8") as f:
        n_lines = sum(1 for _ in f)
    if n_lines != expected_lines:
        raise ValueError(f"Output line count {n_lines} != expected {expected_lines}")
    print(f"Validation passed: {n_lines} lines.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--job_script", required=True, help="Slurm job script to run on each split")
    parser.add_argument("--n_splits", type=int, default=4)
    parser.add_argument("--slurm_args", nargs="*", default=[], help="Extra sbatch args, e.g. ['--partition=short']")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--split_dir", default="splits")
    parser.add_argument("--final_output", default="output.txt")
    args = parser.parse_args()

    splits, total_lines = split_input_file(args.input_file, args.n_splits, args.split_dir)
    
    job_infos = []
    for split_file in splits:
        job_id, out_log, err_log = submit_slurm_job(split_file, args.job_script, args.slurm_args, args.log_dir)
        print(f"Submitted {split_file} as job {job_id}")
        job_infos.append((job_id, out_log, err_log))
    
    monitor_jobs(job_infos)
    
    # Assuming each job writes output to a file named same as split with .out extension
    output_files = [Path(args.log_dir)/f"{split_file.stem}.out" for split_file in splits]
    concatenate_outputs(output_files, args.final_output)
    validate_output(args.final_output, total_lines)

if __name__ == "__main__":
    main()
