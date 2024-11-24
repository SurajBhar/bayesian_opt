# python /home/sur06423/wacv_paper/wacv_paper/ray_job_submission_jarvis/submit_ray_job.py

import ray
from ray import job_submission
import yaml
import os

# Load job configuration from job_config.yaml
def load_job_config(config_path="/home/sur06423/wacv_paper/wacv_paper/ray_job_submission_jarvis/job_config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def submit_ray_job():
    try:
        # Load configuration
        config = load_job_config()

        # Extract configurations
        ray_address = config["ray_address"]
        working_dir = config["working_dir"]
        entrypoint = config["entrypoint"]

        # Define the Ray Jobs API client
        client = job_submission.JobSubmissionClient(ray_address)

        # Submit the job
        submission_id = client.submit_job(
            entrypoint=entrypoint,
            runtime_env={
                "working_dir": working_dir,
                "excludes": [
                    "**/*.pth",   # Exclude PyTorch model files
                    "**/*.pt",    # Exclude additional model checkpoint files
                    "other_tutorials/**",  # Exclude everything in other_tutorials
                    "ray_job_submission/**",  # Exclude everything in this folder
                    "outputs/**",  # Exclude everything in hydra outputs
                    "logs_BO/**", # Exclude all logs
                    "data/**",             # Exclude everything in data
                    "**/.git/**", # Exclude Git directory
                ],
            },
        )

        print(f"Submitted job with ID: {submission_id}")

        # monitor job status and logs
        status = client.get_job_status(submission_id)
        print(f"Job status: {status}")

        # Fetch job logs
        logs = client.get_job_logs(submission_id)
        print("Job logs:", logs)

    except Exception as e:
        print(f"Error during job submission: {e}")

if __name__ == "__main__":
    submit_ray_job()
