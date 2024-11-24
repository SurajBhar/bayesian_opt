import sys
from ray import job_submission

# Pass the Ray head node IP as a command-line argument if necessary
ray_address = "http://10.56.7.41:8265"
client = job_submission.JobSubmissionClient(ray_address)

# Cancel a specific job by ID (replace <job-id>)
job_id = sys.argv[1] if len(sys.argv) > 1 else "04000000"
client.stop_job(job_id)
print(f"Job {job_id} has been cancelled.")
