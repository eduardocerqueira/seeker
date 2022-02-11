#date: 2022-02-11T16:41:20Z
#url: https://api.github.com/gists/53c5de1d18ca19a1d7708b9bddf941cc
#owner: https://api.github.com/users/uchenna-madu

import argparse
import os
import sys
import tarfile
import time
import urllib.request


def main():
    program_name = os.path.basename(sys.argv[0])
    parser = argparse.ArgumentParser(prog=program_name)
    parser.add_argument(
        "-w",
        "--set_wait_counter",
        required=False,
        default="false",
        help="set wait counter",
    )
    parser.add_argument(
        "-s",
        "--set_job_start_time",
        required=False,
        default="false",
        help="set job start time",
    )
    parser.add_argument(
        "-e",
        "--set_job_end_time",
        required=False,
        default="false",
        help="set job end time",
    )
    args = parser.parse_args()

    set_wait = eval(args.set_wait_counter.capitalize())
    set_start = eval(args.set_job_start_time.capitalize())
    set_end = eval(args.set_job_end_time.capitalize())

    publish_metrics = PublishMetrics(set_wait, set_start, set_end)

    if publish_metrics.set_wait:
        # set_wait_counter
        print("set_wait_counter = true")
        # print(f"::set-output name=wait_time::{publish_metrics().now}")
        print(f"::set-output name=wait_time::{int(time.time())}")
    elif publish_metrics.set_start:
        # set_job_start_time
        print("set_job_start_time = true")
        # print(f"::set-output name=start_time::{publish_metrics().now}")
        print(f"::set-output name=start_time::{int(time.time())}")
    elif publish_metrics.set_end:
        # set_job_end_time
        print("set_job_end_time = true")
        # print(f"::set-output name=end_time::{publish_metrics().now}")
        print(f"::set-output name=end_time::{int(time.time())}")
    else:
        print("option is not defined")

    # check/add humioctl
    publish_metrics.add_humioctl()

    # Publish Metric to Humio
    publish_metrics.push_to_humio()

    # backup log in s3
    publish_metrics.save_to_s3()



class PublishMetrics:
    def __init__(self, set_wait, set_start, set_end) -> None:
        self.set_wait = set_wait
        self.set_start = set_start
        self.set_end = set_end

        self.run_id = os.environ.get("RUN_ID")
        self.repository = os.environ.get("REPOSITORY")
        self.humio_repo = os.environ.get("HUMIO_REPO")
        self.humio_url = os.environ.get("HUMIO_URL")
        self.actor = os.environ.get("GITHUB_ACTOR")
        self.base_ref = os.environ.get("BASE_REF")
        self.ref_name = os.environ.get("REF_NAME")
        self.sha = os.environ.get("SHA")
        self.workflow = os.environ.get("WORKFLOW")
        self.job_name = os.environ.get("JOB_NAME")
        self.action_url = os.environ.get("ACTION_URL")
        self.ingest_token = os.environ.get("INGEST_TOKEN")
        self.run_duration = ""
        self.wait_duration = ""
        self.timestamp = int(time.time() * 1000)  # in milisec
        self.now = int(time.time())  # in sec
        self.runner_name = os.environ.get("RUNNER_NAME")

        if set_wait:
            self.runner_name = ""
            self.test_status = "waiting"
            self.wait_start = self.now
        elif set_start:
            self.test_status = "running"
            self.job_start = self.now
            self.wait_start = int(os.environ.get("WAIT_START", "0"))
            self.wait_duration = str(int(self.job_start) - int(self.wait_start)) + "s"
        elif set_end:
            self.test_status = os.environ.get("TEST_STATUS", "")
            self.wait_start = int(os.environ.get("WAIT_START", "0"))
            self.job_start = int(os.environ.get("JOB_START", "0"))
            self.job_end = self.now
            self.wait_duration = str(int(self.job_start) - int(self.wait_start)) + "s"
            self.run_duration = str(int(self.job_end) - int(self.job_start)) + "s"
        else:
            print("The test status is not defined")


    def push_to_humio(self):
        self.data = (
            "{ "
            + f'\\"@timestamp\\":\\"{self.timestamp}\\",\\"github.run_id\\":\\"{self.run_id}\\",\\"github.repository\\":\\"{self.repository}\\",\\"#repo\\":\\"{self.humio_repo}\\",\\"github.actor\\":\\"{self.actor}\\",\\"job.status\\":\\"{self.test_status}\\",\\"github.base_ref\\":\\"{self.base_ref}\\",\\"github.ref_name\\":\\"{self.ref_name}\\",\\"github.sha\\":\\"{self.sha}\\",\\"github.workflow\\":\\"{self.workflow}\\",\\"github.job\\":\\"{self.job_name}\\",\\"github.job.run.duration\\":\\"{self.run_duration}\\",\\"github.job.wait.duration\\":\\"{self.wait_duration}\\",\\"github.runner\\":\\"{self.runner_name}\\",\\"github.action.url\\":\\"{self.action_url}\\"'  # noqa: E501
            + " }"
        )
        print(f"::set-output name=data::{self.data}")

        print("Publishing metrics data to humio")
        push_2humio = f"echo {self.data} | /opt/humioctl/humioctl -a {self.humio_url} ingest {self.humio_repo} -t {self.ingest_token} -q -p json-for-action"  # noqa: E501
        rc0 = os.system(push_2humio)
        if rc0:
            raise Exception(f"Failed to successfully push log data to humio. rc0={rc0}")

        # Printing out metrics data in log
        os.system(f"echo {self.data}")


    def add_humioctl(self):
        folder = "/opt/humioctl"
        file = f"{folder}/humioctl"
        isFile = os.path.isfile(file)
        isdir = os.path.isdir(folder)
        humioctl_pkg = "humioctl_0.28.11_Linux_64-bit.tar.gz"
        if isFile:
            print("humioctl exist")
        else:
            if isdir:
                print("directory exit")
            else:
                os.mkdir(folder)
            print(file)
            print("downloading humioctl")
            url = f"https://github.com/humio/cli/releases/download/v0.28.11/{humioctl_pkg}"
            urllib.request.urlretrieve(
                url, f"{folder}/{humioctl_pkg}"
            )

            # extrating file
            tf = tarfile.open(f"{folder}/{humioctl_pkg}", mode="r")
            tf.extract("humioctl", path=folder)

            # remove tar file
            os.remove(f"{folder}/{humioctl_pkg}")


    def save_to_s3(self):
        runner_name = os.environ.get("RUNNER_NAME")
        date_prefix = time.strftime("%Y%m")
        filename = f"{date_prefix}_{runner_name}.txt"
        temp_file = f"/tmp/{filename}.tmp"
        s3_location = os.environ.get("S3_LOCATION")

        if self.set_end:
            print("Saving data to a file")
            data = self.data
            rc = os.system(f"echo {data} >> {temp_file}")
            if rc:
                raise Exception(
                    f"Failed to successfully save log data to a temp_file. rc={rc}"
                )

            print("Saving metrics to s3")
            test_cmd = f"aws s3 --region us-west-2 ls {s3_location}/ 2>/dev/null | grep -q {filename}"
            test_file = os.system(test_cmd)

            if test_file:
                print("file doesn't exit in s3, adding new file")
                rc1 = os.system(f"cp {temp_file} {filename}")

                cp_2s3 = f"aws s3 --region us-west-2 cp {filename} {s3_location}/{filename}"
                rc2 = os.system(cp_2s3)

                os.system(f"aws s3 --region us-west-2 ls {s3_location}/")

                if rc1 or rc2:
                    raise Exception(
                        f"Failed to successfully copy log file to s3 location, rc1={rc1}, rc2={rc2}"
                    )
            else:
                print("file exit in s3, updating it")
                cp_froms3 = (
                    f"aws s3 --region us-west-2 cp {s3_location}/{filename} {filename}"
                )
                rc1 = os.system(cp_froms3)

                rc2 = os.system(f"cat {temp_file} >> {filename}")

                cp_2s3 = f"aws s3 --region us-west-2 cp {filename} {s3_location}/{filename}"
                rc3 = os.system(cp_2s3)

                os.system(f"aws s3 --region us-west-2 ls {s3_location}/")

                if rc1 or rc2 or rc3:
                    raise Exception(
                        f"Failed to successfully copy log file to s3 location. rc1={rc1}, rc2={rc2}, rc3{rc3}"
                    )

            # Clean up files
            print("Cleaning up temp files")
            os.remove(filename)
            os.remove(temp_file)


if __name__ == "__main__":
    main()
    