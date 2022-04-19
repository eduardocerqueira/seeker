#date: 2022-04-19T17:01:29Z
#url: https://api.github.com/gists/c2bfddac7bd8dc5702f6eec31729fb48
#owner: https://api.github.com/users/kuharan

import signal

def timeout_handler(_signal, _frame):
    global unprocess_bucket
    global unprocess_file

    logger.info("Time exceeded! Creating Unprocessed File.")

    session = boto3.Session()
    s3_client = session.client(service_name="s3")
    s3_client.put_object(
        Body="",
        Bucket=unprocess_bucket,
        Key=unprocess_file.replace(
            unprocess_file.split("/")[1], "doc_pdf/unprocessed_files"
        ),
    )


signal.signal(signal.SIGALRM, timeout_handler)


def lambda_handler(event, context):
    signal.alarm(int(context.get_remaining_time_in_millis() / 1000) - 15)
    ## rest of the code ##