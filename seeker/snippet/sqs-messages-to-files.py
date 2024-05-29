#date: 2024-05-29T16:49:36Z
#url: https://api.github.com/gists/d3f33b8a5a43b48042a20c2ee9afbd6d
#owner: https://api.github.com/users/marcusadair

import json
import os
import sys

import boto3


def receive(queue_url, delete_messages=False):
    print(f"Reading queue {queue_url}", file=sys.stderr)
    sqs_client = boto3.client('sqs')
    while True:
        resp = sqs_client.receive_message(
            QueueUrl=queue_url,
            MessageSystemAttributeNames=['All'],
            MaxNumberOfMessages=10
        )
        try:
            yield from resp['Messages']
        except KeyError:
            print("No messages received", file=sys.stderr)
            return
        entries = [
            {'Id': msg['MessageId'], 'ReceiptHandle': msg['ReceiptHandle']}
            for msg in resp['Messages']
        ]
        if entries and delete_messages:
            print(f"Deleting {len(entries)} messages...", file=sys.stderr)
            resp = sqs_client.delete_message_batch(
                QueueUrl=queue_url, Entries=entries
            )
            print(f"Deleted {len(resp['Successful'])}/{len(entries)} messages", file=sys.stderr)
            if len(resp['Successful']) != len(entries):
                raise RuntimeError(f"Not all requested deletions occurred")


def write(msg, out_dir):
    msg_id = msg['MessageId']
    filename = os.path.join(out_dir, f"{msg_id}.json")
    print(f"Writing {filename}", file=sys.stderr)
    with open(filename, 'w') as f:
        json.dump(msg, f, indent=2)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--delete', action='store_true', help="Delete messages after receiving")
    parser.add_argument('--out', default='out', help="Output directory")
    parser.add_argument('url', help="SQS queue URL")
    args = parser.parse_args()

    count = 0
    for msg in receive(args.url, delete_messages=args.delete):
        count += 1
        write(msg, args.out)

    print(f"Received {count} messages", file=sys.stderr)
    

if __name__ == '__main__':
    main()
