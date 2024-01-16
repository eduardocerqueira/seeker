#date: 2024-01-16T16:46:07Z
#url: https://api.github.com/gists/1376e10e3619e9c65a6747261752af9c
#owner: https://api.github.com/users/ThomasLachaux

# send_email.py
import json
import boto3
import report
import datetime

client = boto3.client("ses")
emails = ["thomas@awesome-company.com", "alex@awesome-company.com"]

def lambda_handler(event, context):
    current_date = datetime.datetime.now().strftime("%d-%m-%Y")

    response = client.send_email(
        Destination={
            "ToAddresses": emails,
        },
        Message={
            "Body": {
                "Html": {
                    "Charset": "UTF-8",
                    "Data": report.render(),
                },
            },
            "Subject": {
                "Charset": "UTF-8",
                "Data": f"Security Hub Report - {current_date}",
            },
        },
        Source="no-reply@awesome-company",
    )

    return {"statusCode": 200, "body": json.dumps("MessageId is: " + response["MessageId"])}


if __name__ == "__main__":
    lambda_handler(None, None)