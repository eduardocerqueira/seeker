#date: 2024-06-05T17:02:53Z
#url: https://api.github.com/gists/7b4149699b2f5c0a02216073f91d24cf
#owner: https://api.github.com/users/sgrilux

import boto3
import botocore

imagebuilder = boto3.client('imagebuilder', region_name='eu-west-1')

# Set the initial value of the token to None
token = "**********"

while True:
    try:
        # Call the list_images operation with the current token value
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
            response = imagebuilder.list_images(
                filters=[
                    {
                        'name': 'platform',
                        'values': ['Linux']
                    }
                ],
                owner='Amazon',
                maxResults=25,
                nextToken= "**********"
            )
        else:
            response = imagebuilder.list_images(
                filters=[
                    {
                        'name': 'platform',
                        'values': ['Linux']
                    }
                ],
                owner='Amazon',
                maxResults=25
            )

        # Print the ARNs of the listed images
        for image in response['imageVersionList']:
            print(image['arn'])

        # Check if there is a next token
        token = "**********"

        # If there is no next token, exit the loop
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
            break

    except botocore.exceptions.ClientError as error:
        print(f"Error: {error}")
        break
