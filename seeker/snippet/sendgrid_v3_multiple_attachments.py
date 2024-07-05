#date: 2024-07-05T16:58:47Z
#url: https://api.github.com/gists/422650380d95182a80b4298cfc4577e6
#owner: https://api.github.com/users/robinreni96

import os
import base64
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import (
    Mail, Attachment, FileContent, FileName,
    FileType, Disposition, ContentId, Personalization, Email, Content, To)


# Use SendGrid API key
api_key = ""

def add_multiple_attachment(folder_path):
    attachement_list = []
    # Get a list of all files and directories in the folder
    contents = os.listdir(folder_path)

    # Filter the list to get only the file names
    file_names = [file for file in contents if os.path.isfile(os.path.join(folder_path, file))]

    # Print the file names
    for file_name in file_names:
        
        # file path
        file_path = os.path.join(folder_path, file_name)

        print(file_path)

        with open(file_path, 'rb') as f:
            data = f.read()

        # Encode contents of file as Base 64
        encoded = base64.b64encode(data).decode()

        """Build attachment"""
        attachment = Attachment()
        attachment.file_content = FileContent(encoded)
        attachment.file_type = FileType('application/pdf')
        attachment.file_name = FileName(file_name)
        attachment.disposition = Disposition('attachment')
        attachment.content_id = ContentId(file_name)
        mail.attachment = attachment

        attachement_list.append(attachment)

    return attachement_list


# Define the email details
from_email = Email('noreply@email.com')
to_email = Email('user@email.com')
subject = "Test"
content = Content("text/plain", "This email contains multiple attachments.")

# Create the mail object
mail = Mail(from_email=from_email, subject=subject, plain_text_content=content)

# Create a Personalization object
personalization = Personalization()
personalization.add_to(To('user@email.com', 'Example User 1'))
mail.add_personalization(personalization)

# Define the folder path containing the attachment files
folder_path = ""

# Get the list of attachment files
attachement_files = add_multiple_attachment(folder_path)

# Add the attachment files to the mail object
mail.attachment = attachement_files

try:
    # Create a SendGrid client and send the email
    sendgrid_client = SendGridAPIClient(api_key)
    response = sendgrid_client.send(message=mail)
    print(response.status_code)
    print(response.body)
    print(response.headers)
except Exception as e:
    print(str(e))