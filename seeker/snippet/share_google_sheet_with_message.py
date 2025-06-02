#date: 2025-06-02T16:51:24Z
#url: https://api.github.com/gists/e16cda74dd3a407100db2ede19120216
#owner: https://api.github.com/users/bolablg

def share_google_sheet_with_message(sheet_id, email_list, acess, service_account_creds, message=""):
    # Authenticate with service account
    drive_service = build('drive', 'v3', credentials=service_account_creds)

    for email_to_share in email_list:
        # Create permission
        permission = {
            'type': 'user',
            'role': acess, #'reader',
            'emailAddress': email_to_share
        }

        # Share the file
        drive_service.permissions().create(
            fileId=sheet_id,
            body=permission,
            sendNotificationEmail=True,
            emailMessage=message
        ).execute()

    print(f"Google Sheet shared with {email_to_share} as {acess}.")