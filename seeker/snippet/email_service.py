#date: 2025-09-09T16:55:50Z
#url: https://api.github.com/gists/c43d6109d4525f5d90fe4781f9e41eae
#owner: https://api.github.com/users/Harshal-3558

import boto3
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from django.template.loader import render_to_string
from django.utils.html import strip_tags
from django.conf import settings
from django.core.mail import send_mail

class InvitationTypes:
    Workspace = 'workspace'
    SignUp = 'signup'


def send_invite_email(emailId, invite_link, invitation_type=InvitationTypes.SignUp):
    try:
        # Render the appropriate HTML template
        if invitation_type == InvitationTypes.SignUp:
            html_body = render_to_string('../templates/invite_email.html', {'invite_link': invite_link})
        elif invitation_type == InvitationTypes.Workspace:
            html_body = render_to_string('../templates/workspace_invitation.html', {'invite_link': invite_link})
        else:
            raise ValueError("Invalid invitation_type provided")

        # Convert HTML to plain text fallback
        text_body = strip_tags(html_body)

        # Create MIME message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = "You're invited to join Databrewery!"
        msg['From'] = settings.EMAIL_FROM_ADDRESS
        msg['To'] = emailId
        msg.attach(MIMEText(text_body, 'plain'))
        msg.attach(MIMEText(html_body, 'html'))

        # Create SES client
        ses = boto3.client(
            'ses',
            region_name=settings.AWS_SES_REGION,
            aws_access_key_id= "**********"
            aws_secret_access_key= "**********"
        )

        # Send email via SES
        ses.send_raw_email(
            Source=settings.EMAIL_FROM_ADDRESS,
            Destinations=[emailId],
            RawMessage={'Data': msg.as_string()},
        )

    except Exception as e:
        logging.exception(f"Failed to send invite email to {emailId}: {e}")
        raise

def send_login_email(emailId, login_link):
    """
    Sends a magic login link to the user using AWS SES.
    """
    try:
        # Render the HTML template for the magic login email
        html_body = render_to_string('users/magic_login_email.html', {'login_link': login_link})
        text_body = strip_tags(html_body)
        subject = "Your Magic Login Link"

        send_mail(
            subject,
            text_body,
            settings.EMAIL_FROM_ADDRESS,
            [emailId],
            html_message=html_body
        )

    except Exception as e:
        logging.exception(f"Failed to send magic login email to {emailId}: {e}")
        raise
email to {emailId}: {e}")
        raise
