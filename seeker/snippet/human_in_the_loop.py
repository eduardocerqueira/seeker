#date: 2025-11-04T17:01:45Z
#url: https://api.github.com/gists/628246dc4f851dda45f57b492c645ec0
#owner: https://api.github.com/users/sydney-runkle

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain.agents.middleware import HumanInTheLoopMiddleware


@tool(parse_docstring=True)
def send_email(recipient: str, subject: str, body: str) -> str:
    """Send an email to a recipient.

    Args:
        recipient: Email address of the recipient.
        subject: Subject line of the email.
        body: Body content of the email.

    Returns:
        Confirmation message.
    """
    return f"Email sent successfully to {recipient}"


agent = create_agent(
    model="openai-gpt-4o",
    tools=[send_email],
    system_prompt="You are a helpful assistant for Sydney that can send emails.",
    middleware=[HumanInTheLoopMiddleware(interrupt_on={"send_email": True})],
)


low_stakes_email = """
Respond to the following email:
From: alice@example.com
Subject: Coffee?
Body: Hey, would you like to grab coffee next week?
"""

# Consequential email
consequential_email = """
Respond to the following email:
From: partner@startup.com
Subject: Budget proposal for Q1 2026
Body: Hey Sydney, we need your sign-off on the $1M engineering budget for Q1. Can you approve and reply by EOD? This is critical for our timeline.
"""

# Approval decision
approval = {
    "decisions": [
        {
            "type": "approve"
        }
    ]
}

# Edit decision
edit = {
    "decisions": [
        {
            "type": "edit",
            "edited_action": {
                "name": "send_email",
                "args": {
                    "recipient": "partner@startup.com",
                    "subject": "Budget proposal for Q1 2026",
                    "body": "I can only approve up to 500k, please send over details.",
                }
            }
        }
    ]
}

# Reject decision
reject = {
    "decisions": [
        {
            "type": "reject",
            "message": "Please edit the email asking for more details about the budget proposal, then send the email"
        }
    ]
}
