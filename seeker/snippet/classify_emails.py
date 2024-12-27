#date: 2024-12-27T16:56:31Z
#url: https://api.github.com/gists/f40a3b25421de285ccdfd1ae01bfe7d5
#owner: https://api.github.com/users/justinkater

import pandas as pd
import json
import re
import os
from pydantic import BaseModel, Field
from typing import Optional
import openai
import matplotlib.pyplot as plt
import logging
from concurrent.futures import ThreadPoolExecutor
import traceback
from tqdm import tqdm

# Path to the JSON file containing emails
email_file = "./data/emails.json"

logger = logging.getLogger(__name__)


# Function to load emails from the JSON file
def load_emails():
    if not os.path.exists(email_file):
        print("Email file not found.")
        return []
    with open(email_file, "r", encoding="utf-8") as f:
        return json.load(f)


# Function to save emails to the JSON file
def save_emails(emails):
    with open(email_file, "w", encoding="utf-8") as f:
        json.dump(emails, f, ensure_ascii=False, indent=4)


# Function to split the 'from' column into name and email address
def split_from_column(df):
    def split_from(value):
        if not isinstance(value, str):
            return None, None
        match = re.match(r"(.*) <(.*)>", value)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return value, None

    df[["name", "email"]] = df["from"].apply(lambda x: pd.Series(split_from(x)))
    return df


# Function to invoke LLM for email classification using new OpenAI parsing method
def classify_email_llm(email: str) -> Optional[str]:
    class EmailClassification(BaseModel):
        type: str = Field(
            description="""Classify the Type of email:
            Classify the following email content into one of the predefined email types. Analyze the text for intent, context, and the sender's purpose. The email type should represent the overall purpose of the email. Select the most appropriate category from the following list of types and provide only the classification type."
            Expanded List of Email Types:
            Newsletter - Periodic informational updates or announcements.
            Personal - Emails sent between individuals for private communication.
            Promotional - Offers, advertisements, or marketing content.
            Social Media - Notifications from social media platforms (e.g., likes, comments, or friend requests).
            App Notification - Updates, alerts, or confirmations from an application.
            Administration - Official communication regarding services or accounts (e.g., password resets, billing information).
            Event Invitation - Invitations to webinars, meetings, or personal events.
            Survey Request - Emails asking for feedback or survey participation.
            Job Opportunity - Communication related to job applications, interviews, or employment offers.
            Technical Support - Responses or updates regarding support tickets or troubleshooting.
            Order Confirmation - Receipts, invoices, or purchase details.
            Shipping Notification - Updates about order shipment or delivery status.
            Spam - Unsolicited or irrelevant emails.
            Legal Notice - Emails containing legal or compliance-related information.
            Financial Update - Statements, reports, or investment summaries.
            Subscription Renewal - Notices for renewing subscriptions or services.
            Educational - Course updates, materials, or notifications from educational institutions.
            Internal Communication - Emails within a company or organization (e.g., memos, meeting schedules).
            Community Updates - Emails related to community groups or forums.
            Security Alert - Notifications about suspicious account activity or breaches.
            Reminder - Emails reminding the recipient of deadlines, appointments, or tasks.
            Welcome Message - Initial onboarding emails for new accounts or subscriptions.
            Other - Any email type not covered by the above categories.
            """
        )

    system_prompt = ""

    template = f"Classify the email: {email}"

    llm = openai.OpenAI(
        api_key="xxxxxx",  # replace with your own API key
    )

    try:
        completion = llm.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": template},
            ],
            response_format=EmailClassification,
            temperature=0.7,
        )
        message = completion.choices[0].message
        print(
            f"Token Usage [prompt= "**********"={completion.usage.completion_tokens}, total={completion.usage.total_tokens}]"
        )
        if message.parsed:
            return message.parsed
        elif message.refusal:
            # handle refusal
            # logger.error(message.refusal)
            return {
                "success": False,
                "reason": message.refusal,
            }
    except Exception as e:
        # Handle edge cases
        if type(e) == openai.LengthFinishReasonError:
            logger.error("Too many tokens: "**********"
            return {"success": "**********": "Too many tokens"}
        else:
            # Handle other exceptions
            logger.error(e)
            return {"success": False, "reason": e.__str__()}


# Function to classify emails by LLM and add type property
def classify_emails(df):
    if "body" not in df.columns:
        print("Email body not found in data.")
        return df

    classifications = []

    def classify_email(row):
        if "type" in row:
            if row["type"] != "nan":
                print("Email already classified.")
                return row["type"]
        elif len(row["body"]) > 100000:
            print("Email body too long to classify.")
            return "Too Long"
        else:
            email_body = (
                row["body"] if isinstance(row["body"], str) else " ".join(row["body"])
            )
            try:
                classification = classify_email_llm(email_body)
                if classification:
                    if classification.type is not None:
                        return classification.type
                    else:
                        print(f"Failed to classify email: {classification['reason']}")
                        return "Failed"
                else:
                    print(f"Failed to classify email: {classification['reason']}")
                    traceback.print_exc()
                    return "Failed"
            except Exception as e:
                print(f"Failed to classify email: {e}")
                traceback.print_exc()
                return "Failed"

    with ThreadPoolExecutor() as executor:
        for idx, classification in tqdm(
            enumerate(executor.map(classify_email, [row for _, row in df.iterrows()])),
            total=len(df),
        ):
            df.at[idx, "type"] = classification
            save_emails(df.to_dict(orient="records"))

    if len(df["type"].dropna()) == 0:
        print("No email types found.")
        return df
    else:
        print(df["type"].tolist())
        pd.set_option("display.max_columns", None)  # Show all columns
        return df


# Function to visualize email types
def visualize_email_types(df):
    if "type" not in df.columns:
        print("No email types found to visualize.")
        return

    type_counts = df["type"].value_counts()
    type_counts.plot(kind="bar", figsize=(10, 6), title="Email Types Distribution")
    plt.xlabel("Email Type")
    plt.ylabel("Number of Emails")
    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    emails = load_emails()
    if not emails:
        print("No emails to process.")
    else:
        df = pd.DataFrame(emails)
        if "from" in df.columns:
            df = split_from_column(df)
        if len(df) > 0:
            df = classify_emails(df)
            visualize_email_types(df)
        else:
            print("No emails to process.")
t("No emails to process.")
