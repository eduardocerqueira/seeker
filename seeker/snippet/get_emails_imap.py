#date: 2024-12-27T16:54:05Z
#url: https://api.github.com/gists/dfba934009735af5533added90a24a32
#owner: https://api.github.com/users/justinkater

import os
import imaplib
import email
import json
import traceback
from email import policy
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Replace these with your Email Server credentials
email_address = "xxx"
password = "**********"

# IMAP Server configuration
imap_server = "imap.gmail.com"
imap_port = 993  # Secure port

# Ensure the directory for saving emails exists
email_file = "./data/emails.json"
last_processed_file = "./data/last_processed.json"
os.makedirs(os.path.dirname(email_file), exist_ok=True)


def load_emails():
    if os.path.exists(email_file):
        with open(email_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_emails(emails):
    try:
        with open(email_file, "w", encoding="utf-8") as f:
            json.dump(emails, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print("Error saving emails:", e)


def load_last_processed():
    if os.path.exists(last_processed_file):
        with open(last_processed_file, "r", encoding="utf-8") as f:
            return json.load(f).get("last_processed_id")
    return None


def save_last_processed(last_processed_id):
    try:
        with open(last_processed_file, "w", encoding="utf-8") as f:
            json.dump(
                {"last_processed_id": last_processed_id},
                f,
                ensure_ascii=False,
                indent=4,
            )
    except Exception as e:
        print("Error saving last processed ID:", e)


def safe_decode(payload, charset):
    try:
        return payload.decode(charset or "utf-8")
    except (UnicodeDecodeError, AttributeError):
        return payload.decode("latin-1", errors="replace")


def extract_body(message):
    if message.is_multipart():
        return [
            safe_decode(part.get_payload(decode=True), part.get_content_charset())
            for part in message.walk()
            if part.get_content_type() == "text/plain"
        ]
    else:
        return [
            safe_decode(message.get_payload(decode=True), message.get_content_charset())
        ]


def create_imap_connection():
    """Create and return a new IMAP connection."""
    imap_conn = imaplib.IMAP4_SSL(imap_server, imap_port)
    imap_conn.login(email_address, password)
    imap_conn.select('"[Google Mail]/All Mail"')
    return imap_conn


def process_single_email(email_id, existing_message_ids, new_emails):
    """Process a single email using its own IMAP connection."""
    print(f"Processing email ID {email_id.decode()}...")
    imap_conn = None
    try:
        imap_conn = create_imap_connection()
        result, msg_data = imap_conn.fetch(email_id, "(RFC822)")
        if result != "OK":
            print(f"Failed to fetch email ID {email_id.decode()}")
            return

        raw_email = msg_data[0][1]
        message = email.message_from_bytes(raw_email, policy=policy.default)

        message_id = message["message-id"]
        if message_id not in existing_message_ids:
            email_data = {
                "message_id": message_id,
                "from": message["from"],
                "to": message["to"],
                "subject": message["subject"],
                "body": extract_body(message),
                "date": message["date"],
            }
            new_emails.append(email_data)
        print(f"Email ID {email_id.decode()} processed.")
    except Exception as e:
        print(f"Error processing email ID {email_id.decode()}:", e)
        traceback.print_exc()
        exit()
    finally:
        if imap_conn:
            try:
                imap_conn.logout()
            except:
                pass


def fetch_emails():
    try:
        print("Connecting to IMAP server...")
        imap_conn = create_imap_connection()

        print("Fetching email list...")
        result, data = imap_conn.search(None, "ALL")
        if result != "OK":
            print("Failed to fetch email list.")
            return

        email_ids = data[0].split()
        print(f"Total emails found: {len(email_ids)}")

        existing_emails = load_emails()
        existing_message_ids = {mail["message_id"] for mail in existing_emails}

        last_processed_id = load_last_processed()
        if last_processed_id:
            last_processed_index = email_ids.index(last_processed_id.encode()) + 1
            email_ids = email_ids[last_processed_index:]

        new_emails = []
        batch_size = 20

        for i in tqdm(range(0, len(email_ids), batch_size)):
            batch_ids = email_ids[i : i + batch_size]
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [
                    executor.submit(
                        process_single_email,
                        email_id,
                        existing_message_ids,
                        new_emails,
                    )
                    for email_id in batch_ids
                ]
                # Wait for all futures to complete
                for future in futures:
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error in thread execution: {e}")
                        traceback.print_exc()
                        exit()

            # Save emails after processing a batch
            if new_emails:
                existing_emails.extend(new_emails)
                save_emails(existing_emails)
                new_emails.clear()  # Clear the list after saving

            # Save the last processed email ID
            if batch_ids:
                save_last_processed(batch_ids[-1].decode())

        print(f"All emails processed and saved.")
        imap_conn.logout()

    except Exception as e:
        print("Error fetching emails:", e)
        traceback.print_exc()
        exit()
    finally:
        try:
            imap_conn.logout()
        except:
            pass


if __name__ == "__main__":
    fetch_emails()
