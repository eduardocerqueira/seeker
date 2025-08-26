#date: 2025-08-26T17:13:27Z
#url: https://api.github.com/gists/d3fa94f367ca860d365a3327c8fa53ec
#owner: https://api.github.com/users/gohjiayi

#!/usr/bin/env python3
"""
Script to match unsafe input prompts from messages_all_three_systems.csv
with corresponding LEA responses from LEA-TransactionDataSet_fixed.csv
"""

import pandas as pd
import sys
from pathlib import Path


def main():
    # File paths
    messages_file = "messages_all_three_systems.csv"
    lea_file = "LEA-TransactionDataSet_fixed.csv"
    output_file = "lea_unsafe_chat_pairs.csv"

    # Check if input files exist
    if not Path(messages_file).exists():
        print(f"Error: {messages_file} not found")
        sys.exit(1)

    if not Path(lea_file).exists():
        print(f"Error: {lea_file} not found")
        sys.exit(1)

    print("Loading CSV files...")

    # Load the messages CSV (unsafe inputs)
    try:
        messages_df = pd.read_csv(messages_file)
        print(f"Loaded {len(messages_df)} messages from {messages_file}")
    except Exception as e:
        print(f"Error loading {messages_file}: {e}")
        sys.exit(1)

    # Load the LEA transaction dataset
    try:
        lea_df = pd.read_csv(lea_file)
        print(f"Loaded {len(lea_df)} records from {lea_file}")
    except Exception as e:
        print(f"Error loading {lea_file}: {e}")
        sys.exit(1)

    print("Processing matches...")

    # Create a dictionary for efficient ChatSessionId lookup
    # Group LEA data by ChatSessionId for faster lookup
    lea_by_session = {}
    for _, row in lea_df.iterrows():
        session_id = row['ChatSessionId']
        if session_id not in lea_by_session:
            lea_by_session[session_id] = []
        lea_by_session[session_id].append(row)

    # Sort each session's data by ConvId for proper ordering
    for session_id in lea_by_session:
        lea_by_session[session_id].sort(key=lambda x: x['ConvId'])

    results = []
    matched_count = 0

    # Process each unsafe message
    for _, msg_row in messages_df.iterrows():
        message_id = msg_row['message_id']
        conversation_id = msg_row['conversation_id']
        message = msg_row['message']

        # Look for matching ChatSessionId
        if conversation_id in lea_by_session:
            session_data = lea_by_session[conversation_id]

            # Find the matching input message
            input_match_found = False
            matches = []

            # First, find all matches to detect duplicates
            for i, lea_row in enumerate(session_data):
                if lea_row['Conversation'] == message:
                    matches.append((i, lea_row))

            # Warn if multiple matches found
            if len(matches) > 1:
                print(f"WARNING: Found {len(matches)} duplicate matches for message_id {message_id} "
                      f"in conversation_id {conversation_id}: '{message[:50]}...'")

            # Process all matches (including duplicates)
            if matches:
                for match_idx, (i, lea_row) in enumerate(matches):
                    # Found the input message, now look for the next LEA response
                    output = ""
                    topic_title = ""
                    input_conv_id = lea_row['ConvId']
                    output_conv_id = ""

                    # Look for the next row where CreatedBy is "LEA"
                    for j in range(i + 1, len(session_data)):
                        next_row = session_data[j]
                        if next_row['CreatedBy'] == 'LEA':
                            output = next_row['Conversation']
                            topic_title = next_row.get('TopicTitle', "")
                            output_conv_id = next_row['ConvId']
                            break

                    # Only add to results if we found a LEA response
                    if output:
                        results.append({
                            'message_id': message_id,
                            'input_conv_id': input_conv_id,
                            'output_conv_id': output_conv_id,
                            'chat_session_id': conversation_id,
                            'topic_title': topic_title,
                            'input': message,
                            'output': output
                        })
                        matched_count += 1
                    else:
                        print(f"WARNING: No LEA response found for message_id {message_id} "
                              f"in conversation_id {conversation_id}: '{message[:50]}...'")

                input_match_found = True

            # If no exact message match found, print warning and skip
            if not input_match_found:
                print(f"WARNING: No message match found for message_id {message_id} "
                      f"in conversation_id {conversation_id}: '{message[:50]}...'")
        else:
            # No matching ChatSessionId found, print warning and skip
            print(f"WARNING: No ChatSessionId match found for message_id {message_id} "
                  f"conversation_id {conversation_id}: '{message[:50]}...')")

    print(f"Found {matched_count} matches out of {len(messages_df)} unsafe messages")

    # Create output DataFrame
    output_df = pd.DataFrame(results)

    # Save to CSV
    try:
        output_df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"Results saved to {output_file}")
        print(f"Total records: {len(output_df)}")
        print(f"Records with LEA responses: {len(output_df[output_df['output'] != ''])}")
    except Exception as e:
        print(f"Error saving output file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()