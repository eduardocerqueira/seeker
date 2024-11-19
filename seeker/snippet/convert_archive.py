#date: 2024-11-19T16:51:28Z
#url: https://api.github.com/gists/e19d5f1b178c45c4034480a849ab3a09
#owner: https://api.github.com/users/fastdaima

import argparse
import json
import logging
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MediaFile:
    id: str
    content_type: str
    path: str
    metadata: Dict[str, Any]

@dataclass
class Content:
    id: str
    text: str
    metadata: Dict[str, Any]
    timestamp: str
    parent_id: Optional[str]
    media_files: List[Dict[str, Any]]
    content_source: str

@dataclass
class Thread:
    id: str
    contents: List[Content]

@dataclass
class Message:
    role: Literal["assistant", "user"]
    content: str

# Data extraction functions
def clean_json_string(json_string: str) -> str:
    return re.sub(r'^window\.[^=]+=\s*', '', json_string.strip()).rstrip(';')

def process_file(file_path: str) -> List[Dict[str, Any]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = clean_json_string(f.read())
            results = json.loads(data)
            return results
    except Exception as e:
        logger.warning(f"Error processing file {file_path}: {e}")
        return []

def extract_manifest(file_path: str) -> Dict[str, Any]:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = clean_json_string(file.read())
            return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r'window\.__THAR_CONFIG\s*=\s*({.*})', content, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        logger.error(f"Could not parse __THAR_CONFIG in manifest file: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error extracting manifest from {file_path}: {e}")
        raise

def get_media_files(tweet_id: str, media_folder: str) -> List[str]:
    try:
        all_files = os.listdir(media_folder)
        media_files = [
            f for f in all_files 
            if f.startswith(f"{tweet_id}-") and os.path.getsize(os.path.join(media_folder, f)) > 0
        ]
        return media_files
    except Exception as e:
        logger.error(f"Error getting media files for tweet_id {tweet_id}: {e}")
        return []

def get_media_type(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext in ('.mp4', '.mov'):
        return 'video'
    elif ext in ('.jpg', '.jpeg', '.png', '.gif'):
        return 'photo'
    return 'unknown'

def extract_content(item: Dict[str, Any], content_source: str, media_folder: str) -> List[Content]:
    content_id = item.get('id') or item.get('tweetId')
    text = item.get('text') or item.get('fullText') or item.get('full_text')

    media_files = get_media_files(content_id, media_folder)
    media_file_objects = [{
        'id': f"{content_id}_{os.path.splitext(media_file)[0]}",
        'content_type': get_media_type(media_file),
        'path': os.path.join(media_folder, media_file),
        'metadata': {
            'parent_tweet': item,
            'media_info': item.get('extended_entities', {}).get('media', [])
        }
    } for media_file in media_files]

    return [Content(
        id=content_id,
        text=text,
        metadata=item,
        timestamp=item.get('created_at', ''),
        parent_id=item.get('in_reply_to_status_id', None),
        media_files=media_file_objects,
        content_source=content_source
    )]

def process_file_wrapper(args: Tuple[str, Dict[str, Any], str, str]) -> List[Content]:
    archive_path, file_info, extractor_name, media_folder = args
    file_path = os.path.join(archive_path, file_info['fileName'])
    file_data = process_file(file_path)
    extractor = globals()[extractor_name]  # Get the extractor function by name
    return extractor(file_data, media_folder)

def extract_content_data(archive_path: str, file_info: Dict[str, Any], extractor: Callable, media_folder: str) -> List[Content]:
    try:
        return extractor(file_info['data'], media_folder)
    except Exception as e:
        logger.error(f"Error extracting data with {extractor.__name__}: {e}")
        return []

def extract_data(archive_path: str, type_info: Dict[str, Any], extractor: Callable) -> List[Content]:
    media_folder = os.path.join(archive_path, 'data', 'tweets_media')
    contents = []
    extractor_name = extractor.__name__

    with ProcessPoolExecutor() as executor:
        args_list = [
            (archive_path, file_info, extractor_name, media_folder) 
            for file_info in type_info.get('files', [])
        ]
        futures = [executor.submit(process_file_wrapper, args) for args in args_list]

        total_futures = len(futures)
        logger.info(f"Processing {total_futures} files with {extractor_name}")
        completed_count = 0

        for future in as_completed(futures):
            result = future.result()
            if result:
                contents.extend(result)
            completed_count += 1
            if completed_count % 10 == 0 or completed_count == total_futures:
                logger.info(f"Processed {completed_count}/{total_futures} files")

    logger.info(f"Total {extractor_name} extracted: {len(contents)} from {len(type_info.get('files', []))} files")
    return contents

def extract_tweets(file_data: List[Dict[str, Any]], media_folder: str) -> List[Content]:
    logger.info(f"Extracting tweets from {len(file_data)} items")
    contents = [
        content 
        for tweet in file_data if 'tweet' in tweet 
        for content in extract_content(tweet['tweet'], 'tweet', media_folder)
    ]
    logger.info(f"Extracted {len(contents)} tweet contents")
    return contents

def extract_likes(file_data: List[Dict[str, Any]], media_folder: str) -> List[Content]:
    logger.info(f"Extracting likes from {len(file_data)} items")
    contents = [
        content 
        for like in file_data if 'like' in like 
        for content in extract_content(like['like'], 'like', media_folder)
    ]
    logger.info(f"Extracted {len(contents)} like contents")
    return contents

def extract_archive_data(archive_path: str) -> Dict[str, List[Content]]:
    try:
        manifest_path = os.path.join(archive_path, 'data', 'manifest.js')
        manifest = extract_manifest(manifest_path)
        data_types = manifest.get('dataTypes', {})
        
        extractors = {
            'tweets': extract_tweets,
            'like': extract_likes,
            # Add more extractors as needed
        }
        
        response = {}
        for data_type, extractor in extractors.items():
            if data_type in data_types:
                response[data_type] = extract_data(archive_path, data_types[data_type], extractor)
        
        return response
    
    except Exception as e:
        logger.error(f"Error occurred during data extraction: {e}")
        return {}

# Data transformation functions
def clean_text(text: str, entities: Optional[Dict] = None) -> str:
    if entities:
        for url in entities.get('urls', []):
            short_url = url.get('url', '')
            expanded_url = url.get('expanded_url', '')
            if short_url and expanded_url:
                text = text.replace(short_url, expanded_url)
    
    text = re.sub(r'https://t.co/\w+', '', text)
    text = re.sub(r'@\w+', '', text)  
    text = re.sub(r'#\w+', '', text)  
    text = re.sub(r'\s+', ' ', text)  
    return text.strip()

def get_all_tweets(data: Dict[str, List[Content]]) -> Dict[str, Content]:
    logger.info("Combining tweets and likes into all_tweets")
    all_tweets = {tweet.id: tweet for tweet in data.get('tweets', []) if tweet.id}
    logger.info(f"Added {len(data.get('tweets', []))} tweets to all_tweets")
    
    likes = data.get('like', [])
    for like in likes:
        if like.id:
            all_tweets[like.id] = like
        else:
            logger.warning("Like without id encountered and skipped.")
    logger.info(f"Added {len(likes)} likes to all_tweets")
    logger.info(f"Total {len(all_tweets)} tweets/likes in all_tweets")
    
    return all_tweets

def get_conversation_texts(conversation: List[Content]) -> List[Tuple[str, str]]:
    return [
        (tweet.text, "assistant" if 'full_text' in tweet.metadata else "user")
        for tweet in conversation
        if tweet.text
    ]

def trim_conversation_to_last_assistant(conversation_data: List[Message]) -> List[Message]:
    for i in range(len(conversation_data) - 1, -1, -1):
        if conversation_data[i].role == "assistant":
            return conversation_data[:i+1]
    return []

def get_conversation_data(conversation: List[Content]) -> List[Message]:
    conversation_data = []
    current_role = None
    current_content = []

    for text, role in get_conversation_texts(conversation):
        cleaned_text = clean_text(text)
        if cleaned_text:
            if role != current_role and current_role is not None:
                conversation_data.append(format_message(current_content, current_role))
                current_content = []
            current_role = role
            current_content.append(cleaned_text)

    if current_content:
        conversation_data.append(format_message(current_content, current_role))

    return trim_conversation_to_last_assistant(conversation_data)

def extract_threads_and_conversations(all_tweets: Dict[str, Content]) -> Tuple[List[Thread], List[List[Content]]]:
    """Extract threads and conversations from all tweets."""
    threads = []
    conversations = []

    # Keep track of processed tweet IDs to avoid duplicates
    processed_ids = set()

    for tweet in all_tweets.values():
        if tweet.id in processed_ids:
            continue

        if tweet.content_source == 'tweet' and tweet.parent_id and tweet.parent_id in all_tweets and not tweet.text.startswith('RT'):
            # Initialize the chain
            chain = [tweet]
            current_tweet = tweet

            # Walk up the chain of replies
            while current_tweet.parent_id and current_tweet.parent_id in all_tweets:
                parent_tweet = all_tweets[current_tweet.parent_id]
                chain.append(parent_tweet)
                current_tweet = parent_tweet

                if current_tweet.id in processed_ids:
                    break  # Avoid cycles

            # Mark tweets as processed
            for t in chain:
                processed_ids.add(t.id)

            # Determine if it's a thread or conversation
            if all(t.content_source == 'tweet' for t in chain):
                # This is a thread (user replying to themselves)
                threads.append(Thread(id=tweet.id, contents=list(reversed(chain))))
            else:
                # This is a conversation (user replying to others)
                conversations.append(list(reversed(chain)))

    return threads, conversations

# Data export functions
def process_media_files(media_files: List[Dict[str, Any]], images_folder: str) -> List[str]:
    media_links = []
    for media_file in media_files:
        media_path = media_file.get('path')
        if media_path and os.path.isfile(media_path):
            orig_filename = os.path.basename(media_path)
            new_filename = f"_{orig_filename}"
            dest_path = os.path.join(images_folder, new_filename)
            shutil.copy(media_path, dest_path)
            media_links.append(f"![{new_filename}](images/{new_filename})")
        else:
            logger.warning(f"Invalid or missing media path: {media_path}")
    return media_links
def save_thread_markdown(thread: Thread, output_dir: str, media_folder: str, images_folder: str):
    if not thread.contents:
        logger.warning("Attempted to save an empty thread.")
        return

    try:
        date_str = thread.contents[0].timestamp
        date = datetime.strptime(date_str, '%a %b %d %H:%M:%S %z %Y').date()
    except ValueError:
        logger.warning(f"Invalid date format: {date_str}")
        date = datetime.today().date()

    frontmatter = f"---\nDate: {date.isoformat()}\n---\n"

    thread_text = []
    for tweet in thread.contents:
        media_links = process_media_files(tweet.media_files, images_folder)
        cleaned_text = clean_text(tweet.text, tweet.metadata.get('entities'))
        combined_text = f"{cleaned_text}\n\n" + '\n\n'.join(media_links)
        thread_text.append(combined_text)

    first_words = ' '.join(thread_text[0].split()[:5])
    sanitized_filename = re.sub(r'[^\w\-_ ]', '', first_words).strip().replace(' ', '_')[:50]
    filename = f"{sanitized_filename}.md"
    file_path = os.path.join(output_dir, filename)

    top_tweet_id = thread.contents[0].id
    top_tweet_link = f"https://twitter.com/i/web/status/{top_tweet_id}"

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"{frontmatter}\n\n" + '\n\n'.join(thread_text) + f"\n\n[View on Twitter]({top_tweet_link})")

def save_tweets_by_date(all_tweets: Dict[str, Content], threads: List[Thread], output_dir: str, images_folder: str):
    thread_ids = {tweet.id for thread in threads for tweet in thread.contents}
    non_thread_tweets = [
        tweet for tweet_id, tweet in all_tweets.items()
        if tweet_id not in thread_ids 
        and not tweet.parent_id 
        and tweet.content_source == 'tweet'
        and not tweet.text.startswith('RT')
    ]

    tweets_by_date: Dict[datetime.date, List[Content]] = {}
    for tweet in non_thread_tweets:
        date_str = tweet.timestamp
        if not date_str:
            logger.warning(f"Tweet missing date information: {tweet}")
            continue
        try:
            date = datetime.strptime(date_str, '%a %b %d %H:%M:%S %z %Y').date()
            tweets_by_date.setdefault(date, []).append(tweet)
        except ValueError:
            logger.warning(f"Invalid date format: {date_str}")

    for date, tweets_on_date in tweets_by_date.items():
        filename = f"{date.isoformat()}.md"
        file_path = os.path.join(output_dir, filename)
        tweets_on_date.sort(key=lambda x: x.timestamp)
        content = '\n\n---\n\n'.join(
            f"*{datetime.strptime(tweet.timestamp, '%a %b %d %H:%M:%S %z %Y').strftime('%I:%M %p')}*  \n{clean_text(tweet.text, tweet.metadata.get('entities'))}" +
            ''.join(process_media_files(tweet.media_files, images_folder))
            for tweet in tweets_on_date
        )
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

def format_message(content: List[str], role: Literal['assistant', 'user']) -> Message:
    return Message(role=role, content="\n\n".join(content))

def format_conversation(conversation_data: List[Message], system_message: str) -> Dict[str, Any]:
    messages = [{"role": "system", "content": system_message}]
    messages.extend([message.__dict__ for message in conversation_data])
    return {"messages": messages}

def save_conversations_to_jsonl(threads: List[Thread], conversations: List[List[Content]], output_path: str, system_message: str = "You have been uploaded to the internet"):
    logger.info(f"Saving {len(conversations) + len(threads)} conversations to {output_path} in oai format")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for thread in threads:
            formatted_thread = get_conversation_data(thread.contents)
            if not formatted_thread:
                continue
            formatted_thread = format_conversation(formatted_thread, system_message)
            f.write(json.dumps(formatted_thread) + '\n')

        for conversation in conversations:
            formatted_conv = get_conversation_data(conversation)
            if not formatted_conv:
                continue
            formatted_conv = format_conversation(formatted_conv, system_message)
            f.write(json.dumps(formatted_conv) + '\n')

def main(archive_path: str, output_dir: str, output_formats: List[str], system_message: str):
    data = extract_archive_data(archive_path)
    all_tweets = get_all_tweets(data)
    threads, conversations = extract_threads_and_conversations(all_tweets)

    if 'markdown' in output_formats:
        threads_output_dir = os.path.join(output_dir, 'threads')
        images_folder = os.path.join(output_dir, 'images')
        non_thread_output_dir = os.path.join(output_dir, 'tweets_by_date')

        os.makedirs(threads_output_dir, exist_ok=True)
        os.makedirs(images_folder, exist_ok=True)
        os.makedirs(non_thread_output_dir, exist_ok=True)

        logger.info(f"Saving {len(threads)} threads")
        for i, thread in enumerate(threads, start=1):
            save_thread_markdown(
                thread, 
                threads_output_dir, 
                os.path.join(archive_path, 'data', 'tweets_media'), 
                images_folder
            )
            if i % 10 == 0 or i == len(threads):
                logger.info(f"Saved {i}/{len(threads)} threads")

        save_tweets_by_date(all_tweets, threads, non_thread_output_dir, images_folder)

    if 'oai' in output_formats:
        output_path = os.path.join(output_dir, 'conversations_oai.jsonl')
        save_conversations_to_jsonl(threads, conversations, output_path, system_message)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Twitter archive")
    parser.add_argument("--archive-path", default="test", help="Path to the Twitter archive directory")
    parser.add_argument("--output-dir", default="output", help="Directory where outputs will be saved")
    parser.add_argument("--output-formats", nargs='+', default=['markdown', 'oai'],
                        help="Output formats to generate (markdown, oai)")
    parser.add_argument("--system-message", default="You have been uploaded to the internet", 
                        help="System message for the conversation")
    args = parser.parse_args()

    main(args.archive_path, args.output_dir, args.output_formats, args.system_message)
