#date: 2025-03-13T16:42:51Z
#url: https://api.github.com/gists/3eb1563cdbaf5f6423c5f545cf9c3642
#owner: https://api.github.com/users/dzakybd

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

def create_session():
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    )
    return session
  
def get_tiktok_info(username, video_id=None):
    session = create_session()
    tag = "userInfo"
    url = f"https://www.tiktok.com/@{username}"
    if video_id:
        url += f"/video/{video_id}"
        tag = "itemStruct"

    try:
        response = session.get(url)
        response.raise_for_status()

        content = response.text

        start_position = content.index(tag) - 2

        def find_closing_bracket_position(s, start_pos):
            # Initialize counter for brackets
            open_count = 0
            for i in range(start_pos, len(s)):
                if s[i] == "{":
                    open_count += 1
                elif s[i] == "}":
                    open_count -= 1
                # When open_count returns to zero, we've found the matching bracket
                if open_count == 0:
                    return i
            return -1  # Return -1 if no matching bracket is found

        end_position = find_closing_bracket_position(content, start_position) + 1

        return json.loads(content[start_position:end_position])
    except Exception as e:
        print(f"Error fetching TikTok {tag} for {username} {video_id}")
        return None