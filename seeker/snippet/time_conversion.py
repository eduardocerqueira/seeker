#date: 2025-05-14T17:12:27Z
#url: https://api.github.com/gists/5771a72bb705fd992161ac9f933a52ef
#owner: https://api.github.com/users/LiZoMark841619

from typing import Tuple
import logging
import sys
import re

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

def valid_time_input(s: str) -> bool:
    """Determine if the input is a valid time."""
    valid_pattern = r'^[0][0-9]|[1][0-2]:[0-5][0-9]:[0-5][0-9][AP]M$'
    return bool(re.match(valid_pattern, s))

def get_valid_time_input(prompt: str) -> str:
    """Get and validate time input from user."""
    while True:
        s = input(prompt)
        if valid_time_input(s):
            return s
        logger.error('Invalid input! Try again! ')
        
def split_string(s: str) -> Tuple[str]:
    hour, minute, second = s.split(':')
    sec, am_pm = second[:2], second[2:]
    return hour, minute, sec, am_pm

def timeConversion(strings_of_a_tuple: Tuple[str]) -> str:
    """Convert 12-hour time format to 24-hour format."""
    hour, minute, sec, am_pm = strings_of_a_tuple
    if am_pm.upper() == 'AM' and hour == '12':
        hour = '00'
    elif am_pm.upper() == 'PM' and hour != '12':
        hour = str(int(hour) + 12)
    return f'{hour}:{minute}:{sec}'

def main():
    """Main function to handle time conversion."""
    input_time = get_valid_time_input('Enter time (HH:MM:SSAM/PM): ')
    splitted_s = split_string(input_time)
    time_conversion = timeConversion(splitted_s)
    return time_conversion
    
if __name__ == '__main__':
    logger.info(main())