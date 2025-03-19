#date: 2025-03-19T16:55:16Z
#url: https://api.github.com/gists/a337dd2350f0adf12de80265950925c4
#owner: https://api.github.com/users/pschanely

import re
from typing import Optional

def parse_year(yearstring: str) -> Optional[int]:
    '''
    Something is wrong with this year parser! Can you guess what it is?
    
    post: __return__ is None or 1000 <= __return__ <= 9999
    '''
    return int(yearstring) if re.match('[1-9][0-9][0-9][0-9]', yearstring) else None
