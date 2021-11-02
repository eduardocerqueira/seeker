#date: 2021-11-02T17:04:26Z
#url: https://api.github.com/gists/8d40b3a13394a0feab21cdf2edf0b9d7
#owner: https://api.github.com/users/mypy-play

from typing import List, Optional, Union

def get_relevant_params() -> List[str]:
    return sorted(list(set(['1', '2', '3'])))

# also doesn't work with: ... -> Union[List[str], None]
def get_config(config_path: Optional[str] = None) -> Optional[List[str]]:
    
    # this doesn't work
    if config_path:
        read_cols: Optional[List[str]] = get_relevant_params()
    else:
        read_cols = None
        
    # # this works
    # read_cols = get_relevant_params() if config_path else None

    return read_cols