#date: 2024-01-09T16:58:47Z
#url: https://api.github.com/gists/646ad533d8d6c0a762e00ade1b151cd6
#owner: https://api.github.com/users/simonLeary42

def string_matches_any_globs(x:str, globs:List[str]):
    return any(fnmatch.fnmatch(x, y) for y in globs)

def find_files(walk_root_path:str, include_globs:List[str], exclude_globs:List[str]=None):
    """
    excluding takes precidence over including
    """
    output = []
    for grandparent_dirname, parent_basenames, basenames in os.walk(walk_root_path):
        for parent_basename in parent_basenames:
            for basename in basenames:
                path = os.path.join(grandparent_dirname, parent_basename, basename)
                if exclude_globs is not None and string_matches_any_globs(path, exclude_globs):
                    continue
                if string_matches_any_globs(path, include_globs):
                    output.append(path)
    return output