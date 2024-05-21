#date: 2024-05-21T16:44:54Z
#url: https://api.github.com/gists/4034ccfd10291a86366aa57ec1de7067
#owner: https://api.github.com/users/sboysel

"""
simple way to get the directory path of the current file.

Extension: simply add additonal `.parent`s if __file__ is in a subdirectory of ROOT
"""
import pathlib
ROOT = pathlib.Path(__file__).parent