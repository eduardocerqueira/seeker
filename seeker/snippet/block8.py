#date: 2022-03-07T17:00:30Z
#url: https://api.github.com/gists/02dc144e48daa83e590646f47830ca94
#owner: https://api.github.com/users/DanyF-github

from flask import Flask, render_template, request
from decouple import config
from opentok import Client