#!/bin/bash

rm -rf seeker/seeker.egg-info
rm -rf seeker.egg-info dist
python3 -m pip install --upgrade build
python3 setup.py bdist
