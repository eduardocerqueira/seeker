import logging
import json
import re
from configparser import ConfigParser
from datetime import datetime, timedelta
from pathlib import Path
from os import remove, rename, listdir
from subprocess import call, check_output

PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PACKAGE_DIR / "seeker.conf"
SNIPPET_DIR = PACKAGE_DIR / "snippet"
REPORT_FILE = PACKAGE_DIR / "report.txt"


def purge():
    for file in listdir(SNIPPET_DIR):
        try:
            with open(SNIPPET_DIR / file, "r", encoding="utf-8", errors="ignore") as fp:
                data = fp.read()
                # Continue with existing logic for valid files
                ...
        except (UnicodeDecodeError, OSError) as e:
            logging.error(f"Failed to process file {file}: {e}")
            continue