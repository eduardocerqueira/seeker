#date: 2025-01-09T17:04:24Z
#url: https://api.github.com/gists/e696de6de307fef99f00fbf7da5ffea6
#owner: https://api.github.com/users/laymonage

from functools import lru_cache
from pathlib import Path

from django.template import TemplateDoesNotExist
from django.template.loader import select_template


@lru_cache
def check_template_override(template_name, expected_location, base_path=None):
    """
    Check if a Django template has been overridden.
    """
    try:
        template = select_template([template_name])
    except TemplateDoesNotExist:
        return False

    root = Path(base_path or __file__).resolve().parent
    expected_path = str(root / expected_location / template_name)

    return template.origin.name != expected_path
