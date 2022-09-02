#date: 2022-09-02T16:57:02Z
#url: https://api.github.com/gists/5b2cdac99c819e6017b7097320536b43
#owner: https://api.github.com/users/adamchainz

from io import StringIO
from django.core.management import call_command
from django.test import TestCase


class PendingMigrationsTests(TestCase):
    def test_no_pending_migrations(self):
        out = StringIO()
        try:
            call_command(
                "makemigrations",
                "--dry-run",
                "--check",
                stdout=out,
                stderr=StringIO(),
            )
        except SystemExit:  # pragma: no cover
            raise AssertionError(
                "Pending migrations:\n" + out.getvalue()
        ) from None