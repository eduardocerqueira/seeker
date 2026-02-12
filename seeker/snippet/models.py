#date: 2026-02-12T17:46:36Z
#url: https://api.github.com/gists/0f8f1a7d3f2d4b62584d5c5ab8d10347
#owner: https://api.github.com/users/dryan

from django.core.exceptions import ValidationError
from django.core.validators import RegexValidator
from django.core.validators import URLValidator
from django.db import models
from django.utils.translation import gettext_lazy as _

path_validator = RegexValidator(r"^/", _("Path must start with /"))


def validate_path_or_url(value: str) -> None:
    err = None
    for validator in [path_validator, URLValidator()]:
        try:
            validator(value)
            return True
        except ValidationError as exc:
            err = exc
    raise err


class Redirect(models.Model):
    path = models.CharField(
        _("path"),
        max_length=255,
        validators=[path_validator],
        db_index=True,
        help_text=_("must start with a /"),
    )
    destination = models.CharField(
        _("destination"),
        max_length=255,
        validators=[validate_path_or_url],
        help_text=_(
            "must start with a / or be a full URL with http:// or https:// "
            "at the beginning."
        ),
    )
    is_system = models.BooleanField(
        _("system managed"),
        default=False,
        editable=False,
    )
    used = models.PositiveBigIntegerField(
        _("times used"),
        default=0,
        editable=False,
    )

    class Meta:
        ordering = ["path"]

    def __str__(self):
        return f"{self.path} → {self.destination}"
