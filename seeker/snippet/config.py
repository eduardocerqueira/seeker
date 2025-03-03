#date: 2025-03-03T17:02:00Z
#url: https://api.github.com/gists/74e601913dc6480c7229c138b629da8b
#owner: https://api.github.com/users/drorasaf

from dynaconf import Dynaconf, Validator


settings = Dynaconf(
    envar_prefix="DYNACONF",
    settings_files="dynaconf_settings.toml",
    validators=[
        Validator(
            "service_account_temp_file",
            "bucket_name",
            is_type_of=str,
            must_exist=True,
        ),
    ],
)