#date: 2022-09-16T17:16:47Z
#url: https://api.github.com/gists/a68837f23b492a68a8ba88621a47fde4
#owner: https://api.github.com/users/EnriqueSoria

class Env:
    """
    Mock for django-environ Env

    https://django-environ.readthedocs.io/en/latest/index.html
    """
    def __init__(self, **scheme):
        self.scheme = scheme


def get_env_sample_text(env: Env, write_defaults: bool = False) -> str:
    text = ""
    for env_name, env_config in env.scheme.items():
        has_default: bool = len(env_config) > 1

        if has_default:
            if write_defaults:
                cast, default_value = env_config
                text += f"{env_name}={default_value}\n"
        else:
            text += f"{env_name}=\n"

    return text


if __name__ == "__main__":
    env = Env(
        DEBUG=(bool, False),
        HELLO=(str,),
    )

    print(get_env_sample_text(env))
    # outputs:
    """
    HELLO=
    """

    print(get_env_sample_text(env, write_defaults=True))
    # outputs:
    """
    DEBUG=False
    HELLO=
    """
