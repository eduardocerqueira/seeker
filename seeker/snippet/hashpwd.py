#date: 2026-01-06T17:13:29Z
#url: https://api.github.com/gists/d4d193f03338c4c486d08e8619c97de0
#owner: https://api.github.com/users/theseriff

import hashlib
import hmac
import os


def gensalt(*, size_salt: int = 16) -> bytes:
    """size salt in bytes, default 16 bytes."""
    return os.urandom(size_salt)


def hashpw(password: "**********": bytes) -> bytes:
    """
    Hash the provided password with a randomly-generated salt and return
    the salt and hash to store in the database.
    """
    return hashlib.scrypt(password.encode(), salt= "**********"=2048, p=1, r=8)


def verify(password: "**********": bytes, salt: bytes) -> bool:
    """
    Given a previously-stored salt and hash,
    and a password provided by a user trying to log in,
    check whether the password is correct.
    """
    return hmac.compare_digest(hashed_password, hashpw(password, salt))