#date: 2024-12-30T16:49:09Z
#url: https://api.github.com/gists/205eb27d0b348ca46559ab8c719ad7f8
#owner: https://api.github.com/users/lpinkhard

import json
import logging
import random
import base64
from hashlib import sha256
from typing import Dict, Any, Optional

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Configuration
config = {
    "fido2_enabled": True,  # Toggle for enabling/disabling FIDO2 sign-in
    "salt": "static_salt_value",  # Salt for hashing
    "timeout": 120000  # Timeout for FIDO2 challenge in milliseconds
}

def configure(update: Dict[str, Any]):
    """
    Update the configuration values dynamically.

    Args:
        update (Dict[str, Any]): Configuration values to update.
    """
    config.update(update)

async def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Lambda handler for the Create Auth Challenge trigger.

    Args:
        event (Dict[str, Any]): Event data passed by AWS Lambda.

    Returns:
        Dict[str, Any]: Modified event data.

    Raises:
        Exception: For internal server errors or unrecognized sign-in methods.
    """
    logger.debug("Received event: %s", json.dumps(event, indent=2))

    try:
        if not event.get("request", {}).get("session"):
            logger.info("No session found. Creating initial session...")
            await provide_auth_parameters(event)
            await add_fido2_challenge(event)
        else:
            raise ValueError("Unrecognized session behavior")

        logger.debug("Final event: %s", json.dumps(event, indent=2))
        return event

    except Exception as error:
        logger.error("Error processing the event: %s", error)
        raise Exception("Internal Server Error")

async def provide_auth_parameters(event: Dict[str, Any]) -> None:
    """
    Provides initial authentication parameters for the client.

    Args:
        event (Dict[str, Any]): Event data to be modified.
    """
    logger.info("Setting up authentication parameters...")
    parameters = {
        "challenge": "PROVIDE_AUTH_PARAMETERS"
    }
    event["response"] = {
        "challengeMetadata": "PROVIDE_AUTH_PARAMETERS",
        "privateChallengeParameters": parameters,
        "publicChallengeParameters": parameters
    }

async def add_fido2_challenge(event: Dict[str, Any]) -> None:
    """
    Add a FIDO2 challenge to the event if FIDO2 is enabled.

    Args:
        event (Dict[str, Any]): Event data to be modified.
    """
    if config.get("fido2_enabled"):
        logger.info("Adding FIDO2 challenge to event...")
        challenge = generate_fido2_challenge()
        event["response"].setdefault("privateChallengeParameters", {})["fido2options"] = challenge
        event["response"].setdefault("publicChallengeParameters", {})["fido2options"] = challenge

def generate_fido2_challenge() -> str:
    """
    Generate a FIDO2 challenge using a random byte sequence.

    Returns:
        str: Base64-encoded FIDO2 challenge.
    """
    challenge = base64.urlsafe_b64encode(random.randbytes(64)).decode("utf-8")
    logger.debug("Generated FIDO2 challenge: %s", challenge)
    return challenge