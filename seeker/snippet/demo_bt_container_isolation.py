#date: 2026-03-13T17:20:39Z
#url: https://api.github.com/gists/af8cd34343a53c3c8e62e2b7942624f6
#owner: https://api.github.com/users/az-rye

#!/usr/bin/env python3
"""Demo: "**********"

Shows the proposed session-based flow where developers never get a raw BT key.
They call us for a scoped session key, tokenize through BT Elements, and the
resulting token is locked to their container. A leaked token ID is useless to
another developer.

Usage:
    export BT_MANAGEMENT_KEY= "**********"=' .secrets.tmo | cut -d= -f2)
    export BT_PRIVATE_KEY= "**********"=' .secrets.env | cut -d= -f2)
    uv run python scripts/demo_bt_container_isolation.py

Requires:
    BT_MANAGEMENT_KEY  — management key (application:create, application:delete)
    BT_PRIVATE_KEY     — private key (token: "**********":delete, session:authorize)
"""

import asyncio
import os
import sys

import httpx

BT_API = "https://api.basistheory.com"

# Test card numbers (Basis Theory test environment — never real cards)
DEV_A_CARD = "4242424242424242"
DEV_B_CARD = "5555555555554444"


def bt_headers(api_key: str) -> dict[str, str]:
    return {"BT-API-KEY": api_key, "Content-Type": "application/json"}


async def create_public_app(
    client: httpx.AsyncClient, mgmt_key: str, name: str
) -> dict:
    """Create a temporary public BT application with token: "**********"
    resp = await client.post(
        f"{BT_API}/applications",
        headers=bt_headers(mgmt_key),
        json={
            "name": name,
            "type": "public",
            "permissions": "**********":create"],
        },
    )
    resp.raise_for_status()
    return resp.json()


async def create_session(
    client: httpx.AsyncClient, public_key: str
) -> dict:
    """Create a BT session (frontend side — uses public key)."""
    resp = await client.post(
        f"{BT_API}/sessions",
        headers=bt_headers(public_key),
    )
    resp.raise_for_status()
    return resp.json()


async def authorize_session(
    client: httpx.AsyncClient,
    private_key: str,
    nonce: str,
    container: str,
) -> None:
    """Authorize a session scoped to a developer container (backend side)."""
    resp = await client.post(
        f"{BT_API}/sessions/authorize",
        headers=bt_headers(private_key),
        json={
            "nonce": nonce,
            "rules": [
                {
                    "description": "**********"
                    "priority": 1,
                    "conditions": [
                        {
                            "attribute": "container",
                            "operator": "starts_with",
                            "value": container,
                        },
                    ],
                    "permissions": "**********":create"],
                    "transform": "mask",
                },
            ],
        },
    )
    resp.raise_for_status()


async def create_token_with_session_key(
    client: httpx.AsyncClient,
    session_key: str,
    container: str,
    card_number: str,
) -> dict:
    """Create a card token using a scoped session key (frontend side)."""
    resp = await client.post(
        f"{BT_API}/tokens",
        headers=bt_headers(session_key),
        json={
            "type": "card",
            "data": {
                "number": card_number,
                "expiration_month": 12,
                "expiration_year": 2028,
                "cvc": "123",
            },
            "containers": [container],
        },
    )
    resp.raise_for_status()
    return resp.json()


async def read_token(
    client: "**********": str, token_id: str
) -> httpx.Response:
    return await client.get(
        f"{BT_API}/tokens/{token_id}", headers= "**********"
    )


async def delete_token(
    client: "**********": str, token_id: str
) -> None:
    await client.delete(f"{BT_API}/tokens/{token_id}", headers= "**********"


async def delete_application(
    client: httpx.AsyncClient, api_key: str, app_id: str
) -> None:
    await client.delete(
        f"{BT_API}/applications/{app_id}", headers=bt_headers(api_key)
    )


def header(text: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}\n")


def ok(label: str) -> None:
    print(f"  \033[32m+ {label}\033[0m")


def fail(label: str) -> None:
    print(f"  \033[31m✗ {label}\033[0m")


def info(label: str) -> None:
    print(f"  \033[33m> {label}\033[0m")


def mask_key(key: str) -> str:
    return key[:20] + "..." + key[-4:]


async def tokenize_card_via_session(
    client: httpx.AsyncClient,
    public_key: str,
    private_key: str,
    developer_id: str,
    card_number: str,
) -> dict:
    """Full session flow: "**********"
    container = f"/developers/{developer_id}/"

    # Step 1: Frontend creates session (public key)
    session = await create_session(client, public_key)
    session_key = session["session_key"]
    nonce = session["nonce"]
    print(f"    1. Frontend creates session")
    print(f"       nonce:       {nonce}")
    print(f"       session_key: {mask_key(session_key)}")

    # Step 2: Backend authorizes session scoped to developer container
    await authorize_session(client, private_key, nonce, container)
    print(f"    2. Backend authorizes session -> {container}")

    # Step 3: "**********"
    token = "**********"
        client, session_key, container, card_number
    )
    print(f"    3. Frontend tokenizes card with session key")
    print(f"       token_id: "**********"
    print(f"       containers: "**********"

    return token


async def main() -> None:
    mgmt_key = os.environ.get("BT_MANAGEMENT_KEY")
    private_key = os.environ.get("BT_PRIVATE_KEY") or os.environ.get(
        "BASIS_THEORY_API_KEY_CHECKOUT_INTENTS"
    )

    if not mgmt_key:
        print("ERROR: set BT_MANAGEMENT_KEY (management key)")
        sys.exit(1)
    if not private_key:
        print("ERROR: set BT_PRIVATE_KEY or BASIS_THEORY_API_KEY_CHECKOUT_INTENTS")
        sys.exit(1)

    created_tokens: "**********"
    created_apps: list[str] = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # ------------------------------------------------------------------
            # Setup: Create a temporary public application
            # ------------------------------------------------------------------
            header("Setup — Create ephemeral public BT application")

            print("  Our backend owns a public BT app used to issue session keys.")
            print("  Creating one for this demo...\n")

            app = await create_public_app(client, mgmt_key, "demo-session-issuer")
            created_apps.append(app["id"])
            public_key = app["key"]
            print(f"    app_id:     {app['id']}")
            print(f"    type:       {app['type']}")
            print(f"    public_key: {mask_key(public_key)}")

            # ------------------------------------------------------------------
            # Part 1: "**********"
            # ------------------------------------------------------------------
            header("Part 1 — Developer A tokenizes card (session flow)")

            print("  Dev A calls POST /bt-session (authenticated as dev_a)\n")
            dev_a = "**********"
                client, public_key, private_key, "dev_a", DEV_A_CARD
            )
            created_tokens.append(dev_a["id"])

            print()
            header("Part 1b — Developer B tokenizes card (session flow)")

            print("  Dev B calls POST /bt-session (authenticated as dev_b)\n")
            dev_b = "**********"
                client, public_key, private_key, "dev_b", DEV_B_CARD
            )
            created_tokens.append(dev_b["id"])

            # ------------------------------------------------------------------
            # Part 2: "**********"
            # ------------------------------------------------------------------
            header("Part 2 — Leaked token: "**********"

            print(
                f"  Dev A's token ID leaks: "**********"
                "  Dev B submits it to our checkout API:\n"
                "\n"
                "    POST /checkout\n"
                f"    {{ paymentToken: "**********"
                "    Authorization: Bearer <dev_b_api_key>\n"
            )

            print("  Our backend authenticates Dev B, then checks the token: "**********"

            resp = "**********"
            token_data = "**********"
            token_containers = "**********"
            expected = "/developers/dev_b/"

            print(f"    Token's actual container: "**********"
            print(f'    Expected for authenticated B: ["{expected}"]')
            print()

 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"e "**********"x "**********"p "**********"e "**********"c "**********"t "**********"e "**********"d "**********"  "**********"n "**********"o "**********"t "**********"  "**********"i "**********"n "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"c "**********"o "**********"n "**********"t "**********"a "**********"i "**********"n "**********"e "**********"r "**********"s "**********": "**********"
                fail(
                    "REJECTED — container mismatch. "
                    "Dev A's card is NOT charged."
                )
            else:
                ok("ACCEPTED")

            print()
            info("The token ID is useless — Dev B cannot charge Dev A's card.\n")

        finally:
            # ------------------------------------------------------------------
            # Cleanup
            # ------------------------------------------------------------------
            header("Cleanup")
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"i "**********"d "**********"  "**********"i "**********"n "**********"  "**********"c "**********"r "**********"e "**********"a "**********"t "**********"e "**********"d "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********": "**********"
                await delete_token(client, private_key, token_id)
                print(f"    Deleted token {token_id}")
            for app_id in created_apps:
                await delete_application(client, mgmt_key, app_id)
                print(f"    Deleted app   {app_id}")
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"c "**********"r "**********"e "**********"a "**********"t "**********"e "**********"d "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"  "**********"a "**********"n "**********"d "**********"  "**********"n "**********"o "**********"t "**********"  "**********"c "**********"r "**********"e "**********"a "**********"t "**********"e "**********"d "**********"_ "**********"a "**********"p "**********"p "**********"s "**********": "**********"
                print("    Nothing to clean up.")
            print()


if __name__ == "__main__":
    asyncio.run(main())
