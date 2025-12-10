#date: 2025-12-10T17:15:49Z
#url: https://api.github.com/gists/a8db65b3179697cf590af5dfed52f234
#owner: https://api.github.com/users/soenmie

"""
Twilio Phone Number Purchase - Async Implementation

This module demonstrates how to search for and purchase phone numbers
from different countries using Twilio's async Python SDK.

Requirements:
- twilio>=8.0.0
- Python 3.9+

Usage:
    PYTHONPATH=. python twilio_phone_purchase.py
"""

import asyncio
import json
from typing import Optional
from twilio.http.async_http_client import AsyncTwilioHttpClient
from twilio.rest import Client

# Configuration - replace with your credentials
TWILIO_ACCOUNT_SID = "your_account_sid"
TWILIO_AUTH_TOKEN = "**********"


async def list_regulatory_bundles() -> list[dict]:
    """List all regulatory bundles in the account.
    
    Returns:
        List of bundle information dictionaries
    """
    async with AsyncTwilioHttpClient() as http_client:
        client = Client(
            TWILIO_ACCOUNT_SID,
            TWILIO_AUTH_TOKEN,
            http_client=http_client,
        )
        bundles = await client.numbers.v2.regulatory_compliance.bundles.list_async()
        
        return [
            {
                "sid": b.sid,
                "friendly_name": b.friendly_name,
                "status": b.status,
                "regulation_sid": b.regulation_sid,
                "date_created": str(b.date_created),
            }
            for b in bundles
        ]


async def list_addresses() -> list[dict]:
    """List all addresses in the account.
    
    Returns:
        List of address information dictionaries
    """
    async with AsyncTwilioHttpClient() as http_client:
        client = Client(
            TWILIO_ACCOUNT_SID,
            TWILIO_AUTH_TOKEN,
            http_client=http_client,
        )
        addresses = await client.addresses.list_async()
        
        return [
            {
                "sid": a.sid,
                "friendly_name": a.friendly_name,
                "customer_name": a.customer_name,
                "street": a.street,
                "city": a.city,
                "region": a.region,
                "postal_code": a.postal_code,
                "iso_country": a.iso_country,
                "validated": a.validated,
            }
            for a in addresses
        ]


async def get_available_number_types(country_code: str) -> dict:
    """Get available phone number types for a country.
    
    Args:
        country_code: ISO country code (e.g., "US", "JP", "GB")
        
    Returns:
        Dictionary with subresource URIs for available number types
    """
    async with AsyncTwilioHttpClient() as http_client:
        client = Client(
            TWILIO_ACCOUNT_SID,
            TWILIO_AUTH_TOKEN,
            http_client=http_client,
        )
        try:
            country_info = await client.available_phone_numbers(country_code).fetch_async()
            return {
                "country_code": country_code,
                "subresource_uris": country_info.subresource_uris,
            }
        except Exception as e:
            return {"country_code": country_code, "error": str(e)}


async def search_available_numbers(
    country_code: str,
    number_type: str = "local",
    limit: int = 10,
    area_code: Optional[str] = None,
    contains: Optional[str] = None,
    voice_enabled: Optional[bool] = None,
    sms_enabled: Optional[bool] = None,
) -> list[dict]:
    """Search for available phone numbers.
    
    Args:
        country_code: ISO country code (e.g., "US", "JP", "GB")
        number_type: Type of number ("local", "toll_free", "mobile", "national")
        limit: Maximum number of results
        area_code: Filter by area code (US/CA only)
        contains: Filter by pattern (supports wildcards *)
        voice_enabled: Filter by voice capability
        sms_enabled: Filter by SMS capability
        
    Returns:
        List of available phone numbers with details
    """
    async with AsyncTwilioHttpClient() as http_client:
        client = Client(
            TWILIO_ACCOUNT_SID,
            TWILIO_AUTH_TOKEN,
            http_client=http_client,
        )
        
        # Build search parameters
        params = {"limit": limit}
        if area_code:
            params["area_code"] = area_code
        if contains:
            params["contains"] = contains
        if voice_enabled is not None:
            params["voice_enabled"] = voice_enabled
        if sms_enabled is not None:
            params["sms_enabled"] = sms_enabled
        
        # Get the appropriate subresource
        subresource = getattr(
            client.available_phone_numbers(country_code),
            number_type,
        )
        numbers = await subresource.list_async(**params)
        
        return [
            {
                "phone_number": n.phone_number,
                "friendly_name": n.friendly_name,
                "locality": getattr(n, "locality", None),
                "region": getattr(n, "region", None),
                "capabilities": n.capabilities,
            }
            for n in numbers
        ]


async def purchase_phone_number(
    phone_number: str,
    bundle_sid: Optional[str] = None,
    address_sid: Optional[str] = None,
    friendly_name: Optional[str] = None,
    voice_url: Optional[str] = None,
    sms_url: Optional[str] = None,
) -> dict:
    """Purchase a phone number.
    
    Args:
        phone_number: E.164 formatted phone number to purchase
        bundle_sid: Regulatory bundle SID (required for some countries)
        address_sid: Address SID (required for some countries)
        friendly_name: Optional friendly name for the number
        voice_url: Optional webhook URL for incoming calls
        sms_url: Optional webhook URL for incoming SMS
        
    Returns:
        Purchased phone number details
        
    Raises:
        Exception: If purchase fails
    """
    async with AsyncTwilioHttpClient() as http_client:
        client = Client(
            TWILIO_ACCOUNT_SID,
            TWILIO_AUTH_TOKEN,
            http_client=http_client,
        )
        
        params = {"phone_number": phone_number}
        if bundle_sid:
            params["bundle_sid"] = bundle_sid
        if address_sid:
            params["address_sid"] = address_sid
        if friendly_name:
            params["friendly_name"] = friendly_name
        if voice_url:
            params["voice_url"] = voice_url
        if sms_url:
            params["sms_url"] = sms_url
        
        number = await client.incoming_phone_numbers.create_async(**params)
        
        return {
            "sid": number.sid,
            "phone_number": number.phone_number,
            "friendly_name": number.friendly_name,
            "capabilities": number.capabilities,
            "date_created": str(number.date_created),
        }


async def release_phone_number(phone_number_sid: str) -> bool:
    """Release (delete) a phone number.
    
    Args:
        phone_number_sid: The SID of the phone number to release
        
    Returns:
        True if successful
    """
    async with AsyncTwilioHttpClient() as http_client:
        client = Client(
            TWILIO_ACCOUNT_SID,
            TWILIO_AUTH_TOKEN,
            http_client=http_client,
        )
        
        await client.incoming_phone_numbers(phone_number_sid).delete_async()
        return True


# Example usage
async def main():
    # List bundles
    print("=== Regulatory Bundles ===")
    bundles = await list_regulatory_bundles()
    for b in bundles:
        print(f"{b['friendly_name']}: {b['status']}")
    
    # Search US numbers
    print("\n=== Available US Numbers ===")
    us_numbers = await search_available_numbers("US", "local", limit=3)
    for n in us_numbers:
        print(f"{n['phone_number']} - {n['capabilities']}")
    
    # Purchase example (uncomment to execute)
    # if us_numbers:
    #     purchased = await purchase_phone_number(us_numbers[0]["phone_number"])
    #     print(f"\nPurchased: {purchased['phone_number']} (SID: {purchased['sid']})")


if __name__ == "__main__":
    asyncio.run(main())
n())
