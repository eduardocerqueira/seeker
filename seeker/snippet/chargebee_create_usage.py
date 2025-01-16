#date: 2025-01-16T17:06:15Z
#url: https://api.github.com/gists/06e6079590f7885f1328eff96f02baa9
#owner: https://api.github.com/users/abnersunwise

def create_usage(subscription_id, quantity=1, note="Default usage note"):
    """
    Creates a usage record in Chargebee, ensuring that the `usage_date` is
    properly converted to a UTC timestamp before being sent.

    Parameters:
    - subscription: The subscription object containing the company details.
    - quantity (int): The usage quantity (default is 1).
    - note (str): A note describing the usage (default is "Default usage note").

    Returns:
    - usage: The created usage object from Chargebee.
    """

    # Get the user's timezone from the subscription data
    # Example: "America/Merida"
    example_timezone = "America/Merida"
    local_tz = pytz.timezone(example_timezone)  

    # Get the current local time in the user's timezone
    local_time = datetime.now(local_tz)

    # Convert local time to UTC
    utc_time = local_time.astimezone(timezone.utc)

    # Convert UTC datetime to UNIX timestamp
    usage_date = int(utc_time.timestamp())

    # Print logs for debugging purposes
    print(f"Local time ({example_timezone}):", local_time.strftime('%Y-%m-%d %H:%M:%S %Z%z'))
    print("Converted UTC time:", utc_time.strftime('%Y-%m-%d %H:%M:%S UTC+0000'))
    print("UNIX timestamp (UTC) for API:", usage_date)

    # Build the payload for Chargebee API request
    payload = {
        "item_price_id": "Uso_Adicional_v5-USD-Monthly",  # Adjust if needed
        "usage_date": usage_date,  # Always in UTC timestamp
        "note": note,
        "quantity": quantity,
    }

    # Send the request to Chargebee API
    result = chargebee.Usage.create(subscription_id, payload)

    return result.usage