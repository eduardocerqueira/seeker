#date: 2025-04-24T17:01:03Z
#url: https://api.github.com/gists/58881c19542307df7d08f758e6ec28b9
#owner: https://api.github.com/users/InformedChoice

"""
Stripe configuration settings.

This module defines Stripe-specific settings and helpers for the Lyndsy application.
"""

import os

# Stripe API keys
STRIPE_PUBLISHABLE_KEY = os.environ.get("STRIPE_PUBLISHABLE_KEY", "dummy_publishable_key")
STRIPE_SECRET_KEY = "**********"
STRIPE_WEBHOOK_SECRET = "**********"

# Stripe product and price IDs
STRIPE_PRICE_ID = os.environ.get("STRIPE_PRICE_ID", "dummy_price_id")
STRIPE_CLINICIAN_PRICE_ID = os.environ.get("STRIPE_CLINICIAN_PRICE_ID", "dummy_clinician_price_id")
STRIPE_PRODUCT_ID = os.environ.get("STRIPE_PRODUCT_ID", "dummy_product_id")

# Payment method settings
STRIPE_ENABLE_APPLE_PAY = os.environ.get("STRIPE_ENABLE_APPLE_PAY", "True").lower() in ('true', 't', '1')
STRIPE_ENABLE_GOOGLE_PAY = os.environ.get("STRIPE_ENABLE_GOOGLE_PAY", "True").lower() in ('true', 't', '1')
STRIPE_ENABLE_LINK = os.environ.get("STRIPE_ENABLE_LINK", "True").lower() in ('true', 't', '1')
STRIPE_ENABLE_ACH = os.environ.get("STRIPE_ENABLE_ACH", "True").lower() in ('true', 't', '1')
STRIPE_ENABLE_BNPL = os.environ.get("STRIPE_ENABLE_BNPL", "True").lower() in ('true', 't', '1')

# Default payment method configuration
DEFAULT_PAYMENT_METHODS = ["card"]

if STRIPE_ENABLE_ACH:
    DEFAULT_PAYMENT_METHODS.append("us_bank_account")

if STRIPE_ENABLE_BNPL:
    DEFAULT_PAYMENT_METHODS.extend(["affirm", "afterpay_clearpay", "klarna"])

# Wallet payment methods are typically dynamically enabled by Stripe
# based on the browser and device, but we include them in our allowed list
WALLET_PAYMENT_METHODS = []

if STRIPE_ENABLE_LINK:
    WALLET_PAYMENT_METHODS.append("link")
    
# Apple Pay and Google Pay are presented by Stripe when available in the browser/device
# They're not explicitly included in DEFAULT_PAYMENT_METHODS because they're
# handled differently by Stripe's Payment Element
 in DEFAULT_PAYMENT_METHODS because they're
# handled differently by Stripe's Payment Element
