#date: 2023-06-13T17:09:33Z
#url: https://api.github.com/gists/3e4afc0a5425fe664cea398c9f8f14c1
#owner: https://api.github.com/users/ninjasitm

import stripe

# Set your Stripe API keys
live_mode_api_key = ""
test_mode_api_key = ""

# Initialize Stripe API with your live mode API key
stripe.api_key = live_mode_api_key

def copy_products_to_test_mode():
    # Retrieve all products in live mode
    products = stripe.Product.list(limit=100)  # Adjust the limit as per your requirement

    # Create the same products in test mode
    for product in products:
        new_product = stripe.Product.create(
            name=product.name,
            description=product.description,
            images=product.images,
            metadata=product.metadata,
            api_key=test_mode_api_key
        )

        # Retrieve prices associated with the live mode product
        prices = stripe.Price.list(product=product.id)

        # Create prices for the corresponding test mode product
        for price in prices:
            stripe.Price.create(
                recurring=price.recurring,
                active=price.active,
                unit_amount=price.unit_amount,
                currency=price.currency,
                product=new_product.id,
                metadata=price.metadata,
                api_key=test_mode_api_key
            )

        print(f"Product '{product.name}' copied to test mode.")

    print("All products copied to test mode successfully.")

# Call the function to initiate the copying process
copy_products_to_test_mode()
