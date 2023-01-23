#date: 2023-01-23T17:00:27Z
#url: https://api.github.com/gists/97ea085358e7ee663e1afa430fe0d979
#owner: https://api.github.com/users/mikeckennedy

# Short usage example in form post handler on the server:

def form_post_handler():
    turnstile_response = form_dict.get('cf-turnstile-response')
    validation_response = turnstile.validate(turnstile_response, get_client_ip())
    if not validation_response.success:
        # Handle the error
        ...

    # All is good from here out...
