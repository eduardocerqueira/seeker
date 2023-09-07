#date: 2023-09-07T16:57:14Z
#url: https://api.github.com/gists/92501d3d9f37283c274e70f5b65420e3
#owner: https://api.github.com/users/brendandagys

def set_failed_sign_orders_to_pending():
    """Set's existing sign orders that failed due to lacking S3 permissions, to status='pending'."""
    from rentalsbase.rentalscore.models import PropertySignRequest

    failed_sign_orders = PropertySignRequest.objects.filter(status="S3UploadFailedError - bad creds or permissions")
    for sign_order in failed_sign_orders:
        sign_order.status = "pending"
        sign_order.save()