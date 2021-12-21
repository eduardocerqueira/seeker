#date: 2021-12-21T17:00:55Z
#url: https://api.github.com/gists/11275272b67761b0f54d6a16da7737df
#owner: https://api.github.com/users/dev-gaster

# this am implementing with graphene_file_upload,
# You can do the same using with GraphqlView
from graphene_file_upload.django import FileUploadGraphQLView

# Here we write our funtion that checks the auth header on graphql requests and verifies the user accordingly
# This function can be used as a wrapper around the main graphql url of the project
class FBTokenAuthenticationView(FileUploadGraphQLView):
    def dispatch(self, request, *args, **kwargs):
        auth_headers = request.META.get('HTTP_AUTHORIZATION')
        if not auth_headers:
            return HttpResponse('Authorization failure', status=401)
        id_token = auth_headers.split(" ").pop()
        decoded_token = None
        try:
            decoded_token = auth.verify_id_token(id_token)
        except Exception:
            return HttpResponse("Invalid Auth Token", status=401)
        if not id_token or not decoded_token:
            return HttpResponse("Authorization error", status=401)
        try:
            uid = decoded_token.get("uid")
        except Exception:
            return HttpResponse("Token key error", status=401)
        user, created = User.objects.get_or_create(uuid=uid, defaults=None)
        if created:
            fire_user = auth.get_user(uid)
            if fire_user.display_name:
                user.account_name = fire_user.display_name
                name_ar = fire_user.display_name.split(" ",1)
                user.first_name = name_ar[0]
                sugUsername = name_ar[0]+name_ar[1]
                if not User.objects.filter(username=sugUsername).exists():
                    user.username = sugUsername
                user.last_name = name_ar[1] if len(name_ar) > 1 else ""
            user.email_verified = fire_user.email_verified
            if fire_user.photo_url:
                user.profile_picture = fire_user.photo_url
            if fire_user.email:
                user.email = fire_user.email
            user.save()
        request.user = user
        return super().dispatch(request, *args, **kwargs)
      