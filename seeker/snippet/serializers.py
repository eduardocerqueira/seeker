#date: 2023-03-23T17:06:11Z
#url: https://api.github.com/gists/4fd8f3a5b58d2fd71b79c3189f3720ae
#owner: https://api.github.com/users/Lord-sarcastic

from rest_framework_simplejwt.serializers import TokenObtainPairSerializer

from .models import User


 "**********"c "**********"l "**********"a "**********"s "**********"s "**********"  "**********"L "**********"o "**********"g "**********"i "**********"n "**********"S "**********"e "**********"r "**********"i "**********"a "**********"l "**********"i "**********"z "**********"e "**********"r "**********"( "**********"T "**********"o "**********"k "**********"e "**********"n "**********"O "**********"b "**********"t "**********"a "**********"i "**********"n "**********"P "**********"a "**********"i "**********"r "**********"S "**********"e "**********"r "**********"i "**********"a "**********"l "**********"i "**********"z "**********"e "**********"r "**********") "**********": "**********"
    user_info = serializers.SerializerMethodField()

    @classmethod
    def get_token(cls, user: "**********":
        token = "**********"

        # Add custom claims
        token["user_info"] = "**********"
            "uuid": str(user.uuid),
            "email": user.email,
            "is_verified": user.is_verified,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "is_staff": user.is_staff,
            "is_superuser": user.is_superuser,
            "is_active": user.is_active,
        }

        return token