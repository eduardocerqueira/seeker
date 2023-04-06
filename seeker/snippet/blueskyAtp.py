#date: 2023-04-06T16:43:58Z
#url: https://api.github.com/gists/0ab19e6d53bbb360d5aa253e27fa7692
#owner: https://api.github.com/users/hsgw

# MIT License (c) 2023, @hsgw

import requests
import datetime
import re


class BlueskyAtpException(Exception):
    """
    Custom exception class for Bluesky ATP.

    Attributes:
        status_code (int): HTTP status code associated with the exception.
        text (str): Error message text associated with the exception.
    """

    def __init__(self, status_code, text):
        """
        Constructor for BlueskyAtpException.

        Parameters:
            status_code (int): HTTP status code associated with the exception.
            text (str): Error message text associated with the exception.
        """
        self.status_code = status_code
        self.text = text
        super().__init__(f"{status_code}: {text}")


class BlueskyAtp:
    def __init__(self, server="https://bsky.social"):
        """
        Initialize a BlueskyAtp object with the given server URL.

        Args:
        - server: (str) The URL of the BlueskyAtp server.
        """
        self.__handle = {}
        self.__password = "**********"
        self.__auth = {}
        if server.endswith("/"):
            server = server[:-1]
        self.__serverUri = f"{server}/xrpc/"

 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"l "**********"o "**********"g "**********"i "**********"n "**********"( "**********"s "**********"e "**********"l "**********"f "**********", "**********"  "**********"h "**********"a "**********"n "**********"d "**********"l "**********"e "**********", "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********") "**********": "**********"
        """
        Authenticate the user with the given handle and password.

        Args:
        - handle: (str) The handle of the user.
        - password: "**********"

        Raises:
        - BlueskyAtpException: If the login fails.
        """
        res = requests.post(
            self.__makeUri("com.atproto.server.createSession"),
            json={"handle": "**********": password},
        )
        if res.ok:
            self.__auth = res.json()
        else:
            raise BlueskyAtpException(res.status_code, res.text)

    def getTimeline(self, limit=50):
        """
        Get the user's timeline.

        Args:
        - limit: (int) The number of items to retrieve.

        Returns:
        - A JSON object representing the timeline.

        Raises:
        - BlueskyAtpException: If the request fails.
        """
        params = {"limit": limit}
        res = requests.get(
            self.__makeUri("app.bsky.feed.getTimeline"),
            headers=self.__makeHeader(),
            params=params,
        )
        if res.ok:
            return res.json()
        else:
            raise BlueskyAtpException(res.status_code, res.text)

    def getProfile(self, handle):
        """
        Get the user's profile.

        Args:
        - handle: (str) The handle of the user.

        Returns:
        - A JSON object representing the user's profile.

        Raises:
        - BlueskyAtpException: If the request fails.
        """
        if handle.startswith("@"):
            handle = handle[1:]
        headers = {"Authorization": f"Bearer {self.__auth['accessJwt']}"}
        params = {"actor": handle}
        res = requests.get(
            self.__makeUri("app.bsky.actor.getProfile"),
            headers=headers,
            params=params,
        )
        if res.ok:
            return res.json()
        else:
            raise BlueskyAtpException(res.status_code, res.text)

    def post(self, text):
        """
        This method creates a new post record with the given text message and mentions
        Args:
            text: A string representing the text message to post
        Raises:
            BlueskyAtpException: If the post request fails
        """
        data = {
            "did": self.__auth.get("did"),
            "repo": self.__auth.get("handle"),
            "collection": "app.bsky.feed.post",
            "record": {
                "text": text,
                "createdAt": self.__getCurrntDate(),
                "$type": "app.bsky.feed.post",
            },
        }

        mentions = self.__searchMentions(text)

        if len(mentions) != 0:
            data["record"]["facets"] = []
            for mention in mentions:
                profile = self.__getProfile(mention["handle"])
                if profile is not None:
                    data["record"]["facets"].append(
                        {
                            "$type": "app.bsky.richtext.facet",
                            "index": {
                                "byteStart": mention["start"],
                                "byteEnd": mention["end"],
                            },
                            "features": [
                                {
                                    "did": profile.get("did"),
                                    "$type": "app.bsky.richtext.facet#mention",
                                }
                            ],
                        }
                    )

        res = requests.post(
            self.__makeUri("com.atproto.repo.createRecord"),
            headers=self.__makeHeader(),
            json=data,
        )

        if not res.ok:
            raise BlueskyAtpException(res.status_code, res.text)

    def getNotifyCount(self):
        """
        This method retrieves the count of unread notifications
        Returns:
            The count of unread notifications
        Raises:
            BlueskyAtpException: If the get request fails
        """
        res = requests.get(
            self.__makeUri("app.bsky.notification.getUnreadCount"),
            headers=self.__makeHeader(),
        )
        if res.ok:
            return res.json()["count"]
        else:
            raise BlueskyAtpException(res.status_code, res.text)

    def getNotifylist(self, limit=10):
        """
        This method retrieves the list of unread notifications with the given limit
        Args:
            limit: An integer representing the limit of notifications to retrieve
        Returns:
            A list of unread notifications
        Raises:
            BlueskyAtpException: If the get request fails
        """
        params = {"limit": limit}
        res = requests.get(
            self.__makeUri("app.bsky.notification.listNotifications"),
            headers=self.__makeHeader(),
            params=params,
        )

        if res.ok:
            return res.json()
        else:
            raise BlueskyAtpException(res.status_code, res.text)

    def setNotifySeen(self, date):
        """
        This method updates the timestamp of the notifications that have been seen
        Args:
            date: A string representing the timestamp of the notifications that have been seen
        Raises:
            BlueskyAtpException: If the post request fails
        """
        data = {"seenAt": date}
        res = requests.post(
            self.__makeUri("app.bsky.notification.updateSeen"),
            headers=self.__makeHeader(),
            json=data,
        )
        if not res.ok:
            raise BlueskyAtpException(res.status_code, res.text)

    def pollNotify(self):
        """
        This method retrieves the list of unread notifications and updates the timestamp of the notifications that have been seen
        Returns:
            A list of unread notifications
        Raises:
            BlueskyAtpException: If the get request or the post request fails
        """
        count = self.getNotifyCount()
        if count != 0:
            currntDate = self.__getCurrntDate()
            ret = self.getNotifylist(count)
            self.setNotifySeen(currntDate)
            return ret
        return None

    def __makeUri(self, uri):
        return f"{self.__serverUri}{uri}"

    def __makeHeader(self, header={}):
        header["Authorization"] = f"Bearer {self.__auth['accessJwt']}"
        return header

    def __getCurrntDate(self):
        return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    def __searchMentions(self, text):
        regex = r"@([a-zA-Z0-9_]+)\.bsky\.social"
        matches = re.finditer(regex, text)
        results = []
        for match in matches:
            result = {
                "handle": match.group(0),
                "start": match.start(),
                "end": match.end(),
            }
            results.append(result)
        return results