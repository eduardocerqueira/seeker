#date: 2022-12-27T16:41:45Z
#url: https://api.github.com/gists/292a3ce49b9fee39185309022476cf70
#owner: https://api.github.com/users/ccaiccie

"""
A script that loads a GitHub repositories info and updates a Firebase database with it.

You will require the following database rules:

{
    "rules": {        
        "repositories": {
          ".write": "auth != null",
          ".read": true
        }
    }
}

"""

import requests

# Firebase Constants
# You can create a new username with password in the Auth section from the Firebase console.
FIREBASE_EMAIL = ""
FIREBASE_PASSWORD = "**********"

FIREBASE_APIKEY = ""
FIREBASE_PROJECTID = ""  # Example: my-project-123456
FIREBASE_NODE = "repositories"  # Must be the same as the one in the rules.

GITHUB_ACCOUNT = "PhantomAppDevelopment"
ACCOUNT_TYPE = "orgs"  # Can be "orgs" or "users"


def firebase_login():
    """Logins the user to your Firebase database."""

    base_url = "https: "**********"
        FIREBASE_APIKEY)

    credentials = dict()
    credentials["email"] = FIREBASE_EMAIL
    credentials["password"] = "**********"
    credentials["returnSecureToken"] = "**********"

    with requests.post(base_url, json=credentials) as login_response:

        # IF the login was successful we return the 'idToken', else we exit the program.
        if login_response.status_code == 200:
            return login_response.json()["idToken"]
        else:
            print(login_response.json()["error"]["message"])
            quit()


def load_profile_repositories():
    """Loads the profile repositories and converts the relevant data into a dictionary."""

    auth_token = "**********"

    base_url = "https://api.github.com/{0}/{1}/repos".format(
        ACCOUNT_TYPE, GITHUB_ACCOUNT)

    with requests.get(base_url) as profile_contents:

        for item in profile_contents.json():

            repo_data = dict()
            repo_data["id"] = item["id"]
            repo_data["name"] = item["name"]
            repo_data["url"] = item["url"]
            repo_data["stars"] = item["stargazers_count"]
            repo_data["forks"] = item["forks_count"]
            repo_data["watchers"] = item["watchers_count"]
            repo_data["issues"] = item["open_issues_count"]

            save_to_firebase(repo_data, auth_token)


 "**********"d "**********"e "**********"f "**********"  "**********"s "**********"a "**********"v "**********"e "**********"_ "**********"t "**********"o "**********"_ "**********"f "**********"i "**********"r "**********"e "**********"b "**********"a "**********"s "**********"e "**********"( "**********"r "**********"e "**********"p "**********"o "**********"_ "**********"d "**********"a "**********"t "**********"a "**********", "**********"  "**********"a "**********"u "**********"t "**********"h "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********") "**********": "**********"
    """
    Saves the repository data to the specified Firebase node.
    We use the repo name as its id for the Firebase database.
    """

    base_url = "https://{0}.firebaseio.com/{1}/{2}.json?auth={3}".format(
        FIREBASE_PROJECTID, FIREBASE_NODE, repo_data["name"], auth_token)

    # We use the PATCH verb and send the data as a JSON string.
    with requests.patch(base_url, json=repo_data) as response:

        # We now determine the status of the operation.
        if response.status_code == 200:
            print("Successfully updated: {0}".format(repo_data["name"]))
        else:
            print(response.json()["error"])


if __name__ == "__main__":
    load_profile_repositories()
