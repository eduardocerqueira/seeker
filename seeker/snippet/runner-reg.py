#date: 2024-02-21T17:01:06Z
#url: https://api.github.com/gists/feefd7eccb5b8ad9b391f6535fc4c537
#owner: https://api.github.com/users/mvandermeulen

#!/usr/bin/env python

from github import Github, GithubIntegration, Consts, Installation, GithubObject
import github.Organization
import re
import requests

my_appid = 123
my_keyfile = "private-key.pem"
my_org = 'myorg'

# stupid pygithub
def get_org_installation(self, org):
    """
    :calls: `GET /orgs/{org}/installation <https://docs.github.com/en/rest/reference/apps#get-an-organization-installation-for-the-authenticated-app>`_
    :param org: str
    :rtype: :class:`github.Installation.Installation`
    """
    headers = {
        "Authorization": f"Bearer {self.create_jwt()}",
        "Accept": Consts.mediaTypeIntegrationPreview,
        "User-Agent": "PyGithub/Python",
    }

    response = requests.get(
        f"{self.base_url}/orgs/{org}/installation",
        headers=headers,
    )
    response_dict = response.json()
    return Installation.Installation(None, headers, response_dict, True)

GithubIntegration.get_org_installation = get_org_installation

 "**********"c "**********"l "**********"a "**********"s "**********"s "**********"  "**********"R "**********"u "**********"n "**********"n "**********"e "**********"r "**********"T "**********"o "**********"k "**********"e "**********"n "**********"( "**********"g "**********"i "**********"t "**********"h "**********"u "**********"b "**********". "**********"G "**********"i "**********"t "**********"h "**********"u "**********"b "**********"O "**********"b "**********"j "**********"e "**********"c "**********"t "**********". "**********"N "**********"o "**********"n "**********"C "**********"o "**********"m "**********"p "**********"l "**********"e "**********"t "**********"a "**********"b "**********"l "**********"e "**********"G "**********"i "**********"t "**********"h "**********"u "**********"b "**********"O "**********"b "**********"j "**********"e "**********"c "**********"t "**********") "**********": "**********"
    def __repr__(self):
        return self.get__repr__({"expires_at": self._expires_at.value})

    @property
 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"( "**********"s "**********"e "**********"l "**********"f "**********") "**********": "**********"
        """
        :type: string
        """
        return self._token.value

    @property
    def expires_at(self):
        """
        :type: datetime
        """
        return self._expires_at.value

    def _initAttributes(self):
        self._token = "**********"
        self._expires_at = github.GithubObject.NotSet

    def _useAttributes(self, attributes):
        if "token" in attributes: "**********"
            self._token = "**********"
        if "expires_at" in attributes:  # pragma no branch
            self._expires_at = self._makeDatetimeAttribute(
                re.sub(r'\.\d{3}Z$', 'Z', attributes["expires_at"])
            )


 "**********"d "**********"e "**********"f "**********"  "**********"g "**********"e "**********"t "**********"_ "**********"s "**********"e "**********"l "**********"f "**********"_ "**********"h "**********"o "**********"s "**********"t "**********"e "**********"d "**********"_ "**********"r "**********"u "**********"n "**********"n "**********"e "**********"r "**********"_ "**********"r "**********"e "**********"g "**********"i "**********"s "**********"t "**********"r "**********"a "**********"t "**********"i "**********"o "**********"n "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"( "**********"s "**********"e "**********"l "**********"f "**********") "**********": "**********"
    """
    : "**********": `POST /orgs/{owner}/actions/runners/registration-token <https://docs.github.com/en/rest/reference/actions#create-a-registration-token-for-an-organization>`_
    : "**********": :class:`RunnerToken`
    """
    headers, data = self._requester.requestJsonAndCheck(
        "POST", f"{self.url}/actions/runners/registration-token"
    )
    return RunnerToken(None, headers, data, True)

 "**********"d "**********"e "**********"f "**********"  "**********"g "**********"e "**********"t "**********"_ "**********"s "**********"e "**********"l "**********"f "**********"_ "**********"h "**********"o "**********"s "**********"t "**********"e "**********"d "**********"_ "**********"r "**********"u "**********"n "**********"n "**********"e "**********"r "**********"_ "**********"r "**********"e "**********"m "**********"o "**********"v "**********"e "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"( "**********"s "**********"e "**********"l "**********"f "**********") "**********": "**********"
    """
    : "**********": `POST /orgs/{owner}/actions/runners/remove-token <https://docs.github.com/en/rest/reference/actions#create-a-remove-token-for-an-organization>`_
    : "**********": :class:`RunnerToken`
    """
    headers, data = self._requester.requestJsonAndCheck(
        "POST", f"{self.url}/actions/runners/remove-token"
    )
    return RunnerToken(None, headers, data, True)

github.Organization.Organization.get_self_hosted_runner_registration_token = "**********"
github.Organization.Organization.get_self_hosted_runner_remove_token = "**********"


with open(my_keyfile, "r") as keyfile:
    key = keyfile.read()

integration = GithubIntegration(my_appid, key)
my_installation_id = integration.get_org_installation(my_org).id
installation_token = "**********"
print(installation_token.expires_at)
gh = "**********"=installation_token.token)
org = gh.get_organization(my_org)
reg_token = "**********"
print(reg_token.expires_at)
print(f"runner\\config.cmd --unattended --url {org.html_url} --token {reg_token.token} --replace")
rem_token = "**********"
print(rem_token.expires_at)
print(f"runner\\config.cmd remove --token {rem_token.token}")