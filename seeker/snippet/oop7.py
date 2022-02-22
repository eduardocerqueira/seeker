#date: 2022-02-22T16:59:10Z
#url: https://api.github.com/gists/0391f47297cb820d8aa723452a7b02b8
#owner: https://api.github.com/users/jimmy-law

    # make API call - handle any http errors
    def call(self, method, params, payload_name):
        request_url = SportsDB.base_url + method + self.__make_query_string(params)
        response = requests.get(request_url)
        if response.status_code == 200:
            payload = response.json().get(payload_name, [])
        elif response.status_code == 429:
            # for this demo we simply catch it here - we could easily implement re-try after some specified time
            payload = "throttled by SportsDB"
        else:
            payload = "other error"
        return payload