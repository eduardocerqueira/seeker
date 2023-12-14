#date: 2023-12-14T16:54:28Z
#url: https://api.github.com/gists/c0530f2a832463391af118ee4b766bae
#owner: https://api.github.com/users/terminalcommandnewsletter

token = "**********"

import os, time, requests

emojis = {"code": ":large_blue_circle:", "firefox": ":fox_face:", "terminator": ":red_circle:"} # Add some more here
def get_emoji(window_name):
    if window_name.lower() in list(emojis.keys()): return emojis[window_name.lower()]
    return ":question:"

 "**********"i "**********"f "**********"  "**********"" "**********"A "**********"d "**********"d "**********"  "**********"y "**********"o "**********"u "**********"r "**********"  "**********"o "**********"w "**********"n "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"  "**********"h "**********"e "**********"r "**********"e "**********"" "**********"  "**********"i "**********"n "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
    print("Replace the token variable on line 1 with your token!")
    exit(1)

while True:
    window_name = os.popen("gdbus call -e -d org.gnome.Shell -o /org/gnome/Shell -m org.gnome.Shell.Eval global.get_window_actors\(\)[`gdbus call -e -d org.gnome.Shell -o /org/gnome/Shell -m org.gnome.Shell.Eval global.get_window_actors\(\).findIndex\(a\=\>a.meta_window.has_focus\(\)===true\) | cut -d\"'\" -f 2`].get_meta_window\(\).get_wm_class\(\) | cut -d'\"' -f 2").read().strip() # Taken from https://gist.github.com/rbreaves/257c3edfa301786e66e964d7ac036269

    graphql_query = """
    mutation {
    changeUserStatus(input: { emoji: \"""" + get_emoji(window_name) + """", message: "Using """ + window_name + """!" }) {
        status {
        emoji
        message
        }
    }
    }
    """

    api_url = "https://api.github.com/graphql"

    headers = {
        "Authorization": "**********"
        "Content-Type": "application/json",
    }

    payload = {"query": graphql_query}

    response = requests.post(api_url, headers=headers, json=payload)

    print("DEBUG: " + response.text)

    time.sleep(60)