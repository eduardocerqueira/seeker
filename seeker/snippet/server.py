#date: 2024-06-13T17:04:08Z
#url: https://api.github.com/gists/fb15d4358f2fa39cab660fa0d6c2733b
#owner: https://api.github.com/users/engineersamuel

from pirate_speak.chain import chain as pirate_speak_chain
 
add_routes(
    app,
    pirate_speak_chain,
    path="/pirate-speak",
    playground_type="chat",
)
