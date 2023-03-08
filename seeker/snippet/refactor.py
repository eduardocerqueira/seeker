#date: 2023-03-08T16:51:54Z
#url: https://api.github.com/gists/230863f714ab43fc92ad55510de2ca4e
#owner: https://api.github.com/users/tito

"""
Refactoring tool using ChatGPT from Vue 2 to Vue 3

$ export OPENAPI_APIKEY=sk.........
$ python refactor.py MyView.vue
"""

import os
import re
import sys

import openai


REFACTOR_PROMPT = """
You are an assistant design to help developper for migrating their code from Vue 2 to Vue 3 using Typescript with Composition API. Here is a set of rules you must absolutely follow:

1. Rewrite the <script lang="ts"> to <script setup lang="ts">
2. The content of the script tag must be a valid Typescript code
3. The component must be flattened into the script setup
4. Remove any "export default".
5. Use the `onMounted` hook instead of the `created` lifecycle hook if necessary
6. Use the `useRoute` approach instead of $route. Same for $router.
7. Store is not using vuex but pinia.
8. Auth related function is accessible in stores/auth.ts, using useAuthStore.
9. Do not use Ref is the type can be infered from the value pass into ref()
10. Do not put all the methods and properties into a global const object
11. Prefer using global "const router = useRouter()" instead of live instanciation when needed
"""

RE_SCRIPT = re.compile(r"(<script lang=\"ts\">.*</script>)", re.DOTALL)


def refactor(filename, model):
    with open(filename, "r", encoding="utf8") as f:
        content = f.read()

    # extract the script tag from <script lang="ts"> to </script>
    match = RE_SCRIPT.search(content)
    if not match:
        print("ERR: No script tag found")
        sys.exit(1)
    spanstart, spanend = match.span()

    # print the start
    print(content[:spanstart])

    # ask for refactoring
    response = openai.ChatCompletion.create(
        model=model,
        stream=True,
        temperature=0,
        messages=[
            {"role": "system", "content": REFACTOR_PROMPT},
            {"role": "user", "content": content[spanstart:spanend]},
        ]
    )

    # get the refactored script
    for entry in response:
        choice = entry["choices"][0]
        if choice["finish_reason"] == "stop":
            break

        if choice["finish_reason"] is not None:
            print("ERR: Unexpected finish_reason", choice["finish_reason"])
            sys.exit(1)

        delta_content = choice["delta"].get("content")
        if delta_content is not None:
            print(delta_content, end="")

    # replace the script tag
    print(content[spanend:])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("file", type=str)
    args = parser.parse_args()

    openai.api_key = os.environ["OPENAI_APIKEY"]
    refactor(args.file, args.model)