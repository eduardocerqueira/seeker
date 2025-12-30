#date: 2025-12-30T17:00:47Z
#url: https://api.github.com/gists/6a9b40c568cf3ebe49beb25960c6147a
#owner: https://api.github.com/users/JonathanArns

from kitty.boss import boss
from kittens.tui.handler import result_handler


def main(args: list[str]) -> str:
    pass

@result_handler(no_ui=true)
def handle_result(args: list[str], stdin_data: str, target_window_id: int, boss: boss) -> none:
    window = args[1]

    w = boss.window_id_map.get(int(window))
    if w is none:
        return "unknown_window"

    if w.screen.is_main_linebuf():
        return "linebuf"
    else:
        return "application"