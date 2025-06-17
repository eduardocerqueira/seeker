#date: 2025-06-17T16:53:24Z
#url: https://api.github.com/gists/e7eda3fa479c57eea118bc11803511d5
#owner: https://api.github.com/users/tn3w

#!/usr/bin/env python3
import re


def reformat_dict_or_list(content: str) -> str:
    """Reformat the file to have compact dictionary and list declarations"""
    pattern = r"([A-Za-z0-9_]+(?:\s*:\s*[A-Za-z\[\]]+(?:\[[A-Za-z0-9_, ]+\])?)?)\s*=\s*[{[].*?[}\]]\s*(?:$|(?=\n\n))"
    declarations = re.finditer(pattern, content, re.DOTALL)

    result = content
    for decl in declarations:
        original = decl.group(0)

        if '"""' in original or "'''" in original:
            continue

        name_part = decl.group(1).strip()

        if "=" not in original:
            continue

        data_part = original.split("=", 1)[1].strip()

        if not (data_part.startswith("{") or data_part.startswith("[")):
            continue

        is_dict = data_part.startswith("{")

        inner_content = data_part[1:-1].strip()
        if not inner_content:
            continue

        items = []
        current_item = ""
        brace_count = 0
        bracket_count = 0
        in_string = False
        escape_next = False
        string_delim = None

        for char in inner_content:
            if escape_next:
                escape_next = False
                current_item += char
                continue

            if char == "\\":
                escape_next = True
                current_item += char
                continue

            if char in ('"', "'") and (string_delim is None or char == string_delim):
                in_string = not in_string
                if in_string:
                    string_delim = char
                else:
                    string_delim = None
                current_item += char
                continue

            if in_string:
                current_item += char
                continue

            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
            elif char == "[":
                bracket_count += 1
            elif char == "]":
                bracket_count -= 1

            if char == "," and brace_count == 0 and bracket_count == 0:
                items.append(current_item.strip())
                current_item = ""
            else:
                current_item += char

        if current_item.strip():
            items.append(current_item.strip())

        if is_dict:
            clean_items = []
            for item in items:
                clean_items.append((item, None))
        else:
            clean_items = []
            for item in items:
                clean_items.append((item, None))

        formatted_items = []
        current_line = []
        current_line_len = 0
        indent = "    "

        for item_tuple in clean_items:
            item, comment = item_tuple

            if comment:
                if current_line:
                    formatted_items.append(", ".join(current_line))
                    current_line = []
                    current_line_len = 0
                formatted_items.append(f"{item} {comment}")
                continue

            item_len = len(item) + 2

            if current_line and (current_line_len + item_len >= 76):
                formatted_items.append(", ".join(current_line))
                current_line = [item]
                current_line_len = len(item)
            else:
                current_line.append(item)
                current_line_len += item_len

        if current_line:
            formatted_items.append(", ".join(current_line))

        open_bracket = "{" if is_dict else "["
        close_bracket = "}" if is_dict else "]"

        new_format = f"{name_part} = {open_bracket}\n"
        for item in formatted_items:
            new_format += f"{indent}{item},\n"
        new_format += f"{close_bracket}"

        result = result.replace(original, new_format)

    return result


def main():
    """Reformat the file to have compact dictionary and list declarations"""
    file_path = "main.py"

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    new_content = reformat_dict_or_list(content)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"Successfully reformatted {file_path}")


if __name__ == "__main__":
    main()
