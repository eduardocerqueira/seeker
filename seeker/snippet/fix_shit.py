#date: 2022-07-27T17:18:23Z
#url: https://api.github.com/gists/836851bc205255fc457f2f1e8aeeaf81
#owner: https://api.github.com/users/pbsds

#!/usr/bin/env python
import re
import os
import sys
from pathlib import Path
import subprocess
import shutil
import concurrent.futures as futures
import requests
import json
from typing import Iterable, Hashable, Callable, Any, Dict

assert shutil.which("diff"), "'diff' not found in PATH"

VAULT = Path(__file__).parent.resolve() # im in root
CONCURRENT_JSON_RPC_CALLS = 4
BIBLIOGRAPHY_FNAME: Path = None

CITATION_PLUGIN_DATA = VAULT / ".obsidian" / "plugins" / "obsidian-citation-plugin" / "data.json"
if CITATION_PLUGIN_DATA.is_file():
    with CITATION_PLUGIN_DATA.open() as f:
        BIBLIOGRAPHY_FNAME = Path(json.load(f).get("citationExportPath"))

# === helpers:

# https://github.com/retorquere/zotero-better-bibtex/blob/2c1861cf92c7a957ee2f98fe6337d7c6b451fe72/content/json-rpc.ts#L195

# requires better biblatex for zotero plugin
N_ZOTERO_QUERIES = 0
IS_ZOTERO_ONLINE = True
def query_zotero_for_attachments(citekey: str):
    # https://retorque.re/zotero-better-bibtex/exporting/json-rpc/
    global N_ZOTERO_QUERIES, IS_ZOTERO_ONLINE
    if not IS_ZOTERO_ONLINE:
        return None
    N_ZOTERO_QUERIES += 1
    try:
        r = requests.post(
            "http://localhost:23119/better-bibtex/json-rpc",
            headers = {
                "Accept": "application/json",
            },
            json = {
                "jsonrpc" : "2.0",
                "method"  : "item.attachments",
                "params"  : [ citekey ],
            },
        ).json()
    except requests.ConnectionError as e:
        IS_ZOTERO_ONLINE = False
        return None

    #r["error"]             : dict
    #r["error"]["code"]     : str
    #r["error"]["message"]  : str
    #r["result"]            : list
    #r["result"][i]         : dict
    #r["result"][i]["open"] : str   "zotero://open-pdf/library/items/M4R8MQSN"
    #r["result"][i]["path"] : str

    for item in r.get("result", []):
        if item["path"] and item["path"].endswith(".pdf"):
            open_pdf = item["open"]
            select = open_pdf.replace("/open-pdf/", "/select/")
            return f"<sup>*[select]({select}) [PDF]({open_pdf})*</sup>"
    else:
        return None

def map_threaded_gather_dict(
        func        : Callable[[Hashable], Any],
        iterable    : Iterable[Hashable],
        max_workers : int   =   CONCURRENT_JSON_RPC_CALLS,
        ) -> Dict[Hashable, Any]:
    with futures.ThreadPoolExecutor(max_workers=max_workers) as e:
        #for future in futures.as_completed(
        #    e.submit(func, i)
        #    for i in iterable
        #):
        #    key, value = future.result()
        #    out[key] = value
        return {
            j : future.result()
            for j, future in [
                ( i, e.submit(func, i) )
                for i in iterable
            ]
        }


# === transformations

transformations: [callable] = []


@transformations.append
def split_ligatures(data: str, filename: Path) -> str:
    return (data
        .replace("–", "-") # TODO: also for file names
        .replace("ﬂ", "fl")
        .replace("ﬁ", "fi")
        .replace("ﬀ", "ff")
        .replace("ﬃ", "ffi")
        .replace("ﬄ", "ffl")
        .replace("→", "->")
    )



@transformations.append
def extracted_annotations__format_headers(data: str, filename: Path) -> str:
    if not filename.parent.name == "zotero": return data
    if not " Extracted Annotations " in filename.name: return data

    data = re.sub(
        r'^> (\"(?P<number>([0-9]+\.)*[0-9])\.? +(?P<title>[^\"]*)\") \(\[(?P<citation>.*)\)\)$',
        r'> ## Section \g<number>: "\g<title>"\n> ([\g<citation>))',
        data,
        flags = re.MULTILINE, # | re.DOTALL,
    )
    return data
    # used in "Neural Fields in Visual Computing and Beyond"
    data = re.sub(
        r'^> Part (?P<number>I+)\. (?P<title>.+) \(\[(?P<citation>.*)\)\)$',
        r'> # == Part \g<number>: "\g<title>" ==\n> ([\g<citation>))',
        data,
        flags = re.MULTILINE, # | re.DOTALL,
    )
    return data

@transformations.append
def extracted_annotations__boldify_definitions(data: str, filename: Path) -> str:
    if not filename.parent.name == "zotero": return data
    if not " Extracted Annotations " in filename.name: return data

    return re.sub(
        # Capitalized, max length 30, ends with ': ', can't include :.,;=
        r'^> "(?P<definition>(?=.{1,30}: )([A-Z][^:.,;=]+)):(?P<padding>[ "])',
        r'> "**\g<definition>:**\g<padding>',
        data,
        flags = re.MULTILINE, # | re.DOTALL,
    )

@transformations.append
def extracted_notes__strip_header_from_single_line_notes(data: str, filename: Path) -> str:
    if not filename.parent.name == "zotero": return data
    if not " - " in filename.name: return data

    citekey, note_fname = filename.name.split(" - ", 1)

    m = re.match(
        r'\* Mdnotes File Name: .*\n\n# (.+)$',
        data.rstrip(),
    )
    if m:
        note_file, = m.groups()
        for i in ":/": # TODO: @?
            note_file = note_file.replace(i, "")
        note_fname = note_fname.removesuffix(".md")
        if note_file.startswith(note_fname.rstrip()):
            # todo: rename file
            return data.replace("\n# ", "\n")
    return data

@transformations.append
def strip_subtitle_from_links(data: str, filename: Path) -> str:
    if "--short" in sys.argv[1:]:
        matches = list(re.finditer(
            r'\[\[(?P<citekey>[a-zA-Z0-9_-]+)(?:\|(?P<label>(?:(?!\]\]).)+))\]\]',
            data,
        ))
        for match in matches[::-1]:
            citekey, label = match.group("citekey"), match.group("label")
            if label and ":" in label:
                short_title, subtitle = label.split(":", 1)
                data = f"{data[:match.start()]}[[{citekey}|{short_title}]]{data[match.end():]}"
    return data


#@transformations.append
def strip_zotero_links(data: str, filename: Path) -> str:
    """
    Removes
    <sup>*[select](zotero://select/library/items/5UD85YP6) [PDF](zotero://open-pdf/library/items/M4R8MQSN*</sup>
    """
    return re.sub(
            r'\<sup\>\*\[select\]\(zotero\://select/[a-zA-Z0-9/]+\) \[PDF\]\(zotero\://open-pdf/[a-zA-Z0-9/]+\)\*\</sup\>',
            r'',
            data,
            flags = re.MULTILINE,# | re.DOTALL,
    )


@transformations.append
def link_mdnotes_to_zotero(data: str, filename: Path) -> str:
    """
    [[neffDONeRFRealTimeRendering2021|foobar]]
    to
    [[neffDONeRFRealTimeRendering2021|foobar]]<sup>*[select](zotero://select/library/items/5UD85YP6) [PDF](zotero://open-pdf/library/items/M4R8MQSN*</sup>
    """

    matches = list(re.finditer(
        r'\[\[(?P<citekey>[a-zA-Z0-9_-]+)(?:\|(?P<label>(?:(?!\]\]).)+))?\]\](?!\<sup\>\*\[select\]\(zotero://)',
        data,
    ))
    #if not any(match.group("label") for match in matches): return data
    citekeys: set = {
        match.group("citekey")
        for match in matches
        if not ( filename.parent.name == "zotero" and filename.name.startswith(match.group("citekey")) )
    }
    citekey_map = map_threaded_gather_dict(query_zotero_for_attachments, citekeys)

    for match in matches[::-1]:
        citekey, label = match.group("citekey"), match.group("label")
        if citekey_map.get(citekey) is not None:
            link = citekey if not label else f"{citekey}|{label}"
            data = f"{data[:match.start()]}[[{link}]]{citekey_map[citekey]}{data[match.end():]}"

    return data

@transformations.append
def link_pandoc_citation_to_zotero(data: str, filename: Path) -> str:
    """
    [@neffDONeRFRealTimeRendering2021]
    to
    [[neffDONeRFRealTimeRendering2021]]<sup>*[select](zotero://select/library/items/5UD85YP6), [PDF](zotero://open-pdf/library/items/M4R8MQSN)*</sup>
    """

    matches = list(re.finditer(
        r'\[@(?P<citekey>[a-zA-Z0-9_-]+)\](?!\<sup\>\*\[select\]\(zotero://)',
        data,
    ))
    citekeys: set = {
        match.group("citekey")
        for match in matches
        if not ( filename.parent.name == "zotero" and filename.name.startswith(match.group("citekey")) )
    }
    citekey_map = map_threaded_gather_dict(query_zotero_for_attachments, citekeys)

    for match in matches[::-1]:
        citekey = match.group("citekey")
        if citekey_map.get(citekey) is not None:
            data = f"{data[:match.start()]}[[{citekey}]]{citekey_map[citekey]}{data[match.end():]}"

    return data

@transformations.append
def link_pandoc_inline_citation_to_zotero(data: str, filename: Path) -> str:
    """
    [@neffDONeRFRealTimeRendering2021]
    to
    [[neffDONeRFRealTimeRendering2021]]<sup>*[select](zotero://select/library/items/5UD85YP6), [PDF](zotero://open-pdf/library/items/M4R8MQSN)*</sup>
    """

    matches = list(re.finditer(
        r'(?<!\[\[)@(?P<citekey>[a-zA-Z0-9_-]*[a-zA-Z0-9])(?!\]\]\<sup\>\*\[select\]\(zotero://)',
        data,
    ))
    citekeys: set = {
        match.group("citekey")
        for match in matches
        if not ( filename.parent.name == "zotero" and filename.name.startswith(match.group("citekey")) )
    }
    citekey_map = map_threaded_gather_dict(query_zotero_for_attachments, citekeys)

    for match in matches[::-1]:
        citekey = match.group("citekey")
        if citekey_map.get(citekey) is not None:
            data = f"{data[:match.start()]}[[{citekey}]]{citekey_map[citekey]}{data[match.end():]}"

    return data



@transformations.append
def fix_wikipedia_inline_equations(data: str, filename: Path) -> str:
    return re.sub(
        r'\{\\displaystyle '                   # start of wikipedia equation
            r'(?P<math>(?:'                    # named capture group
                r'(?! ?\}!\[).)*'              # terminate with negative lookahead assertion
            r')'
            r' ?\}'                            # end of equation
            r'!\['                             # start of image href label
                r'(\{\\displaystyle )?'        # optional prefix
                r'(?:'
                    r'(?:(?!\]\(http).)*'      # terminate with negative lookahead assertion
                r')'
                r'\}?'                         # optional postfix
            r'\]'                              # end of image href label
            r'\(https:\/\/[a-zA-Z0-9\/._]*\)', # image url
        r'$\g<math>$',
        data,
        flags = re.MULTILINE,# | re.DOTALL,
    )

@transformations.append
def fix_wikipedia_block_equations(data: str, filename: Path) -> str:
    return re.sub(
        r'\{\\displaystyle '                   # start of wikipedia equation
            r'(?P<math>(?:'                    # named capture group
                r'(?! ?\}!\[).)*'              # terminate with negative lookahead assertion
            r')'
            r' ?\}'                            # end of equation
            "\n\n"
            r'!\['                             # start of image href label
                r'(\{\\displaystyle )?'        # optional prefix
                r'(?:'
                    r'(?:(?!\]\(http).)*'      # terminate with negative lookahead assertion
                r')'
                r'\}?'                         # optional postfix
            r'\]'                              # end of image href label
            r'\(https:\/\/[a-zA-Z0-9\/._]*\)', # image url
        r'$$\n\g<math>\n$$',
        data,
        flags = re.MULTILINE,# | re.DOTALL,
    )

@transformations.append
def fix_wikipedia_citation_labels(data: str, filename: Path) -> str:
    # TODO: <sup> ?
    return re.sub(
        r'\[\[(?P<label>[0-9]+)\]\]\((?P<url>https?:\/\/[^\)]*)\)',
        r'[\[\g<label>\]](\g<url>)',
        data,
        flags = re.MULTILINE,# | re.DOTALL,
    )


@transformations.append
def deduplicate_url_labels(data: str, filename: Path) -> str:
    if filename.parent.name == "zotero" and filename.name.endswith("-zotero.md"):
        return data

    matches = list(re.finditer(
        r'\[(?P<label>(\\\]|[^\]])+)\]\((?P<url>(\\\)|[^\)])+)\)',
        data,
    ))
    for match in matches[::-1]:
        #print(match.group("label"), match.group("url"))
        if match.group("label") == match.group("url"):
            data = f"{data[:match.start()]}<{match.group('url')}>{data[match.end():]}"

    return data

@transformations.append
def lowercase_todo(data: str, filename: Path) -> str:
    return re.sub(
        r'\#TODO\b',
        r'#todo',
        data,
        #flags = re.MULTILINE,# | re.DOTALL,
    )

@transformations.append
def bullets(data: str, filename: Path) -> str:
    return (data
        .replace(" - • ", " - ")
        .replace(" -• ",  " - ")
        .replace("- • ",  "- ")
        .replace("-• ",   "- ")
        .replace(" *• ",  " * ")
        .replace("* • ",  "* ")
        .replace("*• ",   "* ")
        .replace(" • ",   " * ")
        .replace("• ",    " * ")
        .replace(" • ",   " * ")
        .replace("• ",    "* ")
    )

@transformations.append
def squash_spaces(data: str, filename: Path) -> str:
    return re.sub(
        r' +',
        r' ',
        data,
    )

@transformations.append
def remove_trailing_whitespace(data: str, filename: Path) -> str:
    return re.sub(
        r' +\n',
        r'\n',
        data,
    )

# TODO: filter to join words split with a hyphen across lines?


env = os.environ.copy()
env["PAGER"] = "cat" # TODO: colors?

for file in VAULT.rglob("*.md"):
    print(file)
    with open(file, "r") as f:
        data = f.read()
    orig_data = data
    for transformation in transformations:
        print("    -", transformation.__name__.capitalize().replace("__", " - ").replace(*"_ "), end="\r")
        data, prev_data = transformation(data, file), data
        if data != prev_data:
            print()
        else:
            sys.stdout.write("\033[K") # clear to end of line

    if data.strip() != orig_data.strip():
        #subprocess.run(["diff", "-Bwu", "--color", file, "-"], input=data, text=True)
        subprocess.run([
                "git",
                "diff",
                "-U0",
                #"--word-diff",
                #"--word-diff-regex=[<]|[>]|[^[:space:]]",
                "--no-index",
                "--", file, "-"
            ], input=data, env=env, text=True)
        # TODO: chop off 4 topmost lines ^
        if input("Apply changes? [y/N] ").lower().startswith("y"):
            with open(file, "w") as f:
                f.write(data)
        else:
            print("Changes were not applied.")

if N_ZOTERO_QUERIES:
    print("N_ZOTERO_QUERIES =", N_ZOTERO_QUERIES)

if not IS_ZOTERO_ONLINE:
    print("WARNING: Could not connect to Zotero...")