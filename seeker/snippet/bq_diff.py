#date: 2025-12-26T17:01:48Z
#url: https://api.github.com/gists/6531b58d0230b98b698866492b68e635
#owner: https://api.github.com/users/ivanpu

#!/bin/python
# Copyright 2025 Puntiy Ivan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the “Software”), to deal in the Software without
# restriction, including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
# OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

from __future__ import annotations

import json
import sys
import tkinter as tk
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from dataclasses import dataclass, field
from difflib import ndiff, SequenceMatcher
from enum import StrEnum
from itertools import chain
from pathlib import Path
from tkinter import font, ttk
from types import NoneType
from typing import Any, Final, Self


@dataclass(slots=True, frozen=True, kw_only=True)
class Args:
    old_instance: Path
    new_instance: Path
    quests_base_path: Path


def get_args() -> Args:
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("old_instance", type=Path, help="Path to the old instance")
    parser.add_argument("new_instance", type=Path, help="Path to the new instance")
    parser.add_argument(
        "--quests-base-path", type=Path, metavar="PATH",
        default=".minecraft/config/betterquesting/DefaultQuests",
        help="Relative path to configurations folder")

    return Args(**parser.parse_args().__dict__)


_IS_TRACE: Final[bool] = False
_trace_level: int = 0


def trace(func):
    fname = getattr(func, "__qualname__", func.__name__)
    def trace_decor(*args, **kwargs):
        global _trace_level
        prefix: str = "  " * _trace_level
        _trace_level += 1
        try:
            print(prefix + ">> ENTER >>", fname, "Args:", args, "KWargs:", kwargs, file=sys.stderr)
            value = func(*args, **kwargs)
            print(prefix + "<<  EXIT <<", fname, "->", repr(value), file=sys.stderr)
        except Exception as e:
            print(prefix + "<<  EXIT <<", fname, "^_^", type(e).__name__, file=sys.stderr)
            raise
        finally:
            _trace_level -= 1
        return value

    return trace_decor if _IS_TRACE else func


@dataclass(slots=True, frozen=True)
class QuestLine:
    id: str
    name: str
    desc: str = ""


@dataclass(slots=True, frozen=True)
class Quest:
    id: str
    name: str
    desc: str
    questline: QuestLine = field(compare=False)

    @property
    @trace
    def questline_name(self) -> str:
        return getattr(self.questline, "name", "")

    @property
    @trace
    def questline_id(self) -> str:
        return getattr(self.questline, "id", "")

    @classmethod
    @trace
    def from_file(cls, json_path: Path, questline: QuestLine) -> Self:
        data = json.loads(json_path.read_text())
        quest_id = f"{data['questIDHigh:4']}:{data['questIDLow:4']}"
        quest_name = data["properties:10"]["betterquesting:10"]["name:8"]
        quest_desc = data["properties:10"]["betterquesting:10"]["desc:8"]
        return cls(quest_id, quest_name, quest_desc, questline)


class DiffState(StrEnum):
    def __new__(cls, value: str, color: str, background: str | None = None) -> Self:
        obj: Self = str.__new__(cls, value)
        obj._value_: str = value
        obj.color: str = color
        obj.background: str = background or color
        return obj

    EMPTY = "empty", "gray"
    ADDED = "added", "green", "lightgreen"
    REMOVED = "removed", "red", "lightcoral"
    MOVED = "moved", "darkorange"
    CHANGED = "changed", "blue"
    UNCHANGED = "", "black"


class DiffMode(StrEnum):
    OLD = "old"
    NEW = "new"


type Meta[T] = dict[str, Revision[T]]

@dataclass(slots=True)
class Revision[T]:
    old: T | None = None
    new: T | None = None

    def set(self, mode: DiffMode, value: T) -> None:
        setattr(self, mode, value)

    @property
    @trace
    def state(self) -> DiffState:
        if self.old is self.new is None:
            return DiffState.EMPTY
        elif self.old is None:
            return DiffState.ADDED
        elif self.new is None:
            return DiffState.REMOVED
        elif getattr(getattr(self.old, "questline", None), "id", None) != getattr(getattr(self.new, "questline", None), "id", None):
            return DiffState.MOVED
        elif self.old == self.new:
            return DiffState.UNCHANGED
        else:
            return DiffState.CHANGED

    @trace
    def __get_attr(self, attr_name: str, default: Any) -> Any:
        return getattr(self.old or self.new, attr_name, default)

    @property
    @trace
    def name(self) -> str:
        return self.__get_attr("name", "")

    @property
    @trace
    def questline_name(self) -> str:
        return getattr(self.__get_attr("questline", None), "name", "")

    @property
    @trace
    def desc(self) -> str:
        return self.__get_attr("desc", "")

    @property
    @trace
    def questline(self) -> QuestLine | None:
        return self.__get_attr("questline", self.old or self.new)


@trace
def load_questlines(base_path: Path) -> list[QuestLine]:
    quests: list[QuestLine] = []
    quest_lines_dir: Path = base_path / "QuestLines"
    for line in base_path.joinpath("QuestLinesOrder.txt").read_text().splitlines():
        qid, name = (val.strip() for val in line.split(":", 1))
        ql_meta_files: list[Path] = list(chain(quest_lines_dir.glob(f"*{qid}/QuestLine.json"), quest_lines_dir.glob(f"*{qid}.json")))
        desc: str = ""
        if ql_meta_files:
            ql_meta = json.loads(ql_meta_files[0].read_text())
            desc = ql_meta["properties:10"]["betterquesting:10"]["desc:8"]
        quests.append(QuestLine(qid, name, desc))
    return quests


@trace
def make_questlines_meta(old_instance: Path, new_instance: Path, quests_dir: Path) -> Meta[QuestLine]:
    meta: Meta[QuestLine] = {}
    for instance_path, mode in zip([old_instance, new_instance], DiffMode, strict=True):
        print(f":: {mode} questlines")
        new: int = 0
        for questline in load_questlines(instance_path / quests_dir):
            meta.setdefault(questline.id, Revision()).set(mode, questline)
            new += 1
        print(f"Discovered {new} quest line(s)")
    return meta


@trace
def load_quests(base_path: Path, questline: QuestLine) -> list[Quest]:
    return [Quest.from_file(file, questline) for file in base_path.iterdir()]


@trace
def make_quests_meta(
    old_instance: Path, new_instance: Path, quests_dir: Path, questlines: Meta[QuestLine]
) -> tuple[Meta[Quest], Meta[QuestLine]]:
    quests: Meta[Quest] = {}
    unreg_questlines: Meta[QuestLine] = {}
    old_quests_path: Path = old_instance / quests_dir / "Quests"
    new_quests_path: Path = new_instance / quests_dir / "Quests"
    for base_path, mode in zip([old_quests_path, new_quests_path], DiffMode, strict=True):
        print(f":: {mode} quests")
        new_quests: int = 0
        new_questlines: int = 0
        for folder in base_path.iterdir():
            for ql_id, questline_rev in questlines.items():
                if folder.name.endswith(ql_id):
                    break
            else:
                print("Adding unregistered quest line:", folder.name)
                questline_rev = unreg_questlines.setdefault(folder.name, Revision())
                questline_rev.set(mode, QuestLine(folder.name, folder.name))
                new_questlines += 1

            for quest in load_quests(folder, getattr(questline_rev, mode)):
                quest_rev = quests.setdefault(quest.id, Revision())
                if getattr(quest_rev, mode) is not None:
                    print("Duplicate ID:", quest.id, f"{file.parent.name}/{file.name}")
                else:
                    quest_rev.set(mode, quest)
                    new_quests += 1
        print(f"Discovered {new_quests} quest(s)")
        if new_questlines:
            print(f"Discovered {new_questlines} unregistered quest line(s)")
    return quests, unreg_questlines


class Viewer(ttk.Panedwindow):
    def __init__(self, master: tk.Misc, quests: Meta[Quest], questlines: Meta[QuestLine]):
        super().__init__(orient=tk.HORIZONTAL)

        self.__meta: Meta[Quest | QuestLine] = {**quests, **questlines}

        self.__quests: QuestsTree = QuestsTree(self, quests, questlines)
        self.__quests.tree.bind("<<TreeviewSelect>>", self.__select_quest)
        self.add(self.__quests)

        info = ttk.Frame(self, relief="raised", padding=3)
        self.add(info)
        
        self.__quest_name: tk.StringVar = tk.StringVar(value="<<< Select a quest from the left <<<")
        ttk.Label(info, textvariable=self.__quest_name, font="TkHeadingFont").grid(column=1, row=1, columnspan=5, sticky="w")

        self.__questline_name: tk.StringVar = tk.StringVar()
        ttk.Label(info, textvariable=self.__questline_name, font="TkSmallCaptionFont").grid(column=1, row=0, columnspan=5, sticky="w")

        self.__diff_mode: tk.StringVar = tk.StringVar(value="diff")
        ttk.Radiobutton(info, text="Diff", variable=self.__diff_mode, value="diff", command=self.__select_quest).grid(column=1, row=3)
        ttk.Radiobutton(info, text="Old", variable=self.__diff_mode, value="old", command=self.__select_quest).grid(column=2, row=3)
        ttk.Radiobutton(info, text="New", variable=self.__diff_mode, value="new", command=self.__select_quest).grid(column=3, row=3)

        self.__diff: DiffView = DiffView(info, self.__diff_mode)
        self.__diff.grid(column=1, row=2, columnspan=5, sticky="nsew")

        info.columnconfigure(4, weight=1)
        info.rowconfigure(2, weight=1)
        for child in info.winfo_children():
            child.grid_configure(padx=3, pady=3)

    @trace
    def __select_quest(self, *_) -> None:
        selection = self.__quests.tree.selection()
        if selection:
            revision: Revision[Quest | QuestLine] = self.__meta[selection[0]]
            self.__quest_name.set(self.__diff.simple_diff(revision, "name"))
            self.__questline_name.set(self.__diff.simple_diff(revision, "questline_name"))
            self.__diff.show_diff(getattr(revision.old, "desc", ""), getattr(revision.new, "desc", ""))


class QuestsTree(ttk.Frame):
    def __init__(self, master: tk.Misc, quests: Meta[Quest], questlines: Meta[QuestLine], **kwargs):
        super().__init__(master, **kwargs)

        ttk.Label(self, text="Quests:").grid(column=0, row=0, sticky="w")

        self.__quests: ttk.Treeview = ttk.Treeview(self, show="tree", selectmode="browse")
        self.__quests.grid(column=0, row=1, sticky="nsew")
        for state in DiffState:
            if state is not DiffState.UNCHANGED:
                self.__quests.tag_configure(state, foreground=state.color)

        f: font.Font = font.nametofont("TkDefaultFont")
        max_name_width: int = 0
        for ql_id, ql in questlines.items():
            self.__quests.insert("", "end", ql_id, text=ql.name, tags=ql.state)
            max_name_width = max(max_name_width, f.measure(ql.name))
        for qid, quest in sorted(quests.items(), key=lambda i: i[1].name):  # TODO: any special handling for "§" tags?
            self.__quests.insert(quest.questline.id, "end", qid, text=quest.name, tags=quest.state)
            max_name_width = max(max_name_width, f.measure(quest.name))
        max_name_width += f.measure(4*"M")  # should be enough extra padding for a tree with 2 levels
        self.__quests.column("#0", minwidth=max_name_width)

        sh = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.__quests.xview)
        sv = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.__quests.yview)
        sh.grid(column=0, row=2, sticky="ew")
        sv.grid(column=1, row=1, sticky="ns")
        self.__quests.configure(xscrollcommand=sh.set, yscrollcommand=sv.set)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

    @property
    def tree(self) -> ttk.Treeview:
        return self.__quests


class DiffView(ttk.Frame):
    def __init__(self, master: tk.Misc, mode: tk.StringVar, **kwargs):
        super().__init__(master, relief="sunken", border=1, **kwargs)

        self.__mode: tk.StringVar = mode

        self.__text: tk.Text = tk.Text(self, border=0, font="TkFixedFont", wrap="word")
        self.__text.grid(column=0, row=0, sticky="nsew")
        self.__text.configure(state="disabled")
        self.__text.tag_configure(DiffState.ADDED, background=DiffState.ADDED.background)
        self.__text.tag_configure(DiffState.REMOVED, background=DiffState.REMOVED.background)

        sv = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.__text.yview)
        sv.grid(column=1, row=0, sticky="ns")
        self.__text.configure(yscrollcommand=sv.set)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

    def get_text(self) -> str:
        return self.__text.get("1.0", "end")

    @trace
    def show_diff(self, old: str, new: str) -> None:
        self.__text.configure(state="normal")
        self.__text.delete("1.0", "end")
        match self.__mode.get():
            # TODO: apply formatting ([]-tags, §-tags, etc.) in non-diff modes
            case "old":
                self.__text.insert("end", old)
            case "new":
                self.__text.insert("end", new)
            case "diff":
                seq = SequenceMatcher(a=old, b=new)
                for tag, i1, i2, j1, j2 in seq.get_opcodes():
                    match tag:
                        case 'equal':
                            self.__text.insert("end", old[i1:i2])
                        case 'replace':
                            self.__text.insert("end", old[i1:i2], DiffState.REMOVED)
                            self.__text.insert("end", new[j1:j2], DiffState.ADDED)
                        case 'delete':
                            self.__text.insert("end", old[i1:i2], DiffState.REMOVED)
                        case 'insert':
                            self.__text.insert("end", new[j1:j2], DiffState.ADDED)
        self.__text.configure(state="disabled")

    @trace
    def simple_diff(self, revision: Revision[Quest | QuestLine], attr: str) -> str:
        old_val: str = getattr(revision.old, attr, "")
        new_val: str = getattr(revision.new, attr, "")
        match self.__mode.get():
            case "old":
                return old_val
            case "new":
                return new_val
            case _:
                return (
                    f"{old_val}  >>>  {new_val}"
                    if revision.state in (DiffState.CHANGED, DiffState.MOVED) and old_val != new_val
                    else getattr(revision, attr, "")
                )


def start_gui(name: Revision[str], quests: Meta[Quest], questlines: Meta[QuestLine]) -> None:
    print("\nStarting GUI...")
    root = tk.Tk()
    style = ttk.Style()
    root.title(f"BQ Changes Viewer | {name.old} → {name.new}")
    Viewer(root, quests, questlines).grid(column=0, row=0, sticky="nsew", padx=3, pady=3)
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    root.mainloop()


def main() -> None:
    args: Args = get_args()
    quests: Meta[Quest]
    more_questlines: Meta[QuestLine]

    print("Analyzing...")
    questlines: Meta[QuestLine] = make_questlines_meta(args.old_instance, args.new_instance, args.quests_base_path)
    quests, more_questlines = make_quests_meta(args.old_instance, args.new_instance, args.quests_base_path, questlines)
    questlines.update(more_questlines)

    start_gui(Revision(args.old_instance.name, args.new_instance.name), quests, questlines)


if __name__ == "__main__":
    main()

