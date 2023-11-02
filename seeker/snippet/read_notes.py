#date: 2023-11-02T16:54:28Z
#url: https://api.github.com/gists/3979894ab2f813883170613dbf310d08
#owner: https://api.github.com/users/ttomasz

import bz2
import xml
import xml.dom.pulldom
from dataclasses import dataclass
from datetime import datetime
from typing import Generator, Optional, List
from xml.dom.minidom import Element

import sys

@dataclass(frozen=True)
class Comment:
    action: str
    timestamp: datetime
    uid: Optional[int]
    user: Optional[str]
    text: Optional[str]


@dataclass(frozen=True)
class Note:
    id: int
    lat: float
    lon: float
    created_at: datetime
    closed_at: Optional[datetime]
    comments: List[Comment]


def parse_xml_file(file: bz2.BZ2File) -> Generator[Note, None, None]:
    event_stream = xml.dom.pulldom.parse(file)
    counter = 0
    for event, element in event_stream:
        # element: Element = element  # just for typing
        if event == xml.dom.pulldom.START_ELEMENT and element.tagName == "note":
            counter += 1
            event_stream.expandNode(element)
            comments = []
            for child in element.childNodes:
                # child: Element = child  # for typing
                if type(child) == Element and child.tagName == "comment":
                    uid = child.getAttribute("uid")
                    user = child.getAttribute("user")
                    comments.append(
                        Comment(
                            action=child.getAttribute("action"),
                            timestamp=datetime.fromisoformat(
                                child.getAttribute("timestamp").replace("Z", "+00:00")
                            ),
                            uid=int(uid) if uid else None,
                            user=user if user else None,
                            text=child.firstChild.nodeValue if child.firstChild else None,  # text value is of type xml.dom.minidom.Text
                        )
                    )
            created_at = element.getAttribute("created_at")
            closed_at = element.getAttribute("closed_at")
            yield Note(
                id=int(element.getAttribute("id")),
                lon=float(element.getAttribute("lon")),
                lat=float(element.getAttribute("lat")),
                created_at=datetime.fromisoformat(created_at.replace("Z", "+00:00")),
                closed_at=datetime.fromisoformat(closed_at.replace("Z", "+00:00")) if closed_at else None,
                comments=comments,
            )
            if counter % 10000 == 0:
                print(f"Processed: {counter} features.")
    print(f"Finished parsing file. There were {counter} notes.")


# provide your own filepath 
file_path = "planet-notes-latest.osn.bz2"
with bz2.BZ2File(file_path) as fp:
    for idx, note in enumerate(parse_xml_file(fp)):
        print(idx, note)
