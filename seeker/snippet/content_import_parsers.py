#date: 2024-07-02T16:44:57Z
#url: https://api.github.com/gists/cb8e644abda4c6b7f0298fa917d8a3bc
#owner: https://api.github.com/users/thibaudcolas

import bs4
import mammoth
from django.core import exceptions, validators
from wagtail.embeds import embeds
from wagtail_content_import.parsers import base as base_parser


class DocxHTMLParser(base_parser.DocumentParser):
    def __init__(self, document):
        self.document = document

    def close_paragraph(self, block, stream_data):
        if block:
            stream_data.append({"type": "html", "value": "".join(block)})
        block.clear()
        return

    def parse(self):
        html = mammoth.convert_to_html(self.document).value

        soup = bs4.BeautifulSoup(html, "html5lib")

        stream_data = []

        # Run through contents and populate stream
        current_paragraph_block = []

        for tag in soup.body.recursiveChildGenerator():
            # Remove all inline styles and classes
            if hasattr(tag, "attrs"):
                for attr in ["class", "style"]:
                    tag.attrs.pop(attr, None)

        title = ""

        for tag in soup.body.contents:
            if isinstance(tag, bs4.NavigableString):
                stream_data.append({"type": "html", "value": str(tag)})
            else:
                if tag.name == "h1":
                    if not title:
                        title = tag.text
                    else:
                        self.close_paragraph(current_paragraph_block, stream_data)
                        stream_data.append({"type": "heading", "value": tag.text})
                elif tag.name == "h2":
                    self.close_paragraph(current_paragraph_block, stream_data)
                    stream_data.append({"type": "heading", "value": tag.text})
                elif tag.name in ["h3", "h4", "h5", "h6"]:
                    self.close_paragraph(current_paragraph_block, stream_data)
                    stream_data.append({"type": "subheading", "value": tag.text})
                elif tag.name == "img":
                    # Break the paragraph and add an image
                    self.close_paragraph(current_paragraph_block, stream_data)
                    stream_data.append(
                        {
                            "type": "image",
                            "value": tag.get("src"),
                            "title": tag.get("alt", ""),
                        }
                    )
                elif tag.text:
                    if tag.text.startswith("http:") or tag.text.startswith("https:"):
                        validate = validators.URLValidator()
                        url = tag.text.strip()
                        try:
                            validate(url)

                            if embed := embeds.get_embed(url):
                                self.close_paragraph(
                                    current_paragraph_block, stream_data
                                )
                                stream_data.append({"type": "embed", "value": embed})
                        except exceptions.ValidationError:
                            current_paragraph_block.append(str(tag))
                    else:
                        current_paragraph_block.append(str(tag))

                if tag.find_all("img"):
                    # Break the paragraph and add images
                    self.close_paragraph(current_paragraph_block, stream_data)
                    for img in tag.find_all("img"):
                        stream_data.append(
                            {
                                "type": "image",
                                "value": img.get("src"),
                                "title": img.get("alt", ""),
                            }
                        )

            self.close_paragraph(current_paragraph_block, stream_data)

        return {"title": title, "elements": stream_data}
