#date: 2022-02-10T17:01:09Z
#url: https://api.github.com/gists/71400c3dc1db7c3795cecd605aee5fbd
#owner: https://api.github.com/users/carj

from xml.etree.ElementTree import canonicalize
from pyPreservica import *
import logging
from xml.etree import ElementTree

logger = logging.getLogger(__name__)

LOG_FILENAME = 'arklib.log'
logging.basicConfig(level=logging.INFO, filename=LOG_FILENAME, filemode="a")

consoleHandler = logging.StreamHandler()
logging.getLogger().addHandler(consoleHandler)

ARK_NS = "https://www.library.arkansas.gov/"
field = "ReferenceURL"

xml.etree.ElementTree.register_namespace('ast', ARK_NS)

if __name__ == '__main__':
    client = EntityAPI()
    logger.info(client)
    for entity in client.all_descendants():
        full_entity = client.entity(entity.entity_type, entity.reference)
        if full_entity.has_metadata():
            xml_document = client.metadata_for_entity(full_entity, ARK_NS)
            if xml_document:
                document = ElementTree.fromstring(xml_document)
                logging.info(f"Writing metadata for entity: {full_entity.title} to file {full_entity.reference}.xml")
                with open(f"{full_entity.reference}.xml", mode='w', encoding='utf-8') as fd:
                    canonicalize(xml_document, out=fd)
                url_field = document.find(f'.//{{{ARK_NS}}}{field}')
                if hasattr(url_field, 'text') and not url_field.text.startswith("https://arklib.access.preservica.com"):
                    new_url = f"https://arklib.access.preservica.com/uncategorized/{entity.entity_type.value}_{entity.reference}"
                    xml_document = xml_document.replace(url_field.text, new_url)
                    full_entity = client.update_metadata(full_entity, ARK_NS, xml_document)
                    logging.info(f"Updated asset: {full_entity.title} with new URL {url_field.text}")
