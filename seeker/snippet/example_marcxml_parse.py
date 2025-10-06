#date: 2025-10-06T16:39:50Z
#url: https://api.github.com/gists/f3422fb0a2687262078339b9f2ecb620
#owner: https://api.github.com/users/martinlovell


from pymarc import XmlHandler, parse_xml
import tarfile
import argparse

cnt = 0
def process_record(record):
    global cnt
    cnt += 1
    print(f"{cnt}: {record['001'].data}")

class PymarcXmlHandler(XmlHandler):
    def __init__(self, record_processor):
        super().__init__()
        self.record_processor = record_processor
    def process_record(self, record):
        if self.record_processor:
            self.record_processor(record)

def parse_file(file):
    handler = PymarcXmlHandler(record_processor=process_record)
    if file.endswith('xml') or file.endswith('gz'):
        with open(file, 'rb') as fh:
            if file.endswith('gz'):
                fh = get_first_tar_entry(fh)
            parse_xml(fh, handler)

def get_first_tar_entry(fh):
    tar = tarfile.open(fileobj=fh, mode="r:gz")
    for member in tar.getmembers():
        return tar.extractfile(member)

def main():
    parser = argparse.ArgumentParser(description='Read and print IDs from a MARC XML (e.g. marc_xml.xml, marc_xml.tar.gz)')
    parser.add_argument('-f', '--file', nargs='?', help='MARC XML file')
    options = parser.parse_args()
    parse_file(options.file)

if __name__ == '__main__':
    main()