#date: 2023-07-07T17:09:10Z
#url: https://api.github.com/gists/1fce92dc54b181dbcb156f853d6c591c
#owner: https://api.github.com/users/danielweiv

#!/usr/bin/env python3
import sys
import json
import os
import os.path
import shutil
import logging
import tempfile
import glob
import argparse
import xml.etree.ElementTree as ET
import json
from fnmatch import fnmatch

from tqdm import tqdm # pip install tqdm
import requests # pip install requests



def download_file(url, dest_filepath = None, 
        chunk_size = 32*1024,
        strict_download = False,
        expected_content_type = None
        ):
    """ Download a file a report the progress via the reporthook """

    if not url:
        logging.warning("url not provided : doing nothing")
        return False

    logging.info("Downloading %s in %s" % (url, dest_filepath))
    os.makedirs(os.path.dirname(dest_filepath), exist_ok=True)

    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True, allow_redirects = not strict_download)

    # Raise error if the response isn't a 200 OK
    if strict_download and (r.status_code != requests.codes.ok):
        logging.info("Download failed [%d] : %s \n" % (r.status_code, r.headers))
        #r.raise_for_status()
        return False

    content_type = r.headers.get('Content-Type', "")
    if expected_content_type and content_type != expected_content_type:
        logging.info("Wrong expected type : %s != %s \n" % (content_type, expected_content_type))
        #r.raise_for_status()
        return False

    # Total size in bytes.
    total_size = int(r.headers.get('content-length', 0)); 

    with open(dest_filepath, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            
            for data in r.iter_content(chunk_size):
                read_size = len(data)

                f.write(data)
                pbar.update(read_size)

    logging.info("Download done \n")
    return True

def download_dash_docsets(dest_folder = None, prefered_cdn = "" , docset_pattern = "*"):
    """ 
    Dash docsets are located via dash feeds : https://github.com/Kapeli/feeds
    zip file : https://github.com/Kapeli/feeds/archive/master.zip
    """
    feeds_zip_url = "https://github.com/Kapeli/feeds/archive/master.zip"

    if not dest_folder:
        dest_folder = os.getcwd()

    # Creating destination folder
    dash_docset_dir = dest_folder #os.path.join(dest_folder, "DashDocsets")
    os.makedirs(dash_docset_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdirname:
        logging.debug('created temporary directory : %s', tmpdirname)

        feeds_archive = os.path.join(tmpdirname, "feeds.zip")
        feeds_dir = os.path.join(tmpdirname, "feeds-master")

        # Download and unpack feeds
        download_file(feeds_zip_url, feeds_archive)
        shutil.unpack_archive(feeds_archive, os.path.dirname(feeds_archive))

        # parse xml feeds and extract urls
        for feed_filepath in glob.glob("%s/%s.xml" % (feeds_dir, docset_pattern)):

            feed_name, xml_ext  = os.path.splitext(os.path.basename(feed_filepath))
            logging.debug("%s : %s" % (feed_name, feed_filepath))
            
            cdn_url = None
            tree = ET.parse(feed_filepath)
            root = tree.getroot()
            for url in root.findall("url"):
                logging.debug("\turl found : %s" % url.text)

                if "%s.kapeli.com" % prefered_cdn in url.text:
                    logging.debug("\tselected cdn url : %s" % url.text)
                    cdn_url = url.text


            if cdn_url :
                docset_dest_filepath = os.path.join(dash_docset_dir, "%s.tgz" % feed_name)
                download_file(cdn_url, docset_dest_filepath, strict_download = True)
                shutil.move(feed_filepath, os.path.join(dash_docset_dir, os.path.basename(feed_filepath)))

def download_user_contrib_docsets(dest_folder = None, prefered_cdn = "sanfransisco" , docset_pattern = "*"):
    """ 
    Dash docsets are located via dash feeds : https://github.com/Kapeli/feeds
    zip file : https://github.com/Kapeli/feeds/archive/master.zip
    """
    feeds_json_url = "http://%s.kapeli.com/feeds/zzz/user_contributed/build/index.json" % prefered_cdn

    if not dest_folder:
        dest_folder = os.getcwd()

    # Creating destination folder
    user_contrib_docset_dir = os.path.join(dest_folder, "zzz","user_contributed","build")
    os.makedirs(user_contrib_docset_dir, exist_ok=True)
    download_file(feeds_json_url, os.path.join(user_contrib_docset_dir,"index.json"))

    with tempfile.TemporaryDirectory() as tmpdirname:
        logging.debug('created temporary directory : %s', tmpdirname)

        feeds_json = os.path.join(tmpdirname, "feeds.json")

        # Download feed
        download_file(feeds_json_url, feeds_json)
        with open (feeds_json, "r") as js_fd:
            json_feeds = json.load(js_fd)
        docsets = json_feeds['docsets']
        
        # parse xml feeds and extract urls
        for docset in sorted(filter(lambda x: fnmatch(x, docset_pattern), docsets)):
            docset_info = docsets[docset]

            # url format for packages that specify "specific_versions"
            # docset_url = "http://%s.kapeli.com/feeds/zzz/user_contributed/build/%s/versions/%s/%s" % (
            #     prefered_cdn,
            #     docset,
            #     docset_info['version'],
            #     docset_info['archive'],
            # )

            docset_url = "http://%s.kapeli.com/feeds/zzz/user_contributed/build/%s/%s" % (
                prefered_cdn,
                docset,
                docset_info['archive'],
            )

            docset_dest_filepath = os.path.join(user_contrib_docset_dir, docset, docset_info['archive'])
            download_file(docset_url, docset_dest_filepath, strict_download = True, expected_content_type = 'application/x-tar')



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='A downloader for Dash Docsets'
    )

    parser.add_argument("--dash", 
        help="only download dash docsets", 
        action="store_true"
    )

    parser.add_argument("--user-contrib", 
        help="only download user contrib docsets", 
        action="store_true"
    )

    parser.add_argument("-d", "--docset", 
        help="only download a specifics docsets. This option support the glob pattern",
        default="*", 
    )

    parser.add_argument("-v", "--verbose", 
        help="increase output verbosity", 
        action="store_true"
    )

    parser.add_argument("-o", "--output", 
        help="change output directory ", 
        default=os.getcwd()
    )

    parser.add_argument("-c", "--cdn", 
        help="choose cdn (sanfrancisco by default)",
        default = "sanfrancisco", 
        choices=[
            'sanfrancisco',
            'london',
            'newyork',
            'tokyo',
            'frankfurt',
            'sydney',
            'singapore',
        ],
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    os.makedirs(args.output, exist_ok=True)

    with open(os.path.join(args.output, "latencyTest.txt"), 'w') as latency:
        pass
    with open(os.path.join(args.output, "latencyTest_v2.txt"), 'w') as latency:
        pass

    if not args.user_contrib:
        download_dash_docsets(
            dest_folder = args.output,
            prefered_cdn = args.cdn,
            docset_pattern = args.docset
        )

    if not args.dash:
        download_user_contrib_docsets(
            dest_folder = args.output,
            prefered_cdn = args.cdn,
            docset_pattern = args.docset
        )