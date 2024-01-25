#date: 2024-01-25T16:42:51Z
#url: https://api.github.com/gists/8da607468162436d46162ee8f713ff2a
#owner: https://api.github.com/users/oggers

"""Manage article images."""
import argparse
import getopt
from io import BytesIO
import logging
from OFS.Application import Application
import os
from PIL import Image
from PIL.GifImagePlugin import GifImageFile
from PIL.JpegImagePlugin import JpegImageFile
from PIL.PngImagePlugin import PngImageFile
from PIL.TiffImagePlugin import TiffImageFile
from PIL.WebPImagePlugin import WebPImageFile
from plone import api
from plone.namedfile import NamedBlobImage
import re
import sys
import transaction
from typing import Dict, List
from zope.component.hooks import setSite


logging.basicConfig(format='%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


IMAGE_FORMATS = {
    GifImageFile.format: '.gif',
    JpegImageFile.format: '.jpg',
    PngImageFile.format: '.png',
    TiffImageFile.format: '.tif',
    WebPImageFile.format: '.webp',
}


REGISTERED_EXTENSIONS = dict([
    (ext, format_)
    for ext, format_ in Image.registered_extensions().items()
    if format in IMAGE_FORMATS.keys()])


def manage_object_images(
        app: Application,
        site: str,
        portal_type: str,
        image_field: str,
        min_size: int = None,
        content_type: str = None,
        path: str = None,
        output_file: str = None,
        change_format: str = None,
        change_width: int = None,
        dry: bool = False,
        no_image: bool = False):
    """Manage artivle images."""
    portal = app[site]
    setSite(portal)

    print(f'Searching {portal_type} in {site}...')
    objs = []
    for brain in api.content.find(
            portal_type=portal_type):
        if path and not path.match(brain.getPath()):
            continue
        obj = brain.getObject()
        field = getattr(obj, image_field)
        if (no_image and field is not None) or field is None:
            continue
        if min_size and min_size > field.getSize():
            continue
        if content_type and content_type != field.contentType:
            continue
        objs.append(obj)

    total = len(objs)
    if output_file:
        ofile = open(output_file, 'w', encoding='utf-8')
    for counter, obj in enumerate(objs, 1):
        field = getattr(obj, image_field)
        image_size = str(field.getSize()) if field else '-'
        content_type = field.contentType if field else '-'
        print(f'{counter}/{total} {obj.title} {content_type} {image_size}')
        if output_file:
            path = '/'.join(obj.getPhysicalPath())
            ofile.write(
                f'"{obj.title}",{path},{content_type},{image_size}\n')
        if change_format or change_width:
            changed = False
            if change_format:
                stem, ext = os.path.splitext(
                    field.filename)
                if REGISTERED_EXTENSIONS.get(ext) != change_format:
                    ext = IMAGE_FORMATS[change_format]
                new_filename = stem + ext
            else:
                new_filename = field.filename

            with Image.open(BytesIO(field.data)) as im:
                new_im = im
                if change_width and change_width < im.width:
                    changed = True
                    height = int(change_width / im.width * im.height)
                    new_im = new_im.resize((change_width, height))
                    print(f'Resizing to ({new_im.width}, {height})')
                format_ = change_format or im.format
                if format_ != im.format:
                    changed = True
                    if format_ == JpegImageFile.format and im.mode != 'RGB':
                        new_im = new_im.convert('RGB')
                stream = BytesIO()
                new_im.save(stream, format_)
                stream.seek(0)
            if changed:
                orig_filename = field.filename
                orig_size = field.getSize()
                article.image = NamedBlobImage(
                    data=stream.getvalue(),
                    filename=new_filename)
                if not dry:
                    transaction.commit()
                print(f'changed {orig_filename} -> {field.filename}, '
                      f'size: {orig_size} -> {field.getSize()}')

    if output_file:
        ofile.close()
    if dry:
        print('Dry selected, therefore nothing was changed.')


def get_args(argv: List[str]) -> Dict:
    """get command line arguments."""
    help_text = (
        'manage_article_images.py -s <site> '
        '[ -d ] '
        '[ -o <output file> ] '
        '[ -n | -m <size>, -t <format>, -p <regex path> ] '
        f'[ -x {"|".join(IMAGE_FORMATS.keys())} ] '
        '[ -w <resize-to-max-width> ]')
    try:
        opts, args = getopt.getopt(
            argv, 'dm:no:p:s:t:w:x:',
            ['change=', 'dry', 'minsize=', 'noimage',
             'ofile=', 'path=', 'site=', 'width=', 'type='])
    except getopt.GetoptError:
        print(help_text)
        sys.exit(2)

    args = {}
    no_image_incompatible = False
    for opt, arg in opts:
        if opt == '-h':
            print(help_text)
            sys.exit()
        elif opt in ('-s', '--site'):
            args['site'] = arg
        elif opt in ('-n', '--noimage'):
            args['no_image'] = True
        elif opt in ('-m', '--minsize'):
            args['min_size'] = int(arg)
            no_image_incompatible = True
        elif opt in ('-t', '--type'):
            args['content_type'] = arg
            no_image_incompatible = True
        elif opt in ('-p', '--path'):
            args['path'] = re.compile(arg)
        elif opt in ('-o', '--ofile'):
            args['output_file'] = arg
        elif opt in ('-x', '--change'):
            args['change_format'] = arg
            no_image_incompatible = True
        elif opt in ('-w', '--witdh'):
            args['change_width'] = int(arg)
            no_image_incompatible = True
        elif opt in ('-d', '--dry'):
            args['dry'] = True

    if ('change_format' in args
            and args['change_format'] not in IMAGE_FORMATS):
        print(help_text)
        sys.exit(2)

    if 'no_image' in args and no_image_incompatible:
        print(help_text)
        sys.exit(2)

    if 'site' not in args:
        print(help_text)
        sys.exit(2)

    return args


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="manage_object_images.py",
        description="Manage object images.")
    parser.add_argument(
        '-d', '--dry', help="don't change anything", action='store_true')
    parser.add_argument(
        'site', help="Plone site to perform actions on", metavar='PLONESITE')
    parser.add_argument(
        'portal_type', help="Scan this portal_type", metavar='PORTALTYPE')
    parser.add_argument(
        'image_field', help="Scan this attribute", metavar='ATTRIBUTE')
    parser.add_argument(
        '-x', '--change', help="list scale names used in site",
        choices=IMAGE_FORMATS)
    parser.add_argument(
        '-m', '--minsize',
        help="Filter images, only consider images with minimum size", type=int)
    parser.add_argument(
        '-t', '--type', help="Only consider images with this content type")
    parser.add_argument(
        '-p', '--path', help="Use this regex to filter the path of the image")
    parser.add_argument(
        '-o', '--ofile', help="Save results to this csv file")
    parser.add_argument(
        '-w', '--width', help="Change images with greater width to this width")
    parser.add_argument(
        '-n', '--noimage', help="Show only objects with empty image")
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = _parse_args(sys.argv[3:])
    manage_object_images(
        app,  # noqa: F821
        site=args.site,
        portal_type=args.portal_type,
        image_field=args.image_field,
        min_size=args.minsize,
        content_type=args.type,
        path=re.compile(args.path) if args.path else None,
        output_file=args.ofile,
        change_format=args.change,
        change_width=args.width,
        dry=args.dry,
        no_image=args.noimage)
