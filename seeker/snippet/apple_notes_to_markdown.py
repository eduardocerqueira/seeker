#date: 2025-12-09T16:56:49Z
#url: https://api.github.com/gists/f2e9cf51f6dde57260442536cc8af16b
#owner: https://api.github.com/users/joswr1ght

#!/usr/bin/env python3
# /// script
# requires-python = '>=3.10'
# dependencies = []
# ///
"""
Convert Apple Notes from NoteStore.sqlite to Markdown files.

Written with Claude Code, 2025-12-09 Joshua Wright

Based on dunhamsteve/notesutils (https://github.com/dunhamsteve/notesutils)
to decode Apple Notes' CRDT-based protobuf format.

Usage:
    uv run apple_notes_to_markdown.py <input_folder> <output_folder>
"""

import argparse
import re
import shutil
import sqlite3
import struct
from pathlib import Path
from zlib import decompress


# Protobuf parser

def uvarint(data, pos):
    """Decode unsigned variable-length integer."""
    x = s = 0
    while True:
        b = data[pos]
        pos += 1
        x = x | ((b & 0x7f) << s)
        if b < 0x80:
            return x, pos
        s += 7


def readbytes(data, pos):
    """Read length-delimited bytes."""
    length, pos = uvarint(data, pos)
    return data[pos:pos + length], pos + length


def readstruct(fmt, length):
    """Create a struct reader."""
    return lambda data, pos: (struct.unpack_from(fmt, data, pos)[0], pos + length)


READERS = [uvarint, readstruct('<d', 8), readbytes, None, None, readstruct('<f', 4)]


def parse(data, schema):
    """Parse a protobuf message according to schema."""
    obj = {}
    pos = 0
    while pos < len(data):
        val, pos = uvarint(data, pos)
        wire_type = val & 7
        key = val >> 3
        val, pos = READERS[wire_type](data, pos)
        if key not in schema:
            continue
        name, repeated, field_type = schema[key]
        if isinstance(field_type, dict):
            val = parse(val, field_type)
        if field_type == 'string':
            val = val.decode('utf8')
        if repeated:
            val = obj.get(name, []) + [val]
        obj[name] = val
    return obj


# Protobuf schemas for Apple Notes

SCHEMA_STRING = {
    2: ['string', 0, 'string'],
    5: ['attributeRun', 1, {
        1: ['length', 0, 0],
        2: ['paragraphStyle', 0, {
            1: ['style', 0, 0],
            4: ['indent', 0, 0],
            5: ['todo', 0, {
                1: ['todoUUID', 0, 'bytes'],
                2: ['done', 0, 0]
            }]
        }],
        5: ['fontHints', 0, 0],
        6: ['underline', 0, 0],
        7: ['strikethrough', 0, 0],
        9: ['link', 0, 'string'],
        12: ['attachmentInfo', 0, {
            1: ['attachmentIdentifier', 0, 'string'],
            2: ['typeUTI', 0, 'string']
        }]
    }]
}

SCHEMA_DOC = {2: ['version', 1, {3: ['data', 0, SCHEMA_STRING]}]}

SCHEMA_OID = {
    2: ['unsignedIntegerValue', 0, 0],
    4: ['stringValue', 0, 'string'],
    6: ['objectIndex', 0, 0]
}
SCHEMA_DICTIONARY = {1: ['element', 1, {1: ['key', 0, SCHEMA_OID], 2: ['value', 0, SCHEMA_OID]}]}
SCHEMA_TABLE = {2: ['version', 1, {3: ['data', 0, {
    3: ['object', 1, {
        1: ['registerLatest', 0, {2: ['contents', 0, SCHEMA_OID]}],
        6: ['dictionary', 0, SCHEMA_DICTIONARY],
        10: ['string', 0, SCHEMA_STRING],
        13: ['custom', 0, {
            1: ['type', 0, 0],
            3: ['mapEntry', 1, {
                1: ['key', 0, 0],
                2: ['value', 0, SCHEMA_OID]
            }]
        }],
        16: ['orderedSet', 0, {
            1: ['ordering', 0, {
                1: ['array', 0, {
                    1: ['contents', 0, SCHEMA_STRING],
                    2: ['attachments', 1, {1: ['index', 0, 0], 2: ['uuid', 0, 0]}]
                }],
                2: ['contents', 0, SCHEMA_DICTIONARY]
            }],
            2: ['elements', 0, SCHEMA_DICTIONARY]
        }]
    }],
    4: ['keyItem', 1, 'string'],
    5: ['typeItem', 1, 'string'],
    6: ['uuidItem', 1, 'bytes']
}]}]}


def process_archive(table):
    """Decode a 'CRArchive' for tables."""
    def dodict(v):
        rval = {}
        for e in v.get('element', []):
            rval[coerce(e['key'])] = coerce(e['value'])
        return rval

    def coerce(o):
        [(k, v)] = o.items()
        if k == 'custom':
            rval = dict((table['keyItem'][e['key']], coerce(e['value'])) for e in v['mapEntry'])
            typ = table['typeItem'][v['type']]
            if typ == 'com.apple.CRDT.NSUUID':
                return table['uuidItem'][rval['UUIDIndex']]
            if typ == 'com.apple.CRDT.NSString':
                return rval['self']
            return rval
        if k == 'objectIndex':
            return coerce(table['object'][v])
        if k == 'registerLatest':
            return coerce(v['contents'])
        if k == 'orderedSet':
            elements = dodict(v['elements'])
            contents = dodict(v['ordering']['contents'])
            rval = []
            for a in v['ordering']['array']['attachments']:
                value = contents[a['uuid']]
                if value not in rval and a['uuid'] in elements:
                    rval.append(value)
            return rval
        if k == 'dictionary':
            return dodict(v)
        if k in ('stringValue', 'unsignedIntegerValue', 'string'):
            return v
        raise Exception(f'unhandled type {k}')

    return coerce(table['object'][0])


def render_table_markdown(table):
    """Render a table to markdown."""
    table = process_archive(table)
    rows = []
    columns = table.get('crColumns', [])

    for row_id in table.get('crRows', []):
        row_cells = []
        for col_id in columns:
            cell = table.get('cellColumns', {}).get(col_id, {}).get(row_id)
            cell_text = render_markdown(cell, {}) if cell else ''
            # Clean up cell text for table format
            cell_text = cell_text.replace('\n', ' ').strip()
            row_cells.append(cell_text)
        rows.append(row_cells)

    if not rows:
        return ''

    # Build markdown table
    lines = []
    # Header (first row)
    lines.append('| ' + ' | '.join(rows[0]) + ' |')
    lines.append('| ' + ' | '.join(['---'] * len(rows[0])) + ' |')
    # Data rows
    for row in rows[1:]:
        lines.append('| ' + ' | '.join(row) + ' |')

    return '\n'.join(lines)


def render_markdown(note, attachments=None):
    """Convert note attributed string to Markdown."""
    if note is None:
        return ''
    if attachments is None:
        attachments = {}

    # Paragraph style mappings
    # 0: title/h1, 1: heading/h2, 4: monospace/code
    # 100: dash list, 101: numbered list, 102: bullet list, 103: checkbox

    txt = note.get('string', '')
    if not txt:
        return ''

    lines = []
    current_line = []
    pos = 0

    for run in note.get('attributeRun', []):
        length = run['length']

        for frag in re.findall(r'\n|[^\n]+', txt[pos:pos + length]):
            if frag == '\n':
                # End of paragraph - flush current line
                line_text = ''.join(current_line)

                # Get paragraph style from this run
                pstyle = run.get('paragraphStyle', {}).get('style', -1)
                indent = run.get('paragraphStyle', {}).get('indent', 0)
                todo_info = run.get('paragraphStyle', {}).get('todo', {})
                is_done = todo_info.get('done', 0)

                indent_prefix = '  ' * indent

                if line_text.strip():
                    if pstyle == 0:  # Title/H1
                        lines.append(f'# {line_text}')
                    elif pstyle == 1:  # Heading/H2
                        lines.append(f'## {line_text}')
                    elif pstyle == 4:  # Monospace/code
                        lines.append(f'`{line_text}`')
                    elif pstyle == 100:  # Dash list
                        lines.append(f'{indent_prefix}- {line_text}')
                    elif pstyle == 101:  # Numbered list
                        lines.append(f'{indent_prefix}1. {line_text}')
                    elif pstyle == 102:  # Bullet list
                        lines.append(f'{indent_prefix}* {line_text}')
                    elif pstyle == 103:  # Checkbox
                        checkbox = '[x]' if is_done else '[ ]'
                        lines.append(f'{indent_prefix}- {checkbox} {line_text}')
                    else:
                        lines.append(line_text)
                else:
                    lines.append('')

                current_line = []
            else:
                # Apply inline formatting
                formatted = frag

                # Handle attachments
                info = run.get('attachmentInfo')
                if info:
                    attach_id = info.get('attachmentIdentifier')
                    attach = attachments.get(attach_id, {})
                    if 'markdown' in attach:
                        formatted = attach['markdown']
                    elif 'filename' in attach:
                        formatted = f'![{attach["filename"]}](images/{attach["dest_filename"]})'
                    else:
                        formatted = ''  # Skip unknown attachments

                # Apply text styling
                font_hints = run.get('fontHints', 0)
                strikethrough = run.get('strikethrough', 0)
                link = run.get('link')

                if font_hints & 1:  # Bold
                    formatted = f'**{formatted}**'
                if font_hints & 2:  # Italic
                    formatted = f'*{formatted}*'
                if strikethrough:
                    formatted = f'~~{formatted}~~'
                if link:
                    formatted = f'[{formatted}]({link})'

                current_line.append(formatted)

        pos += length

    # Flush any remaining content
    if current_line:
        lines.append(''.join(current_line))

    # Clean up multiple blank lines
    result = '\n'.join(lines)
    result = re.sub(r'\n{3,}', '\n\n', result)

    return result.strip()


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be used as a filename."""
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = re.sub(r'[_\s]+', '_', name)
    name = name.strip(' ._')
    if len(name) > 100:
        name = name[:100]
    return name or 'untitled'


def convert_notes(input_folder: Path, output_folder: Path) -> None:
    """Convert Apple Notes to Markdown files."""
    db_path = input_folder / 'NoteStore.sqlite'

    if not db_path.exists():
        raise FileNotFoundError(f'NoteStore.sqlite not found in {input_folder}')

    output_folder.mkdir(parents=True, exist_ok=True)
    images_dir = output_folder / 'images'
    images_dir.mkdir(exist_ok=True)

    db = sqlite3.connect(str(db_path))

    # Process attachments first
    attachments = {}
    copied_images = {}

    media_query = '''
        SELECT a.zidentifier, a.zmergeabledata, a.ztypeuti, b.zidentifier, b.zfilename, a.zurlstring, a.ztitle
        FROM ziccloudsyncingobject a
        LEFT JOIN ziccloudsyncingobject b ON a.zmedia = b.z_pk
        WHERE a.zcryptotag IS NULL AND a.ztypeuti IS NOT NULL
    '''

    for att_id, data, typ, media_id, fname, url, title in db.execute(media_query):
        if typ == 'com.apple.notes.table' and data:
            try:
                doc = parse(decompress(data, 47), SCHEMA_TABLE)
                table_md = render_table_markdown(doc['version'][0]['data'])
                attachments[att_id] = {'markdown': table_md}
            except Exception:
                pass
        elif typ == 'public.url':
            attachments[att_id] = {'markdown': f'[{title or url}]({url})'}
        elif fname and media_id:
            # Find and copy the media file
            for account_dir in (input_folder / 'Accounts').iterdir():
                if not account_dir.is_dir():
                    continue

                # Try Media folder
                media_dir = account_dir / 'Media' / media_id
                if media_dir.exists():
                    for version_dir in media_dir.iterdir():
                        if version_dir.is_dir():
                            file_path = version_dir / fname
                            if file_path.exists():
                                # Determine destination filename
                                if media_id in copied_images:
                                    dest_filename = copied_images[media_id]
                                else:
                                    dest_filename = fname
                                    dest_path = images_dir / dest_filename
                                    counter = 1
                                    while dest_path.exists():
                                        stem = Path(fname).stem
                                        suffix = Path(fname).suffix
                                        dest_filename = f'{stem}_{counter}{suffix}'
                                        dest_path = images_dir / dest_filename
                                        counter += 1

                                    shutil.copy2(file_path, dest_path)
                                    copied_images[media_id] = dest_filename

                                attachments[att_id] = {
                                    'filename': fname,
                                    'dest_filename': dest_filename
                                }
                                break
                    if att_id in attachments:
                        break

    # Get folders
    folders = {}
    folder_query = '''
        SELECT z_pk, ztitle2 FROM ziccloudsyncingobject WHERE ztitle2 IS NOT NULL
    '''
    for pk, title in db.execute(folder_query):
        folders[pk] = title

    # Process notes
    note_query = '''
        SELECT a.zidentifier, a.ztitle1, n.zdata, a.zfolder
        FROM zicnotedata n
        JOIN ziccloudsyncingobject a ON a.znotedata = n.z_pk
        WHERE n.zcryptotag IS NULL AND n.zdata IS NOT NULL
    '''

    seen_filenames = set()
    count = 0

    for note_id, title, data, folder_pk in db.execute(note_query):
        try:
            pb = decompress(data, 47)
            doc = parse(pb, SCHEMA_DOC)['version'][0]['data']
            content = render_markdown(doc, attachments)

            # Get folder name
            folder_name = folders.get(folder_pk, 'Notes')
            folder_name = sanitize_filename(folder_name)

            # Build markdown content
            title = title or 'Untitled'
            md_lines = []

            # Check if content starts with the title
            if content and content.startswith(title):
                content = content[len(title):].lstrip('\n')

            md_lines.append(f'# {title}')
            md_lines.append('')

            if content:
                md_lines.append(content)

            # Create folder and write file
            note_folder = output_folder / folder_name
            note_folder.mkdir(exist_ok=True)

            file_name = sanitize_filename(title)
            full_name = f'{folder_name}/{file_name}'

            counter = 1
            while full_name in seen_filenames:
                file_name = f'{sanitize_filename(title)}_{counter}'
                full_name = f'{folder_name}/{file_name}'
                counter += 1

            seen_filenames.add(full_name)
            file_path = note_folder / f'{file_name}.md'

            file_path.write_text('\n'.join(md_lines), encoding='utf-8')
            print(f'  Created: {folder_name}/{file_name}.md')
            count += 1

        except Exception as e:
            print(f'  Error processing note {note_id}: {e}')

    print('\nConversion complete!')
    print(f'  Notes: {count}')
    print(f'  Images copied: {len(copied_images)}')
    print(f'  Output: {output_folder}')

    db.close()


def main():
    parser = argparse.ArgumentParser(
        description='Convert Apple Notes to Markdown files'
    )
    parser.add_argument(
        'input_folder',
        type=Path,
        help='Input folder containing NoteStore.sqlite and Accounts/'
    )
    parser.add_argument(
        'output_folder',
        type=Path,
        help='Output folder for Markdown files and images'
    )

    args = parser.parse_args()
    convert_notes(args.input_folder, args.output_folder)


if __name__ == '__main__':
    main()
