#date: 2024-04-12T16:58:24Z
#url: https://api.github.com/gists/d277f667942f5a4c2d819dad95f24417
#owner: https://api.github.com/users/barronh

#!/usr/bin/env python
__doc__ = """
Overview
========

Module designed to make backups of scripts (.sh, .csh, .py), makefiles,
configuration files, and README files.

Contents
--------

backup : function
    Function to create a backup of scripts and other meta-data.

Examples
--------

.. code-block::python

    from backups import backup
    backup('/path/to/project')

.. code-block::bash

    python backup.py /path/to/project

"""
__version__ = '1.0.0'

_def_exclude = ['venv', '.ipynb_checkpoints']
_def_prefixes = 'readme makefile'.split()
_def_suffixes = '.py .sh .csh .json .md'.split()


def add_exclude(exdir):
    """
    Add directory to default excluded
    """
    if exdir not in _def_exclude:
        _def_exclude.extend(exdir)


def add_prefixes(pfx):
    """
    Add prefix to default included
    """
    if pfx not in _def_prefixes:
        _def_prefixes.extend(pfx)


def add_suffixes(sfx):
    """
    Add suffix to default included
    """
    if sfx not in _def_suffixes:
        _def_suffixes.extend(sfx)


def backup(
    projpath, tarpath=None, prefixes=None, suffixes=None, exclude=None,
    dryrun=False, verbose=0
):
    """
    Create a backup of scripts and metadata

    Arguments
    ---------
    projpath : str
        Path to project to backup.
    tarpath : str
        Path for tarfile to be saved (default: ~/backups/{projpath}.tar.gz)
    prefixes : list
        List of case-insensitive prefixes. Defaults to readme and makefile
    suffixes : list
        List of case-insensitive suffixes. Defaults to .py, .sh, .csh, .json,
        and .md
    exclude : list
        List of subdirectories to exclude. Defaults to venv and
        .ipynb_checkpoints
    dryrun : bool
        If True, print what it would do instead of doing it.
    verbose : int
        Level of verbosity (aka number of print statements)

    Returns
    -------
    None
    """
    from datetime import datetime
    import os
    import tarfile

    
    if exclude is None:
        exclude = _def_exclude
    elif len(exclude) == 0:
        exclude = _def_exclude
    if verbose > 0:
        print('Exclude:', exclude)
    if projpath.endswith('/'):
        projpath = projpath[:-1]

    if prefixes is None:
        prefixes = _def_prefixes

    if verbose > 0:
        print('Prefixes:', prefixes)

    if suffixes is None:
        suffixes = _def_suffixes

    if verbose > 0:
        print('Suffixes:', suffixes)

    if tarpath is None:
        if projpath.startswith('/'):
            tppath = projpath[1:]
        else:
            tppath = projpath
        today = datetime.now().strftime('%Y-%m-%d')
        tarpath = os.path.expanduser(
            os.path.join('~', 'backups', tppath)
        ) + '.' + today + '.tar.gz'
    tdir = os.path.dirname(tarpath)
    if verbose > 0:
        print('Output:', tarpath)
    if dryrun:
        print('mkdir -p ' + tdir)
        print('tar czf ' + tarpath + ' ...')
    else:
        if not os.path.exists(tdir):
            os.makedirs(tdir)
        tf = tarfile.open(tarpath, mode='w:gz')

    for root, dirs, files in os.walk(projpath):
        dirs[:] = [d for d in dirs if d not in exclude]
        if verbose > 0:
            print('working ' + root)
        if verbose > 1 and len(dirs) > 0:
            print('next ' + str(dirs))

        for f in files:
            lf = f.lower()
            if (
                any([lf.startswith(pfx) for pfx in prefixes])
                or any([lf.endswith(sfx) for sfx in suffixes])
            ):
                inpath = os.path.join(root, f)
                if dryrun:
                    print('  ' + inpath)
                else:
                    tf.add(inpath)

    if not dryrun:
        tf.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-v', '--verbose', action='count', default=0, help='Show more detail'
    )
    parser.add_argument(
        '-n', '--dry-run', dest='dryrun', action='store_true',
        default=False, help='Show, but do not do'
    )
    hstr = 'Exclude subdirectories (e.g., -x venv to exclude virtual'
    hstr += ' environment)'
    parser.add_argument(
        '-x', '--exclude', dest='exclude', action='append', default=[],
        help=hstr
    )
    parser.add_argument('projpath', help='Root path of project')
    hstr = 'Tar file output path'
    parser.add_argument('tarpath', nargs='?', default=None, help=hstr)
    parser.description = (
        'Backup Scripts, Makefiles, Configuration, and README Files'
    )
    parser.epilog = """
Example:

   $ ./backup.py /path/to/project

Description:

   All *.py, *.sh, *.csh, *.json, *.md, Makefile*, and README*
   will be put in a tar file at ~/backups/path/to/project.tar.gz
   file name testing is case insensitive.
"""
    args = parser.parse_args()
    backup(**vars(args))
