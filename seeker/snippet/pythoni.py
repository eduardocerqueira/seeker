#date: 2023-02-16T17:04:48Z
#url: https://api.github.com/gists/44af7676f8e1fb853e50afab0aa5f292
#owner: https://api.github.com/users/bnorick

import argparse
import code
import functools
import os
import pathlib
import shlex
import stat
import sys


class ArgumentParser(argparse.ArgumentParser):

    def error(self, message):
        self.print_help(sys.stderr)

    def exit(self, status=0, message=None):
        pass


def _argparse_export(pythoni, args):
    pythoni.export(args.path, *args.indices, command=args.command, overwrite=args.overwrite, all=args.all)


class Pythoni(code.InteractiveConsole):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._history = []

        self.command_parser = ArgumentParser()  # NOTE: requires python 3.9
        subparsers = self.command_parser.add_subparsers(title='command')
        export = subparsers.add_parser('export')
        export.add_argument('path', type=pathlib.Path)
        export.add_argument('indices', type=int, nargs='*', help='indices from history to export')
        export.add_argument('-c', '--command', type=str, required=False)
        export.add_argument('-o', '--overwrite', action='store_true')
        export.add_argument('-a', '--all', action='store_true')
        export.set_defaults(func=functools.partial(_argparse_export, self))

        sys.ps1 = '[0] >>> '

    def runsource(self, source, filename="<input>", symbol="single"):
        if source.startswith('%'):
            try:
                args = self.command_parser.parse_args(shlex.split(source[1:].strip()))
                args.func(args)
            except Exception as e:
                self.error(str(e))
            return False
        else:
            result = super().runsource(source, filename=filename, symbol=symbol)
            if not result:
                self._history.append('\n'.join(self.buffer))
            sys.ps1 = f'[{len(self._history)}] >>> '
            sys.ps2 = ' ' * sys.ps1.index('>') + '... '
            return result

    def _get_export_code(self, *indices, all=False):
        if indices and all:
            self.error(f'Both indices and all=True passed, only use one or the other.')
            return

        code = [self._history[index] for index in indices] if indices else self._history
        return '\n\n'.join(code)

    def export(self, path, *indices, command=None, overwrite=False, all=False):
        path = pathlib.Path(path)
        if path.exists() and not overwrite:
            self.error(f'path exists, pass overwrite=True to overwrite: {path}')
            return

        code = self._get_export_code(*indices, all=all)
        if command is None:
            with path.open('w', encoding='utf8') as f:
                f.write(code)

            print(f'Exported python script to {path}')
        else:
            code = 'import sys\nstdin = sys.stdin.readlines()\n\n' + code
            script = f'#!/usr/bin/env bash\nset -Eeuo pipefail\ntrap exit SIGINT SIGTERM ERR EXIT\nCODE={shlex.quote(code)}\n{command} | python -c "$CODE"'
            with path.open('w', encoding='utf8') as f:
                f.write(script)

            path.chmod(mode=path.stat().st_mode|stat.S_IRWXU)

            print(f'Exported executable bash script to {path}')



    def error(self, message):
        print(f'ERROR: {message}', file=sys.stderr)


redirected_out = not sys.stdout.isatty()
redirected_in = not sys.stdin.isatty()

if redirected_in:
    stdin = sys.stdin.readlines()
    print(f'stdin read to "stdin" variable\n')

if not redirected_out:
    # ref: https://github.com/python/cpython/issues/36029#issuecomment-1093968541
    _stdin = os.dup(0)
    os.close(0)
    tty = os.open("/dev/tty", os.O_RDONLY)
    assert tty == 0
    import readline
    import rlcompleter
    variables = globals().copy()
    variables.update(locals())
    readline.set_completer(rlcompleter.Completer(variables).complete)
    readline.parse_and_bind("tab: complete")
    pythoni = Pythoni(variables)
    variables.update(pythoni=pythoni)
    pythoni.interact()
