#date: 2025-01-14T16:49:06Z
#url: https://api.github.com/gists/8e4026159d311ec55513b5fbc921f5bd
#owner: https://api.github.com/users/ericsnowcurrently

from collections import namedtuple
import logging
import os
import os.path
import re
import shutil
import subprocess
import sys
import textwrap
import time


#SHORTCUT = False
SHORTCUT = True


VERBOSITY = 3  # logging.INFO

logger = logging.getLogger()


#######################################
# logging

def verbosity_to_loglevel(verbosity, *, maxlevel=logging.CRITICAL):
    return max(1,  # 0 disables it, so we use the next lowest.
               min(maxlevel,
                   maxlevel - verbosity * 10))


def loglevel_to_verbosity(level, *, maxlevel=logging.CRITICAL):
    return max(0, maxlevel - level) // 10


def configure_logger(logger, verbosity=VERBOSITY, *,
                     maxlevel=logging.CRITICAL,
                     ):
    level = verbosity_to_loglevel(verbosity)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        logger.addHandler(handler)


def get_verbosity():
    loglevel = logger.getEffectiveLevel()
    assert loglevel > 0, loglevel
    return loglevel_to_verbosity(loglevel)


#######################################
# OS utils

def resolve_cmd_capture(capture=None):
    if capture is None:
        return not (get_verbosity() < VERBOSITY)
    elif isinstance(capture, str) and capture.startswith('?'):
        if get_verbosity() < VERBOSITY:
            return None
        else:
            return capture[1:]
    else:
        return capture


def run_cmd(cmd, *args, capture=None, cwd=None, env=None, quiet=False):
    capture = resolve_cmd_capture(capture)

    kwargs = dict(
        cwd=cwd,
        env=env,
    )
    if capture is True:
        kwargs['stdout'] = subprocess.PIPE
        kwargs['stderr'] = subprocess.PIPE
        kwargs['text'] = True
    elif capture == 'stdout':
        kwargs['stdout'] = subprocess.PIPE
        kwargs['text'] = True
    elif capture == 'stderr':
        kwargs['stderr'] = subprocess.PIPE
        kwargs['text'] = True
    elif capture == '+stderr':
        kwargs['stdout'] = subprocess.PIPE
        kwargs['stderr'] = subprocess.STDOUT
        kwargs['text'] = True
    elif capture:
        raise ValueError(f'unsupported capture {capture!r}')

    if os.path.basename(cmd) == cmd:
        raise NotImplementedError(repr(cmd))
        cmd = shutil.which(cmd)
    argv = [cmd, *args]

    if not quiet:
        if env:
            logger.info(f'# running: {" ".join(argv)}  ({cwd or os.getcwd()}) {env}')
        else:
            logger.info(f'# running: {" ".join(argv)}  ({cwd or os.getcwd()})')
    proc = subprocess.run(argv, **kwargs)
    return proc.returncode, proc.stdout, proc.stderr


#######################################
# git

GIT = shutil.which('git')
CWD = os.getcwd()


def find_repo_root(*, fail=False):
    rc, stdout, stderr = run_cmd(
        GIT, 'rev-parse', '--show-toplevel',
        capture=True,
        quiet=True,
    )
    if rc != 0:
        if os.path.exists(os.path.join(CWD, 'cpython', '.git')):
            return os.path.join(CWD, 'cpython')
        if fail:
            sys.exit('ERROR: repo root not found')
        return None
    assert stdout.strip()
    return os.path.abspath(stdout.strip())


def resolve_repo_root(reporoot, *, fail=False):
    if reporoot == '':
        return os.get_cwd()
    elif reporoot is None:
        return find_repo_root(fail=fail)
    else:
        return os.path.abspath(reporoot)


#######################################
# C code

def render_header_file(name, body):
    name = os.path.basename(name).replace('.', ' ').split()[0]
    name = f'{name.upper()}_H'
    return os.linesep.join([
        f'#ifndef {name}',
        f'#define {name}',
        '',
        '',
        textwrap.dedent(body),
        '',
        f'#endif  /* {name} */',
        ''
    ])


def get_func_vartype(header, funcname):
    header = header.replace('\r\n', ' ').replace('\n', ' ')
    vartype, _ = header.split(f' {funcname}(')
    vartype = vartype.strip()
    if vartype.startswith('static '):
        vartype = vartype[7:].lstrip()
    return vartype


def find_function(text, name):
    starting = 0
    while True:
        starting = text.find(f'\n{name}(', starting)
        if starting < 0:
            return None, None, None, None

        body = text.find('\n{\n', starting)
        if body < 0:
            return None, None, None, None
        body += 1

        if ';' not in text[starting:body]:
            # It must have been a forward declaration.
            break
        starting += 1

    start = text.rfind('\n', 0, starting-1)
    if start < 0:
        return None, None, None, None
    start += 1
    assert start < starting, (starting, start, body)

    end = text.find('\n}\n', body)
    if end < 0:
        return None, None, None, None
    end += 2

    return (
        text[:start],
        text[start:body],
        text[body:end],
        text[end:],
    )


def find_text_section(text, start_marker, end_marker, *, start=0):
    assert start_marker.endswith('\n'), repr(start_marker)
    assert end_marker.endswith('\n'), repr(end_marker)

    if start < 0:
        end = len(text) + start + 1
        pos2 = text.rfind(end_marker, 0, end)
        if pos2 < 0:
            if text.rfind(start_marker, 0, end) >= 0:
                raise ValueError('missing end marker')
            return None, None, None
        pos1 = text.rfind(start_marker, 0, pos2)
        if pos1 < 0:
            raise ValueError('missing start marker')
    else:
        pos1 = text.find(start_marker, start)
        if pos1 < 0:
            if text.find(end_marker, start) >= 0:
                raise ValueError('missing start marker')
            return None, None, None
        pos2 = text.find(end_marker, pos1)
        if pos2 < 0:
            raise ValueError('missing end marker')
    start = text.rfind('\n', 0, pos1) + 1
    section = text[start:pos2+len(end_marker)]
    indent = text[start:pos1]
    assert section
    assert indent.strip() == '', repr(section)
    assert text[text.rfind('\n', pos2):pos2].strip() == '', repr(section)
    return start, section, indent


#######################################
# cpython

class CPythonConfig(namedtuple('CPythonConfig', 'debug freethreading pgo lto')):

    CACHEFILE = '../python-config.cache'

    prefix = None
    cflags = None
    ipv6 = None
    expat = None
    ffi = None
    optimizelevel = 0

    @classmethod
    def from_opts(cls, opts):
        if not opts:
            return cls(
                debug=True,
                freethreading=False,
                pgo=False,
                lto=False,
            )
        elif isinstance(opts, str):
            kwargs = {}
            for opt in opts.split():
                if opts == '--with-pydebug':
                    kwargs['debug'] = True
                elif opts == '--enable-optimizations':
                    kwargs['pgo'] = True
                elif opts == '--with-lto':
                    kwargs['lto'] = True
                elif opts == '--disable-gil':
                    kwargs['freethreading'] = True
                else:
                    raise ValueError(f'unsupported config opt {opt!r}')
        else:
            raise NotImplementedError(repr(opts))

    def __new__(cls, debug=False, freethreading=False, pgo=False, lto=False):
        return super().__new__(
            cls,
            debug=debug,
            freethreading=freethreading,
            pgo=pgo,
            lto=lto,
        )

    def as_configure_args(self, cachefile=''):
        argv = []

        if cachefile == '':
            cachefile = self.CACHEFILE
        if cachefile:
            argv.append(f'--cache-file={cachefile}')

        if self.prefix:
            argv.append(f'--prefix={self.prefix}')

        cflags = self.cflags
        if self.optimizelevel is not None:
            assert not cflags or not isinstance(cflags, str), cflags
            cflags = [*(cflags or ()), f'-O{self.optimizelevel}']
        if cflags:
            # XXX Support some generic CFlags type?
            if not isinstance(cflags, str):
                cflags = ' '.join(cflags)
            argv.append(f'CFLAGS={cflags}')

        if self.debug:
            argv.append('--with-pydebug')
        if self.freethreading:
            argv.append('--disable-gil')
        if self.ipv6:
            argv.append('--enable-ipv6')

        if self.expat == 'system':
            argv.append('--with-system-expat')
        elif self.expat:
            raise NotImplementedError(self.expat)

        if self.ffi == 'system':
            argv.append('--with-system-ffi')
        elif self.ffi:
            raise NotImplementedError(self.ffi)

        if self.pgo:
            argv.append('--enable-optimizations')
        if self.lto:
            argv.append('--with-lto')

        return argv


class CPythonBuildSpec(namedtuple('CPythonBuildSpec', 'builddir cfg')):

    MAKE = shutil.which('make')

    OUT_OF_TREE_CLEAN_FILES = [
        'python',
        'Programs/python.o',
        'Python/frozen_modules/importlib._bootstrap.h',
    ]

    @classmethod
    def add_cli(cls, parser):
        parser.add_argument('--build-dir')
        parser.add_argument('--build-config')
        parser.add_argument('--force-build', dest='forcebuild',
                            action='store_const', const=True)

        def handle_args(args, reporoot=None):
            ns = vars(args)

            builddir = ns.pop('build_dir')
            if not builddir:
                builddir = resolve_repo_root(reporoot)

            cfg = ns.pop('build_config')
            if cfg is not None:
                if not builddir:
                    parser.error('missing --build-dir')
                cfg = CPythonConfig.from_opts(cfg)

            if not builddir and not cfg:
                if args.forcebuild:
                    parser.error('missing --build-dir')
                return None

            # We leave args.forcebuild in place.

            return cls(builddir, cfg)
        return handle_args

    def verify(self, python, *, fail=True):
        # XXX Check it somehow.
        return True
        if fail:
            raise ValueError(f'build {self!r} doesn\'t match given python {python!r}')
        return False

    def prepare_for_build(self, reporoot=None, *extraheaders):
        reporoot = resolve_repo_root(reporoot, fail=True)
        if self.builddir == reporoot:
            return
        # else it is an out-of-tree build.

        os.makedirs(self.builddir, exist_ok=True)
        for relfile in extraheaders:
            src = os.path.join(reporoot, relfile)
            tgt = os.path.join(self.builddir, relfile)
            if os.path.exists(src):
                dirname = os.path.dirname(tgt)
                if dirname:
                    os.makedirs(dirname, exist_ok=True)
                shutil.copyfile(src, tgt)

        rc = -1
        if os.path.exists(os.path.join(reporoot, 'Makefile')):
            filename = os.path.join(reporoot, 'python')
            if os.path.exists(filename):
                os.unlink(filename)
            rc, _, _ = run_cmd(self.MAKE, 'clean', cwd=reporoot, capture=True)
        if rc != 0:
            for relfile in self.OUT_OF_TREE_CLEAN_FILES:
                filename = os.path.join(reporoot, relfile)
                if os.path.exists(filename):
                    os.unlink(filename)

    def get_configure_argv(self, reporoot=None, cachefile=None):
        reporoot = resolve_repo_root(reporoot, fail=True)
        cmd = os.path.join(reporoot, 'configure')
        return [cmd, *self.cfg.as_configure_args()]

    def get_build_argv(self):
        return [self.MAKE, '-j8', "CFLAGS='-Werror'"]


def build_cpython(spec=None, reporoot=None, *extraheaders, hide=None):
    logger.info('# building cpython')
    if spec is None:
        spec = CPythonBuildSpec()
    reporoot = resolve_repo_root(reporoot, fail=True)
    capture = '+stderr' if hide else '?+stderr'

    spec.prepare_for_build(reporoot, *extraheaders)

    if spec.cfg:
        rc, stdout, _ = run_cmd(
            *spec.get_configure_argv(reporoot),
            cwd=spec.builddir,
            capture=capture,
        )
    if not spec.cfg or rc == 0:
        if not os.path.exists(os.path.join(reporoot, 'Makefile')):
            raise Exception('build not configured')
        rc, stdout, _ = run_cmd(
            *spec.get_build_argv(),
            cwd=spec.builddir,
            capture='+stderr' if hide else '?+stderr',
        )
    if rc != 0:
        if not stdout:
            raise NotImplementedError(rc)
        elif resolve_cmd_capture(capture):
            print(stdout)
            raise NotImplementedError(rc)
        else:
            raise NotImplementedError(rc, stdout)


def run_python_text(python, script, *,
                    nosite=False,
                    capture=None,
                    env=None,
                    ):
    if capture is not False:
        capture = 'stdout' if capture else '?stdout'
    argv = [python]
    if nosite:
        argv.append('-S')
    argv.extend(['-c', script])

    start = time.time()
    rc, stdout, _ = run_cmd(*argv, capture=capture, env=env)
    pytotal = Duration(time.time() - start)
    assert rc == 0, rc
    return stdout, pytotal


#######################################
# other utils

def touch(filename):
    with open(filename, 'a') as outfile:
        outfile.write('')


class Timestamp(float):

    def __repr__(self):
        return f'{type(self).__name__}({float.__repr__(self)})'

    def __str__(self):
        return f'{self:.6f}s'

    def __sub__(self, other):
        return Duration(super().__sub__(other))

    def __rsub__(self, other):
        return Duration(super().__rsub__(other))


class Duration(float):

    def __repr__(self):
        return f'{type(self).__name__}({float.__repr__(self)})'

    def __str__(self):
        return self.render()

    def __add__(self, other):
        return type(self)(super().__add__(other))

    def __radd__(self, other):
        return type(self)(super().__radd__(other))

    def __sub__(self, other):
        return type(self)(super().__sub__(other))

    def __rsub__(self, other):
        return type(self)(super().__rsub__(other))

    def __mul__(self, other):
        return type(self)(super().__mul__(other))

    def __rmul__(self, other):
        return type(self)(super().__rmul__(other))

    def __div__(self, other):
        return type(self)(super().__div__(other))

    def __rdiv__(self, other):
        return type(self)(super().__rdiv__(other))

    def __truediv__(self, other):
        return type(self)(super().__truediv__(other))

    def __rtruediv__(self, other):
        return type(self)(super().__rtruediv__(other))

    def render(self, units=None):
        if units is None:
            if self < 0.001:
                units = 'ns'
            elif self < 1:
                units = 'ms'
            else:
                units = 's'

        if units == 's':
            return f'{self:,.6f}s'
        elif units == 'ms':
            return f'{self * 1_000:,.3f}ms'
        elif units == 'ns':
            return f'{int(self * 1000000):,}ns'
        else:
            raise ValueError(f'unsupported units {units!r}')


class SourceLocation(namedtuple('SourceLocation', 'filename lno func')):

    def __new__(cls, filename, lno, func):
        if not lno and not isinstance(lno, int):
            lno = None
        self = super().__new__(
            cls,
            filename or None,
            int(lno) if lno is not None else None,
            func or None,
        )
        return self


#######################################
# profile logging

class LogEntry(namedtuple('LogEntry', 'timestamp source context kind message')):

    class KINDS:
        ENTER = 'ENTER'
        PREFIX = 'PREFIX'  # MISC
        START = 'START'
        INFIX = 'INFIX'  # MISC
        STOP = 'STOP'
        POSTFIX = 'POSTFIX'  # MISC
        RETURN = 'RETURN'
    _KINDS = {v: v
              for k, v in vars(KINDS).items()
              if not k.startswith('_')}

    C_ENTER = 'func-entered'
    C_START = 'started'
    C_MISC = 'misc'
    C_STOP = 'stopped'
    C_RETURN = 'func-returning'

    # (timestamp, ctx, kind, msg)
    C_SIMPLE_FORMAT = '{}:{}:{}:{}'
    # (timestamp, file, line, func, ctx, kind, msg)
    C_FORMAT = '{} ({}:{}:{}) {}:{}:{}'

    REGEX = re.compile(r"""
        ^
        ( \d+ (?: \. \d+ )? )  # <timestamp>
        (?:
            (?:
                \s+
                \(
                ( [./\\\w] [^:]* )  # <file>
                :
                ( \d+ )  # <lno>
                :
                ( \w+ )  # <func>
                \)
                \s+
             )
            |
            :
         )
        ( \w[-\w]* )?  # <ctx>
        :
        ( \w[-\w]* )  # <kind>
        :
        ( .* )  # <msg>
        $
        """, re.VERBOSE)

    @classmethod
    def parse_all(cls, lines):
        if isinstance(lines, str):
            lines = lines.splitlines()
        active = []
        prev = None
        for line in lines:
            entry, implicit = cls._parse(line, active, prev)
            if entry is None:
                continue
            for entry in cls._expand_implicit(entry, implicit, prev):
                if entry.kind is cls.KINDS.ENTER:
                    active.append(entry)
                elif entry.kind is cls.KINDS.START:
                    active.append(entry)
                elif entry.kind is cls.KINDS.STOP:
                    assert active[-1].kind is cls.KINDS.START, (entry, active)
                    active.pop()
                elif entry.kind is cls.KINDS.RETURN:
                    assert active[-1].kind is cls.KINDS.ENTER, (entry, active)
                    active.pop()
                yield entry
            prev = entry
        assert not active, active

    @classmethod
    def _parse(cls, line, active=None, prev=None):
        line = line.strip()
        if not line or line.startswith('#'):
            return None, None

        m = cls.REGEX.match(line)
        if not m:
            raise ValueError(f'unsupported log entry {line!r}')
        (timestamp,
         filename, lno, func,
         ctx, ckind, msg,
         ) = m.groups()

        loc = SourceLocation(filename, lno, func) if filename else None

        orig_ctx = ctx
        ctx = cls._resolve_context(ctx, func, msg)
        if ctx is None:
            raise ValueError(f'missing context in {line!r}')

        kind, implicit = cls._resolve_kind(ckind or msg, ctx, active, prev, line)

        self = cls(timestamp, loc, ctx, kind, msg)
        if ctx == orig_ctx:
            self._raw = line

        return self, implicit

    @classmethod
    def _resolve_context(cls, context, func, msg):
        if context:
            return context
        elif func:
            return func
        elif msg and re.match(r'^[-\w]+$', msg):
            return msg
        else:
            return None

    @classmethod
    def _resolve_kind(cls, ckind, ctx, active, prev, line):
        if ckind == cls.C_ENTER:
            kind = cls.KINDS.ENTER
        elif ckind == cls.C_START:
            kind = cls.KINDS.START
        elif ckind == cls.C_MISC:
            kind = None
        elif ckind == cls.C_STOP:
            kind = cls.KINDS.STOP
        elif ckind == cls.C_RETURN:
            kind = cls.KINDS.RETURN
        else:
            raise ValueError(f'unsupported ckind {ckind!r} in {line!r}')

        if active is None:
            return kind, None
        assert not active or active[-1].kind is not cls.KINDS.RETURN, active
        assert not active or active[-1].kind is not cls.KINDS.STOP, active

        implicit = None
        if kind is cls.KINDS.ENTER:
            pass
        elif kind is cls.KINDS.START:
            if not active:
                implicit = cls.KINDS.ENTER
            elif active[-1].ctx != ctx:
                implicit = cls.KINDS.ENTER
            elif active[-1].kind is cls.KINDS.ENTER:
                pass
            elif active[-1].kind is cls.KINDS.START:
                raise ValueError(f'already started for {line!r}')
            else:
                raise NotImplementedError
        elif kind is None:
            if not active:
                implicit = cls.KINDS.ENTER
                kind = cls.KINDS.PREFIX
            elif active[-1].ctx != ctx:
                implicit = cls.KINDS.ENTER
                kind = cls.KINDS.PREFIX
            elif active[-1].kind is cls.KINDS.ENTER:
                if prev is not None and prev.kind is cls.KINDS.PREFIX:
                    kind = cls.KINDS.PREFIX
                else:
                    kind = cls.KINDS.POSTFIX
            elif active[-1].kind is cls.KINDS.START:
                kind = cls.KINDS.INFIX
            else:
                raise NotImplementedError
        elif kind is cls.KINDS.STOP:
            if not active:
                raise ValueError(f'missing start for {line!r}')
            elif active[-1].ctx != ctx:
                raise ValueError(f'missing start for {line!r}')
            elif active[-1].kind is cls.KINDS.ENTER:
                if prev is active[-1]:
                    implicit = cls.KINDS.START
                else:
                    raise ValueError(f'missing start for {line!r}')
            elif active[-1].kind is cls.KINDS.START:
                pass
            else:
                raise NotImplementedError
        elif kind is cls.KINDS.RETURN:
            if not active:
                raise ValueError(f'missing function enter for {line!r}')
            elif active[-1].ctx != ctx:
                raise ValueError(f'missing function enter for {line!r}')
            elif active[-1].kind is cls.KINDS.ENTER:
                if prev is active[-1]:
                    implicit = cls.KINDS.START
            elif active[-1].kind is cls.KINDS.START:
                implicit = cls.KINDS.STOP
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError(kind)

        return kind, implicit

    @classmethod
    def _expand_implicit(cls, entry, implicit, prev):
        if not implicit:
            yield entry
            return

        if entry.kind is cls.KINDS.ENTER:
            raise NotImplementedError('implicit not expected', entry)
        elif entry.kind is cls.KINDS.PREFIX:
            assert implcit is cls.KINDS.ENTER, (implicit, entry)
            yield entry._as_implicit(implicit)
        elif entry.kind is cls.KINDS.START:
            assert implicit is cls.KINDS.ENTER, (implicit, entry)
            yield entry._as_implicit(implicit)
        elif entry.kind is cls.KINDS.INFIX:
            raise NotImplementedError('implicit not expected', entry)
        elif entry.kind is cls.KINDS.STOP:
            raise NotImplementedError('implicit not expected', entry)
        elif entry.kind is cls.KINDS.POSTFIX:
            raise NotImplementedError('implicit not expected', entry)
        elif entry.kind is cls.KINDS.RETURN:
            if implicit is cls.KINDS.START:
                assert prev.ctx == entry.ctx, (prev, entry)
                assert prev.kind is cls.KINDS.ENTER, (prev, entry)
                yield prev._as_implicit(implicit)
                yield entry._as_implicit(cls.KINDS.STOP)
            else:
                assert implicit is cls.KINDS.STOP, (implicit, entry)
                yield entry._as_implicit(implicit)
        else:
            NotImplementedError(entry)

        yield entry

    def __new__(cls, timestamp, source, context, kind, message):
        self = super().__new__(
            cls,
            Timestamp(timestamp) if timestamp is not None else None,
            source or None,
            context or None,
            kind or None,
            message,
        )
        return self

    def __str__(self):
        try:
            raw = self._raw
        except AttributeError:
            raw = None
        if raw is not None:
            return raw

        cls = type(self)
        args = [
            str(self.timestamp) if self.timestamp is not None else '',
        ]

        if self.source is not None:
            fmt = cls.C_FORMAT
            assert self.source.filename, self.source
            assert self.source.lno is not None, self.source
            assert self.source.func, self.source
            args.extend(self.source)
        else:
            fmt = cls.C_SIMPLE_FORMAT

        if self.kind is cls.KINDS.ENTER:
            kind = cls.C_ENTER
        elif self.kind is cls.KINDS.PREFIX:
            kind = cls.C_MISC
        elif self.kind is cls.KINDS.START:
            kind = cls.C_START
        elif self.kind is cls.KINDS.INFIX:
            kind = cls.C_MISC
        elif self.kind is cls.KINDS.STOP:
            kind = cls.C_STOP
        elif self.kind is cls.KINDS.POSTFIX:
            kind = cls.C_MISC
        elif self.kind is cls.KINDS.RETURN:
            kind = cls.C_RETURN
        else:
            raise NotImplementedError(self)

        args.extend([
            self.ctx or '',
            kind or '',
            self.msg or '',
        ])
        return fmt.format(*args)

    @property
    def location(self):
        return self.source

    @property
    def loc(self):
        return self.source

    @property
    def ctx(self):
        return self.context

    @property
    def msg(self):
        return self.message

    def _as_implicit(self, implicit):
        implicit = self._replace(kind=implicit, message=None)
        implicit._raw = None
        return implicit


class LogEntryInterval(namedtuple('LogEntryInterval', 'start stop')):

    @property
    def duration(self):
        return self.stop.timestamp - self.start.timestamp


class ProfilerHelperFile:

    FILE = None

    @classmethod
    def from_raw(cls, raw):
        if isinstance(raw, cls):
            return cls
        elif isinstance(raw, str):
            return cls(raw)
        elif raw is None:
            return cls()
        else:
            raise TypeError(f'unsupported raw value {raw!r}')

    def __init__(self, filename=None):
        if not filename:
            filename = self.FILE
            if not filename:
                raise ValueError('missing filename')
        self._filename = filename
        self._relative = not os.path.isabs(filename)

    def __repr__(self):
        return f'{type(self).__name__}(filename={self._filename!r})'

    def __str__(self):
        return self._filename

    @property
    def filename(self):
        return self._filename

    @property
    def include(self):
        if not self._relative:
            raise Exception('must be relative')
        if not self._filename.startswith('Include'):
            return self._filename
        if 'pycore_' in self._filename:
            return os.path.basename(self._filename)
        _, _, name = self._filename.partition('Include')
        return name[1:]

    @property
    def isrelative(self):
        return self._relative

    def as_relative(self, reporoot):
        if self._relative:
            return self
        if not self._filename.startswith(reporoot + '/'):
            raise NotImplementedError(reporoot)
        relfile = os.path.relpath(self._filename, reporoot)
        cls = type(self)
        return cls(relfile)

    def as_absolute(self, reporoot=None):
        if not self._relative:
            if reporoot:
                if not self._filename.startswith(reporoot + '/'):
                    raise NotImplementedError(reporoot)
            return self
        reporoot = resolve_repo_root(reporoot)
        filename = os.path.join(reporoot, self._filename)
        filename = os.path.abspath(filename)
        cls = type(self)
        return cls(filename)

    def render(self, *includes):
        if not includes:
            return self._render()
        lines = [f'#include "{incl}"' for incl in includes]
        lines.append('')
        lines.append(self._render())
        return '\n'.join(lines)

    def write(self, *includes, force=False):
        if self._relative:
            raise NotImplementedError

        text = self.render(*includes)
        if not text.endswith('\n'):
            text += '\n'
        actual = self._read()
        if actual is not None:
            before, actual = self._split(actual)
            if not force and text == actual:
                return False
        else:
            before = None

        if before and not actual:
            with open(self._filename, 'a') as outfile:
                outfile.write(text)
        else:
            with open(self._filename, 'w') as outfile:
                if before:
                    outfile.write(before)
                outfile.write(text)
        return True

    def _render(self):
        raise NotImplementedError

    def _read(self):
        try:
            with open(self._filename) as infile:
                text = infile.read()
        except FileNotFoundError:
            text = None
        return text

    def _split(self, text):
        return None, text

#    def _check_dirty(self, expected, actual=None):
#        if actual is None:
#            actual = self._read()
#        if actual is None:
#            # not found
#            return True
#        return actual != expected


class ProfilerHelperInstalled(ProfilerHelperFile):
    # It gets installed at the end.

    MARKER = f'added by {__file__}'

    def __init__(self, filename=None, helper=None, marker=None):
        if not marker:
            marker = self.MARKER
        super().__init__(filename)
        self._helper = helper
        self._marker = marker
        self._start_marker = f'// {marker}'
        self._end_marker = None

    @property
    def helper(self):
        return self._helper

    @property
    def marker(self):
        return self._marker

    def render(self):
        return super().render()

    def write(self, force=False):
        return super().write(force=force)

    def _render(self):
        if self._helper is None:
            raise ValueError(f'helper not set')
        if self._helper.isrelative:
            include = self._helper.include
        else:
            dirname, basename = os.path.split(self._helper.filename)
            include = os.path.join(
                os.path.relpath(dirname, os.path.dirname(self._filename)),
                basename,
            )
        text = '\n'.join([
            '',
            self._start_marker,
            f'#include "{include}"',
            '',
        ])
        return text

    def _split(self, text):
        pos = text.rfind(f'\n{self._start_marker}')
        if pos < 0:
            return text, None
        else:
            return text[:pos], text[pos:]


class ProfilerHelpersHeader(ProfilerHelperFile):

    FILE = os.path.join('Include', '_profiling_helpers.h')

    LOGFILE = 'profiler.log'

    PREAMBLE = rf"""
        #define _PyProfiler_SIMPLE_FMT \
                "{LogEntry.C_SIMPLE_FORMAT.format('%f', '%s', '%s', '%s')}\n"
        #define _PyProfiler_FMT \
                "{LogEntry.C_FORMAT.format('%f', '%s', '%d', '%s', '%s', '%s', '%s')}\n"
        
        /* event kinds */
        #define _PyProfiler_EVENT_ENTER "{LogEntry.C_ENTER}"
        #define _PyProfiler_EVENT_START "{LogEntry.C_START}"
        #define _PyProfiler_EVENT_MISC "{LogEntry.C_MISC}"
        #define _PyProfiler_EVENT_STOP "{LogEntry.C_STOP}"
        #define _PyProfiler_EVENT_RETURN "{LogEntry.C_RETURN}"

        #define _PyProfiler_DEFAULT_LOGFILE "{{defaultlogfile}}"
        """[1:-1]

    TEXT = r"""
        PyAPI_FUNC(FILE *) _PyProfiler_EnsureLogfile(void);

        typedef struct {
            struct {
                const char *file;
                int line;
                const char *func;
            } source;
            PyTime_t time;
            const char *ctx;
            const char *kind;
            const char *msg;
        } _PyProfiler_event_t;

        static inline void
        _PyProfiler_write_event(FILE *outfile, _PyProfiler_event_t *evt)
        {
            fprintf(outfile, _PyProfiler_SIMPLE_FMT,
                    PyTime_AsSecondsDouble(evt->time), evt->ctx, evt->kind, evt->msg);
        }

        static inline void
        _PyProfiler_write_event_full(FILE *outfile, _PyProfiler_event_t *evt)
        {
            fprintf(outfile, _PyProfiler_FMT,
                    PyTime_AsSecondsDouble(evt->time),
                    evt->source.file, evt->source.line, evt->source.func,
                    evt->ctx, evt->kind, evt->msg);
        }

        #define _PyProfiler_WRITE_EVENT(OUTFILE, CTX, KIND, MSG) \
            do { \
                _PyProfiler_event_t event = { \
                    .ctx = CTX, \
                    .kind = KIND, \
                    .msg = MSG, \
                }; \
                /* We could also use PyTime_TimeRaw(). */ \
                (void)PyTime_PerfCounterRaw(&event.time); \
                _PyProfiler_write_event(OUTFILE, &event); \
            } while (0)

        /* copied from pycore_initconfig.h */
        #ifdef _MSC_VER
        /* Visual Studio 2015 doesn't implement C99 __func__ in C */
        #  define _PyProfiler_GET_FUNC() __FUNCTION__
        #else
        #  define _PyProfiler_GET_FUNC() __func__
        #endif

        #define _PyProfiler_WRITE_FULL_EVENT(OUTFILE, CTX, KIND, MSG) \
            do { \
                _PyProfiler_event_t event = { \
                    .source = { \
                        .file = __FILE__, \
                        .line = __LINE__, \
                        .func = _PyProfiler_GET_FUNC(), \
                    }, \
                    .ctx = CTX, \
                    .kind = KIND, \
                    .msg = MSG, \
                }; \
                /* We could also use PyTime_TimeRaw(). */ \
                (void)PyTime_PerfCounterRaw(&event.time); \
                _PyProfiler_write_event_full(OUTFILE, &event); \
            } while (0)

        #ifdef PY_PROFILER_FORCE_STDOUT
        # define _PyProfiler_ADD_ENTER() \
            _PyProfiler_WRITE_FULL_EVENT(stdout, "", _PyProfiler_EVENT_ENTER, "")
        # define _PyProfiler_ADD_START(ctx) \
            _PyProfiler_WRITE_FULL_EVENT(stdout, ctx, _PyProfiler_EVENT_START, "")
        # define _PyProfiler_ADD_OTHER(ctx, msg) \
            _PyProfiler_WRITE_FULL_EVENT(stdout, ctx, _PyProfiler_EVENT_MISC, msg)
        # define _PyProfiler_ADD_STOP(ctx) \
            _PyProfiler_WRITE_FULL_EVENT(stdout, ctx, _PyProfiler_EVENT_STOP, "")
        # define _PyProfiler_ADD_RETURN() \
            _PyProfiler_WRITE_FULL_EVENT(stdout, "", _PyProfiler_EVENT_RETURN, "")
        #else
        # define _PyProfiler_ADD_ENTER() \
            static FILE *logfile = NULL; \
            if (logfile == NULL) { \
                logfile = _PyProfiler_EnsureLogfile(); \
            } \
            _PyProfiler_WRITE_FULL_EVENT(logfile, "", _PyProfiler_EVENT_ENTER, "");
        # define _PyProfiler_ADD_START(ctx) \
            _PyProfiler_WRITE_FULL_EVENT(logfile, ctx, _PyProfiler_EVENT_START, "");
        # define _PyProfiler_ADD_OTHER(ctx, msg) \
            _PyProfiler_WRITE_FULL_EVENT(logfile, ctx, _PyProfiler_EVENT_MISC, msg);
        # define _PyProfiler_ADD_STOP(ctx) \
            _PyProfiler_WRITE_FULL_EVENT(logfile, ctx, _PyProfiler_EVENT_STOP, "");
        # define _PyProfiler_ADD_RETURN() \
            _PyProfiler_WRITE_FULL_EVENT(logfile, "", _PyProfiler_EVENT_RETURN, "");
        #endif
        """[1:-1]

    def render(self, *includes):
        if includes:
            raise NotImplementedError(includes)
        return super().render()

    def write(self, force=False):
        return super().write(force=force)

    def _render(self):
        cls = type(self)
        text = '\n'.join([
            cls.PREAMBLE.format(defaultlogfile=cls.LOGFILE),
            cls.TEXT,
        ])
        text = render_header_file(self._filename, text)
        return text


class ProfilerHelpersImpl(ProfilerHelperFile):

    FILE = os.path.join('Python', '_profiling_helpers.c')

    TEXT = textwrap.dedent(r"""
        FILE *
        _PyProfiler_EnsureLogfile(void)
        {
            static FILE *logfile = NULL;
            if (logfile != NULL) {
                return logfile;
            }
            // We do not worry about encodings for the filename or file.
            const char *filename = getenv("PROFILING_LOGFILE");
        #ifdef PROFILING_LOGFILE
            if (filename == NULL) {
                filename = ##PROFILING_LOGFILE;
            }
        #endif
            if (filename == NULL || strlen(filename) == 0) {
                filename = _PyProfiler_DEFAULT_LOGFILE;
                if (filename == NULL) {
                    filename = "-";
                }
                else if (strlen(filename) == 0) {
                    filename = "profile.log";
                }
                assert(filename != NULL);
            }
            if (strcmp(filename, "-") == 0) {
                logfile = stdout;
            }
            else {
                logfile = fopen(filename, "w");
            }
            if (logfile == NULL) {
                PyObject *filenameobj = PyUnicode_FromString(filename);
                PyErr_SetFromErrnoWithFilenameObject(PyExc_OSError, filenameobj);
                Py_DECREF(filenameobj);
            }
            return logfile;
        }
        """[1:-1])

    def _render(self):
        return self.TEXT


class ProfilerHelperFiles(namedtuple('ProfilerHelperFiles',
                                     'header impl implincluded')):

    IMPLINCLUDED = os.path.join('Python', 'pylifecycle.c')

    @classmethod
    def _resolve_implincluded(cls, implincluded, impl):
        if implincluded is None:
            implincluded = cls.IMPLINCLUDED
        if not implincluded:
            return None
        resolved = ProfilerHelperInstalled.from_raw(implincluded)
        if resolved.helper is None:
            resolved._helper = impl
        elif resolved.helper != impl:
            raise ValueError(f'implincluded does not match impl')
        return resolved

    @classmethod
    def _resolve_relative(cls, header, impl, implincluded):
        if header.isrelative:
            if not impl.isrelative:
                raise ValueError(f'expected relative path for impl, got {impl!r}')
            if implincluded and not implincluded.isrelative:
                raise ValueError(f'expected relative path for implincluded, got {implincluded!r}')
            return True
        else:
            if impl.isrelative:
                raise ValueError(f'expected absolute path for impl, got {impl!r}')
            if implincluded and implincluded.isrelative:
                raise ValueError(f'expected absolute path for implincluded, got {implincluded!r}')
            return False

    def __new__(cls, header=None, impl=None, implincluded=None):
        header = ProfilerHelpersHeader.from_raw(header)
        impl = ProfilerHelpersImpl.from_raw(impl)
        implincluded = cls._resolve_implincluded(implincluded, impl)

        relative = cls._resolve_relative(header, impl, implincluded)

        self = super().__new__(cls, header, impl, implincluded)
        self._isrelative = relative
        return self

    @property
    def isrelative(self):
        return self._isrelative

    def as_absolute(self, reporoot=None):
        if not self._isrelative:
            return self
        cls = type(self)
        copied = cls(
            os.path.join(reporoot, self.header.filename),
            os.path.join(reporoot, self.impl.filename),
            (os.path.join(reporoot, self.implincluded.filename)
             if self.implincluded
             else None),
        )
        copied._isrelative = True
        return copied

    def as_relative(self, reporoot):
        if self._isrelative:
            return self
        cls = type(self)
        copied = cls(
            os.path.relpath(self.header.filename, reporoot),
            os.path.relpath(self.impl.filename, reporoot),
            (os.path.relpath(self.implincluded.filename, reporoot)
             if self.implincluded
             else None),
        )
        copied._isrelative = False
        return copied

    def write(self, reporoot=None, *, force=False):
        reporoot = resolve_repo_root(reporoot)
        files = self.as_absolute(reporoot)
        relfiles = self.as_relative(reporoot)

        dirty1 = files.header.write(force=force)

        include = relfiles.header.include
        dirty2 = files.impl.write(include, force=force)

        return dirty1 or dirty2

    def install(self, reporoot=None, *, force=False):
        reporoot = resolve_repo_root(reporoot)
        files = self.as_absolute(reporoot)
        return files.implincluded.write(force=force)

#    def write(self, reporoot=None, *, force=False):
#        # First prep the header file.
#        headers_text = '\n'.join([
#            cls.HEADER_PREAMBLE.format(defaultlogfile=cls.LOGFILE),
#            cls.HEADER_TEXT,
#        ])
#        headers_text = render_header_file(self.header, headers_text)
#        if os.path.exists(self.header) and not force:
#            with open(self.header) as infile:
#                actual = infile.read()
#            if headers_text == actual:
#                headers_text = None
#
#        # Then prep the impl file.
#        include = os.path.relpath(headerfile, reporoot)
#        impl_text = '\n'.join([
#            f'#include "{include}"',
#            '',
#            cls.IMPL_TEXT,
#            '',
#        ])
#        if os.path.exists(implfile):
#            with open(implfile) as infile:
#                actual = infile.read()
#            if impl_text == actual:
#                impl_text = None
#
#    def install(self, reporoot=None, *, force=False):
#        # Then prep the file that includes the impl, if applicable.
#        if implincluded:
#            marker = f'// added by {__file__}'
#            include = os.path.relpath(implfile, implincluded)
#            implincluded_text = '\n'.join([
#                '',
#                marker,
#                f'#include "{}"',
#                '',
#            ])
#            if os.path.exists(implincluded):
#                with open(implincluded) as infile:
#                    actual = infile.read()
#                if actual.endswith(implincluded_text):
#                    if force:
#                        pos = actual.rfind(f'\n{marker)')
#                        implincluded_actual = actual[:pos]
#                    else:
#                        implincluded_text = None
#
#        # Finally, write out the helpers.
#        dirty = False
#        if headers_text:
#            logger.info(f'# Adding helpers in {headerfile}')
#            with open(headerfile, 'w') as outfile:
#                outfile.write(headers_text)
#            dirty = True
#        if impl_text:
#            logger.info(f'# Adding helpers in {implfile}')
#            with open(implfile, 'w') as outfile:
#                outfile.write(impl_text)
#            dirty = True
#        if implincluded and implincluded_text:
#            logger.info(f'# Installing helpers in {implincluded}')
#            if implincluded_actual:
#                with open(implincluded, 'a') as outfile:
#                    outfile.write(implincluded_text)
#            else:
#                with open(implincluded, 'w') as outfile:
#                    outfile.write(implincluded_actual)
#                    outfile.write(implincluded_text)
#            dirty = True
#
#        return dirty


#        if logfile is None or logfile == '-':
#            parts.append(cls.STDOUT_LOGGING_CODE)
#        else:
#            if not logfile:
#                # XXX CWD?
#                logfile = cls.LOGFILE
#            elif logfile.endswith('/') or os.path.isdir(logfile):
#                logfile = os.path.join(logfile[:-1], cls.LOGFILE)
#            os.makedirs(os.path.dirname(logfile), exist_ok=True)
#            parts.append(cls.FILE_LOGGING_CODE.format(logfile=logfile))
#        text = '\n'.join(parts)
#        text = render_header_file(filename, text)
#
#        if os.path.exists(filename) and not force:
#            # XXX Match logfile?
#            return False, logfile
#
#        logger.info(f'# Adding helpers in {filename}')
#        with open(filename, 'w') as outfile:
#            outfile.write(text)
#        return True, logfile
    

class ProfilerHelpers:

    @classmethod
    def from_files(cls, header=None, impl=None, implincluded=None, *,
                   reporoot=None,
                   ):
        files = ProfilerHelperFiles(header, impl, implincluded)
        self = cls(files)
        if reporoot:
            self._relfiles = files.as_relative(reporoot)
        return self

    @classmethod
    def from_relfiles(cls, reporoot, header=None, impl=None, implincluded=None):
        relfiles = ProfilerHelperFiles(header, impl, implincluded)
        files = relfiles.as_absolute(reporoot)
        self = cls(files)
        self._relfiles = relfiles
        return self

    def __init__(self, files):
        self._files = files

    @property
    def files(self):
        return self._files

    @property
    def relfiles(self):
        try:
            return self._relfiles
        except AttributeError:
            self._relfiles = self._files.as_relfiles()
            return self._relfiles

#    def add(self, reporoot=None, *, force=False):
#        if self._relative:
#            raise NotImplementedError
#        reporoot = resolve_repo_root(reporoot)
#        filename = self._files.header.filename
#        filename = os.path.join(reporoot, relfile)

#        dirty = self.files.write(force=force)
#        return dirty

#        # Now install them.
#
#        targetfile = os.path.join(reporoot, 'Include', 'Python.h')
#        with open(targetfile) as infile:
#            text = infile.read()
#        marker = f'profiling helpers added by {__file__}'
#        text = cls._install_helpers(text, marker, filename,
#                                    addapi=False, bottom=True)
#        if text is None:
#            if dirty:
#                touch(targetfile)
#            return dirty
#        else:
#            logger.info(f'# Installing helpers in {targetfile}')
#            with open(targetfile, 'w') as outfile:
#                outfile.write(text)
#            dirty = True
#
#        return dirty

    def wrap_func(self, filename, funcname, install=True):
        with open(filename) as infile:
            text = infile.read()

        dirty = False
        marker = f'profiling added by {__file__}'
        start_marker = f'// START: {marker}'
        end_marker = f'// END: {marker}'

        before, header, body, after = find_function(text, funcname)
        if body is None:
            raise NotImplementedError(filename, funcname)
        assert before is not None and header is not None and after is not None, (before, header, body, after)

        lines = body.splitlines()

        # Add the enter logging.
        wrapped = []
        for line in lines:
            if not wrapped:
                assert line.endswith('{'), repr(line)
                wrapped.append(line)
            else:
                if line.lstrip() != start_marker:
                    indent = line[:len(line) - len(line.lstrip())]
                    wrapped.append(f'{indent}{start_marker}')
                    wrapped.append(f'{indent}_PyProfiler_ADD_ENTER();')
                    wrapped.append(f'{indent}{end_marker}')
                    dirty = True
                break
        else:
            raise NotImplementedError(body)

        # Add the return logging.
        lines = iter(lines)
        prev3 = prev2 = ''
        prev1 = next(lines)  # the opening "{"
        returnvoid = True
        end = False
        for line in lines:
            assert not end, repr(line)
            if line == '}':
                if returnvoid:
                    if prev1.lstrip() != end_marker:
                        wrapped.append(f'    {start_marker}')
                        wrapped.append(f'    _PyProfiler_ADD_RETURN();')
                        wrapped.append(f'    {end_marker}')
                    else:
                        assert prev3.lstrip() == start_marker, (funcname, prev3)
                else:
                    assert prev1.lstrip() != end_marker, (funcname, prev1)
                end = True
            elif line.lstrip().startswith('return '):
                returnvoid = False
                if prev1.lstrip() != end_marker:
                    indent = line[:len(line) - len(line.lstrip())]
                    if '(' in line:
                        _, expr = line.split('return ')
                        while not expr.endswith(';'):
                            raise NotImplementedError
                        expr = expr[:-1]
                        vartype = get_func_vartype(header, funcname)

                        wrapped.append(indent + '{')
                        wrapped.append(f'{indent}    {vartype} res = {expr};')
                        wrapped.append(f'{indent}    {start_marker}')
                        wrapped.append(f'{indent}    _PyProfiler_ADD_RETURN();')
                        wrapped.append(f'{indent}    {end_marker}')
                        wrapped.append(f'{indent}    return res;')
                        line = indent + '}'
                    else:
                        wrapped.append(f'{indent}{start_marker}')
                        wrapped.append(f'{indent}_PyProfiler_ADD_RETURN();')
                        wrapped.append(f'{indent}{end_marker}')
                    dirty = True
                else:
                    assert prev3.lstrip() == start_marker, (funcname, prev3, line)
            elif line.lstrip() == 'return;':
                if prev1.lstrip() != end_marker:
                    indent = line[:len(line) - len(line.lstrip())]
                    wrapped.append(f'{indent}{start_marker}')
                    wrapped.append(f'{indent}_PyProfiler_ADD_RETURN();')
                    wrapped.append(f'{indent}{end_marker}')
                    dirty = True
                else:
                    assert prev3.lstrip() == start_marker, (funcname, prev3, line)
            wrapped.append(line)
            prev3 = prev2
            prev2 = prev1
            prev1 = line
        body = '\n'.join(wrapped)

        if dirty:
            if install:
                installed = self._install(before, marker)
                if installed is not None:
                    before = installed

            logger.info(f'# Wrapping {funcname}() in {filename}')
            text = f'{before}{header}{body}{after}'
            with open(filename, 'w') as outfile:
                outfile.write(text)
        return dirty

    def _install(self, text, marker, *,
                 addapi=True,
                 top=None,
                 bottom=False,
                 ):
        if top is None:
            top = not bottom
        else:
            assert top or bottom, (top, bottom)

        assert marker.rstrip() == marker, repr(marker)
        start_marker = f'// START: {marker}\n'
        end_marker = f'// END: {marker}\n'
        helpers_h = self.relfiles.header.include
        lines = [
            start_marker.rstrip(),
            *(('#include "Python.h"',) if addapi else ()),
            f'#include "{os.path.basename(helpers_h)}"',
            end_marker.rstrip(),
            '',
        ]
        expected = '\n'.join(lines)

        before = after = ''
        body = text

        if top:
            before = expected + '\n'
            (pos, section, indent,
             ) = find_text_section(text, start_marker, end_marker, start=0)
            if section:
                assert pos == 0, (pos, section)
                assert indent == '', (indent, section)
                if section == expected:
                    assert text.startswith(before), f'\n{before!r}\n---\n{section!r}\n---\n{text[:len(section)]!r}'
                    before = ''
                else:
                    assert not text.startswith(before), f'\n{before!r}\n---\n{section!r}\n---\n{text[:len(section)]!r}'
                    body = text[len(section):]
            else:
                assert not text.startswith(before), f'\n{before!r}\n---\n{section!r}\n---\n{text[:len(before)]!r}'

        if bottom:
            after = '\n' + expected
            (pos, section, indent,
             ) = find_text_section(text, start_marker, end_marker, start=-1)
            if section:
                assert pos + len(section) == len(text), (pos, section)
                assert indent == '', (indent, section)
                if section == expected:
                    assert text.endswith(after), f'\n{after!r}\n---\n{section!r}\n---\n{text[-len(section):]!r}'
                    after = ''
                else:
                    assert not text.endswith(after), f'\n{after!r}\n---\n{section!r}\n---\n{text[-len(section):]!r}'
                    body = text[:pos]
            else:
                assert not text.endswith(after), f'\n{after!r}\n---\n{section!r}\n---\n{text[-len(after):]!r}'

        if not before:
            if not after:
                return None
            return body + after
        elif not after:
            return before + body
        else:
            return before + body + after


#######################################
# analysis

class ProfileNode(namedtuple('ProfileNode', 'context entries')):

    FORMATS = ['summary', 'tree', 'raw']

    @classmethod
    def from_entries(cls, entries):
        entries = iter(entries)
        node = cls._resolve_tree(entries)
        while node is not None:
            yield node
            node = cls._resolve_tree(entries)

    @classmethod
    def _resolve_tree(cls, entries, entry=None):
        if entry is None:
            try:
                entry = next(entries)
            except StopIteration:
                return None
        assert entry.kind is LogEntry.KINDS.ENTER, repr(entry)
        if entry.source is not None:
            filename = entry.source.filename
            func = entry.source.func
        else:
            filename = func = None

        ctx = entry.ctx
        if not ctx:
            raise ValueError(f'missing context in {entry!r}')

        node = [entry]
        for entry in entries:
            if entry.ctx != ctx:
                subtree = cls._resolve_tree(entries, entry)
                node.append(subtree)
            else:
                assert filename is None or (entry.source.filename == filename), entry
                if entry.kind is LogEntry.KINDS.RETURN:
                    node.append(entry)
                    break
                else:
                    assert entry.kind is not LogEntry.KINDS.ENTER, entry
                    node.append(entry)

        self = cls(ctx, tuple(node))
        return self

    @property
    def duration(self):
        return self.end - self.start

    @property
    def own_duration(self):
        children = self.other_duration
        if children is None:
            return self.duration
        return self.duration - children

    @property
    def other_duration(self):
        children = None
        for entry in self.entries:
            if isinstance(entry, ProfileNode):
                if children is None:
                    children = 0.0
                children += entry.duration
        return children if children is not None else None

    @property
    def start(self):
        return self.entries[0].timestamp

    @property
    def end(self):
        return self.entries[-1].timestamp

    @property
    def intervals(self):
        raise NotImplementedError

    @property
    def children(self):
        return self.entries[1:-1]

    def flatten(self):
        for entry in self.entries:
            if isinstance(entry, ProfileNode):
                yield from entry.flatten()
            else:
                yield entry

    def flatten_nodes(self):
        yield self
        for node in self.entries:
            if isinstance(node, ProfileNode):
                yield from node.flatten_nodes()

    def summarize(self):
        return ProfileNodeSummary.with_children(
            self.entries[0].source.filename,
            self.entries[0].source.func,
            self.context,
            self.duration,
            self.own_duration,
            self.other_duration,
        )

    def collect_summary(self, collected):
        for node in self.flatten_nodes():
            # XXX Collate contexts?
            s = node.summarize()
            key = (s.filename, s.func)
            try:
                count, total, children = collected[key]
            except KeyError:
                count = 0
                total = 0.0
                children = None
            count += 1
            total += s.duration
            if s.children is not None:
                if children is None:
                    children = Duration(0.0)
                children += s.children
            collected[key] = (count, total, children)

    def render(self, fmt='summary'):
        if fmt == 'summary':
            yield self._render_summary()
        elif fmt == 'raw':
            yield from self._render_raw()
        else:
            raise ValueError(f'unsupported fmt {fmt!r}')

    def render_all(self, fmt='tree'):
        if fmt == 'tree' or not fmt:
            yield from self._render_tree()
        elif fmt == 'summary':
            yield self._render_summary()
            for node in self.entries:
                if isinstance(node, ProfileNode):
                    yield from node.render_all(fmt)
        elif fmt == 'raw':
            for node in self.flatten():
                yield str(entry)
        else:
            raise ValueError(f'unsupported fmt {fmt!r}')

    def _render_tree(self, depth=0, singleindent='  '):
        indent = singleindent * depth
        yield indent + self._render_summary()
        for node in self.entries:
            if not isinstance(node, ProfileNode):
                continue
            yield from node._render_tree(depth+1, singleindent)

    def _render_summary(self):
        return str(self.summarize())

    def _render_raw(self):
        for entry in self.entries:
            if isinstance(entry, ProfileNode):
                continue
            yield str(entry)


class ProfileNodeSummary(namedtuple('ProfileNodeSummary', 'filename func context duration own')):

    @classmethod
    def with_children(cls, filename, func, context, duration, own, children):
        self = cls(filename, func, context, duration, own)
        self._children = children
        return self

    def __str__(self):
        context = self.context
        if not context or context == self.func:
            context = ''
        duration = self.duration.render('ns')
        if self.own == self.duration:
            return f'{self.filename}:{self.func}() -{context}- {duration:>5}'
        else:
            own = self.own.render('ns')
            return f'{self.filename}:{self.func}() -{context}- {duration:>5} {own:>5}'

    @property
    def children(self):
        try:
            return self._children
        except AttributeError:
            if self.own == self.duration:
                return None
            return self.duration - self.own


#######################################
# targets

TARGETS = 'profiling-targets.txt'

def resolve_targets(targets, reporoot=None):
    if isinstance(targets, str):
        targets = [targets]
    reporoot = resolve_repo_root(reporoot)
    if not reporoot:
        raise Exception

    resolved = []
    for target in _iter_raw_targets(targets):
        relfile, sep, func = target.rpartition(':')
        if not sep or not relfile or not func:
            raise ValueError(f'invalid target {target!r}')
        filename = os.path.join(reporoot, relfile)
        loc = SourceLocation(filename, None, func)
        if loc not in resolved:
            resolved.append(loc)
    return resolved


def _iter_raw_targets(targets):
    for _target in targets:
        for target in _target.replace(',', ' ').split():
            if target.startswith('file:'):
                filename = target[5:]
                for target in _read_targets_file(filename):
                    if target.startswith('file:'):
                        raise NotImplementedError(filename, target)
                    yield target
            else:
                yield target


def _read_targets_file(filename):
    with open(filename) as infile:
        text = infile.read()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        yield line


def prepare_targets(helpers, targets, install=True):
    if not targets:
        raise ValueError('missing targets')
#        filename = os.path.join(reporoot, 'Objects', 'typeobject.c')
#        return ProfilingUtils., installwrap_func(filename, 'type_new')
    dirty = False
    for filename, _, func in targets:
        if helpers.wrap_func(filename, func, install):
            dirty = True
    return dirty


def prepare_main(helpers, reporoot, install=True):
    targetfile = os.path.join(reporoot, 'Programs', 'python.c')
    targets = [
        (targetfile, 'wmain'),
        (targetfile, 'main'),
    ]
    dirty = False
    for filename, func in targets:
        if helpers.wrap_func(filename, func, install):
            dirty = True
    return dirty


#######################################
# command helpers

def resolve_python(python, reporoot, build, *, fail=True, force=False):
    built = os.path.join(build.builddir, 'python') if build else None
    dev = os.path.join(reporoot, 'python') if reporoot else None

    resolved = python or built or dev or None
    if not resolved:
        return None, None, None

    if resolved == built:
        builddir = build.builddir
    elif resolved == dev:
        builddir = reporoot
    else:
        builddir = None

    if os.path.exists(resolved):
        if resolved == python:
            if python == built:
                if not force and not build.verify(python, fail=fail):
                    builddir = None
            elif python == dev:
                if not force:
                    # XXX Check it somehow?
                    pass
        exists = True
    elif not fail:
        exists = False
    else:
        if resolved == python:
            if builddir:
                raise ValueError(f'given python executable {python!r} not built yet')
            else:
                raise ValueError(f'given python executable {python!r} not found')
        else:
            raise ValueError(f'python executable not built yet (in {builddir})')

    return resolved, exists, builddir


def prepare_for_build(reporoot, targets=None, *,
                      helpers=None,
                      install=None,
                      builddir=None,
                      ):
    if not targets:
        relfile = TARGETS
        for targetsfile in [
            *((os.path.join(builddir, relfile),) if builddir else ()),
            os.path.join(reporoot, relfile),
            os.path.join(reporoot, '..', relfile),
        ]:
            if os.path.exists(targetsfile):
                targets = [f'file:{targetsfile}']
                break
        else:
            targetsfile = None
            if install is False:
                targets = ()
            else:
                raise ValueError('missing targets')
        if targetsfile:
            targets = [f'file:{targetsfile}']
    targets = resolve_targets(targets, reporoot)

    if install is not False:
        if helpers is None:
            helpers = ProfilerHelpers.from_files(reporoot=reporoot)
        force = bool(install)
        dirty1 = helpers.files.write(force=force)
#        dirty1 = helpers.add(reporoot, force=forceinstall)
        if dirty1:
            helpers.files.install(force=force)
        dirty2 = prepare_main(helpers, reporoot, install=True)
        dirty3 = prepare_targets(helpers, targets, install=True)
        dirty = dirty1 or dirty2 or dirty3
    else:
        dirty = None
    return dirty


def prepare(reporoot, targets=None, *,
            helpers=None,
            install=None,
            build=None,
            forcebuild=False,
            ):
    built = None
#    if helpers is None:
#        helpers = ProfilerHelpers.from_files()
    dirty = prepare_for_build(
        reporoot=reporoot,
        targets=targets,
        helpers=helpers,
        install=install,
        builddir=build.builddir if build else None,
    )
#    helpers_h = helpers.relfiles.header.include
    if dirty:
        built = build_cpython(build, reporoot, hide=False)
        forcebuild = False
    if forcebuild:
        built = build_cpython(build, reporoot)
    return built, dirty


def run_for_profile(python=None, *,
                    logfile=None,
                    builddir=None,
                    withsite=True,
                    capture=True,
                    ):
    if not python:
        reporoot = find_repo_root(rfail=True)
        python = os.path.join(reporoot, 'python')
        if not os.path.exists(python):
            build = CPythonBuildSpec(builddir or reporoot, cfg=None)
            python, _ = prepare(reporoot, build=build)
            assert python
    elif not os.path.exists(python):
        raise ValueError(f'given python executable {python!r} not found')

    env = {}

    if logfile:
        if os.path.exists(logfile):
            # XXX rotate it?
            os.unlink(logfile)
        env['PROFILING_LOGFILE'] = logfile

    stdout, pytotal = run_python_text(
        python, 'pass',
        nosite=not withsite,
        capture=capture,
        env=env or None,
    )

    if logfile:
        if stdout:
            with open(logfile, 'w') as outfile:
                outfile.write(stdout)
        elif os.path.exists(logfile):
            with open(logfile) as infile:
                stdout = infile.read()
        elif not capture:
            logfile = None
            stdout = None
        else:
            stdout = ''
    elif not stdout:
        stdout = '' if capture else None

    return stdout, pytotal


def show_profile_data(data, pytotal=None, fmt='tree'):
    if isinstance(data, str):
        entries = LogEntry.parse_all(data)
    else:
        entries = data

    if fmt != 'tree':
        for root in ProfileNode.from_entries(entries):
            yield from root.render_all(fmt)
        return

    summaries = {}
    roots = iter(ProfileNode.from_entries(entries))
    try:
        root = next(roots)
    except StopIteration:
        pass
    else:
        loc = root.entries[0].loc
        if loc.func in ('main', 'wmain'):
            #assert loc.filename == 'Programs/python.c'
            if list(roots):
                raise NotImplementedError
            pytotal = root.duration
            roots = iter(root.children)
        else:
            yield from root.render_all(fmt)
            root.collect_summary(summaries)
        for root in roots:
            yield from root.render_all(fmt)
            root.collect_summary(summaries)

    byfile = {}
    maxfile = maxfunc = maxcount = maxavg = maxownavg = 0
    for (filename, func), (count, total, children) in summaries.items():
        maxfile = max(maxfile, len(filename))
        maxfunc = max(maxfunc, len(func))
        maxcount = max(maxcount, count)
        assert isinstance(total, Duration), repr(total)
        avg = total / count
        maxavg = max(maxavg, avg)
        if children is not None:
            assert isinstance(children, Duration), repr(children)
            ownavg = (total - children) / count
            maxownavg = max(maxownavg, ownavg)
        else:
            ownavg = None
        try:
            funcs = byfile[filename]
        except KeyError:
            funcs = byfile[filename] = {}
        assert func not in funcs, func
        funcs[func] = (count, avg, ownavg)
    widths = (
        max(maxfile, maxfunc+4) - 4,
        len(str(maxcount)),
        len(maxavg.render('ns')),
        5 if maxavg < 10 else 6,
    )
    fmt = '    {:%d}   {:>%d} times, avg {:>%d} = {:>%d}' % widths
    if maxownavg:
        ownwidth = len(maxownavg.render('ns'))
        ownpctwidth = 5 if maxownavg < 10 else 6
        ownfmt = fmt + ' (own {:>%d} = {:>%d})' % (ownwidth, ownpctwidth)

    yield ''
    for filename, funcs in byfile.items():
#        for func, (count, avg) in funcs.items():
#            yield f'{filename + ":" + func + "()":50} {count:>5} times, avg {avg}'
        yield filename
        rows = sorted(funcs.items(), key=(lambda v: v[1]), reverse=True)
        for func, (count, avg, ownavg) in rows:
            pct = float(avg / pytotal).__format__('.2%')
            avg = avg.render('ns')
            if ownavg is None:
#                yield f'    {func + "()":30} {count:>5} times, avg {avg:>4} {pct}'
                yield fmt.format(func, count, avg, pct)
            else:
                ownpct = float(ownavg / pytotal).__format__('.2%')
                ownavg = ownavg.render('ns')
#                yield f'    {func + "()":30} {count:>5} times, avg {avg:>4} {pct} ({ownavg} {ownpct})'
                yield ownfmt.format(func, count, avg, pct, ownavg, ownpct)
    yield ''
    yield f'total: {pytotal.render("ns")}'


#######################################
# commands

class CommandFailure(RuntimeError):
    ...


def cmd_prepare(reporoot=None, targets=None, *,
                install=None,
                build=None,
                forcebuild=False,
                ):
    reporoot = resolve_repo_root(reporoot, fail=True)
    if logfile == '-':
        logfile = None

    built, installed = prepare(
        reporoot=reporoot,
        targets=targets,
        install=install,
        build=build,
        forcebuild=forcebuild,
    )

    if built:
        yield f'prepared python executable (with profiling): {built}'
    elif installed:
        yield f'prepared to build python executable (with profiling) in {reporoot}'
    else:
        yield 'did nothing'


def cmd_run(python=None, *,
            logfile=None,
            builddir=None,
            withsite=True,
            ):
    if logfile != '-':
        if not logfile:
            logfile = ProfilerHelpersHeader.LOGFILE
        yield f'logging to {logfile}'
    stdout, pytotal = run_for_profile(
        python=python,
        logfile=logfile,
        builddir=builddir,
        withsite=withsite,
        capture=not SHORTCUT,
    )
    if logfile:
        yield f'raw profile data written to {logfile}'
    elif stdout:
        for line in stdout.splitlines():
            yield line
    elif stdout is None:
        yield 'results written to stdout or unknown log file'
    else:
        yield 'results written to unknown log file'
    yield 'ran in {pytotal} seconds'


def cmd_show():
    ...


def cmd_default(python=None, logfile=None, targets=None, *,
                reporoot=None,
                install=None,
                build=None,
                forcebuild=None,
                withsite=True,
                analyze=False,
                fmt='tree',
                ):
    if logfile != '-':
        if not logfile:
            logfile = ProfilerHelpersHeader.LOGFILE
        yield f'logging to {logfile}'
    reporoot = resolve_repo_root(reporoot, fail=True)

    (python, exists, builddir,
     ) = resolve_python(python, reporoot, build, fail=False)
    if not python:
        raise CommandFailure('missing python arg')
    elif builddir:
        if not exists:
            forcebuild = True
        built, _ = prepare(
            reporoot=reporoot,
            targets=targets,
            install=install,
            build=build,
            forcebuild=forcebuild,
        )
        assert not built or built == python, (built, python)
    else:
        if not exists:
            raise CommandFailure(f'given python executable {python!r} not found')
        if install:
            raise CommandFailure('got unexpected install arg')
        if forcebuild:
            raise CommandFailure('got unexpected forcebuild arg')
        if build:
            raise CommandFailure(f'got unexpected build arg {build!r}')

    stdout, pytotal = run_for_profile(
        python=python,
        logfile=logfile,
        withsite=withsite,
        capture=not SHORTCUT or fmt != 'raw',
    )

    if stdout:
        yield from show_profile_data(stdout, pytotal, fmt)


COMMANDS = {
    None: cmd_default,
    'prepare': cmd_prepare,
    'run': cmd_run,
    'show': cmd_show,
}


#######################################
# the script

def parse_args(argv=sys.argv[1:], prog=sys.argv[0]):
    import argparse
    parser = argparse.ArgumentParser(prog=prog)

    parser.add_argument('-v', '--verbose', action='count', default=0)
    parser.add_argument('-q', '--quiet', action='count', default=0)

    parser.add_argument('--reporoot')

    parser.add_argument('--force-install', dest='install',
                        action='store_const', const=True)
    parser.add_argument('--no-install', dest='install',
                        action='store_const', const=False)
    parser.add_argument('--target', dest='targets', action='append')

    handle_build = CPythonBuildSpec.add_cli(parser)
    parser.add_argument('--python')

    parser.add_argument('--with-site', dest='withsite',
                        action='store_const', const=True)
    parser.add_argument('--no-site', '-S', dest='withsite',
                        action='store_const', const=False)
    parser.set_defaults(withsite=True)

    parser.add_argument('--format', dest='fmt',
                        choices=ProfileNode.FORMATS, default='tree')

    parser.add_argument('logfile', nargs='?')

    args = parser.parse_args(argv)
    ns = vars(args)

    cmd = None

    verbose = ns.pop('verbose')
    quiet = ns.pop('quiet')
    verbosity = VERBOSITY + verbose - quiet

    args.reporoot = resolve_repo_root(args.reporoot)

    args.build = handle_build(args, args.reporoot)

    return cmd, ns, verbosity



def main(cmd, cmd_kwargs):
    try:
        run_cmd = COMMANDS[cmd]
    except KeyError:
        sys.exit(f'ERROR: unsupported command {cmd!r}')
    try:
        for line in run_cmd(**cmd_kwargs):
            print(line)
    except CommandFailure as exc:
        sys.exit(f'ERROR: {exc}')


if __name__ == '__main__':
    cmd, cmd_kwargs, verbosity = parse_args()
    configure_logger(logger, verbosity)
    main(cmd, cmd_kwargs)
