#date: 2023-05-03T16:58:59Z
#url: https://api.github.com/gists/77d7179ef1ffb67629aba161a1c2dc0c
#owner: https://api.github.com/users/pfultz2

# Create a find function to refactor nodes. This takes a node which is a clang
# cursor and ClangReplacements class. It will return true if it should
# traverse the children.
#
# def find(node, r):
#     ...
#     return True
# 
# refactor = ClangRefactor(compiler_path='/usr')
# refactor.walk(find)

from ctypes import *
import clang.cindex as clang
import os, sys, re, subprocess, yaml
from multiprocessing import Pool

cursor_and_range_visit = CFUNCTYPE(c_int, c_void_p, clang.Cursor, clang.SourceRange)

def get_include_locations(tu, f):
    for fi in tu.get_includes():
        loc = fi.location
        if str(loc.file) == str(f):
            yield loc

def get_specialized_cursor_template(c):
    return clang.conf.lib.clang_getSpecializedCursorTemplate(c)

def get_template_cursor_kind(c):
    return clang.CursorKind.from_id(clang.conf.lib.clang_getTemplateCursorKind(c))

class ClangReplacements:
    def __init__(self, src, dir=None):
        self.src = src
        self.replacements = []
        self.dir = dir or os.path.join(os.getcwd(), 'fixits')
        self.includes = {}

    def make_id(self):
        return re.sub('[^0-9a-zA-Z_]','_', self.src) + '.yaml'

    def replace(self, c, text):
        self.replacements.append(self.convert_replacement(c, text))

    def insert(self, c, text):
        self.replacements.append(self.convert_replacement(c, text, length=0))

    def insert_at(self, src, offset, text):
        self.replacements.append(self.get_replacement(src, offset, 0, text))

    def insert_include(self, node, include):
        tu = node.translation_unit
        src = str(node.location.file)
        if not src in self.includes:
            self.includes[src] = []
        if not include in self.includes[src]:
            includes = list(get_include_locations(tu, src))
            loc = includes[-1]
            loc2 = clang.SourceLocation.from_position(tu, loc.file, loc.line, 1)
            self.insert_at(loc2.file, loc2.offset, "#include {}\n".format(include))
            self.includes[src].append(include)

    def get_replacement(self, filepath, offset, length, text):
        return {'FilePath': str(filepath), 'Offset': offset, 'Length': length, 'ReplacementText': str(text)}

    def convert_replacement(self, c, text, length=None):
        sr = c.extent
        ln = length or sr.end.offset-sr.start.offset
        loc = c.location
        return self.get_replacement(loc.file, sr.start.offset, ln, text)

    def merge_replacements(self):
        offsets = {}
        rep = []
        for replacement in self.replacements:
            if replacement['Offset'] in offsets and replacement['Length'] == 0:
                r = offsets[replacement['Offset']]
                r['ReplacementText'] = r['ReplacementText'] + replacement['ReplacementText']
            else:
                offsets[replacement['Offset']] = replacement
        for offset, r in offsets.items():
            rep.append(r)
        return rep

    def to_clang_replacements(self):
        return {'MainSourceFile': self.src, 'Replacements': self.merge_replacements()}

    def to_file(self, f):
        stream = open(f, 'w')
        r = self.to_clang_replacements()
        yaml.dump(r, stream)

    def save(self):
        if len(self.replacements) > 0:
            self.to_file(os.path.join(self.dir, self.make_id()))

def expression_string(c):
    return ' '.join([tok.spelling for tok in c.get_tokens()])

def walk_cursors(node, v, r):
    if not v(node, r):
        for child in node.get_children():
            walk_cursors(child, v, r)

def show_label(c, label):
    if c:
        print(label, c.kind, c.displayname)
    else:
        print(label, 'None')

def show(c):
    print(c.kind, c.displayname)
    show_label(c.referenced, 'referenced:')
    for child in c.get_children():
        show_label(child, '     child:')
    for arg in c.get_arguments():
        show_label(arg, '     arg:')

def skip_children(c, *kinds):
    for child in c.get_children():
        if child.kind in kinds:
            for gchild in skip_children(child, *kinds):
                yield gchild
        else:
            yield child

def get_kinds(cs, *kinds):
    for c in cs:
        if c.kind in kinds:
            yield c

def has_kinds(cs, *kinds):
    for c in get_kinds(cs, *kinds):
        return True
    return False

def get_spellings(cs):
    return [c.spelling for c in cs]

def get_base_fields(c):
    for base in get_kinds(c.get_children(), clang.CursorKind.CXX_BASE_SPECIFIER):
        for field in get_fields(base.referenced):
            yield field

def get_fields(c):
    if not c:
        return []
    if c.kind == clang.CursorKind.TYPE_REF:
        return get_fields(c.referenced)
    t = get_specialized_cursor_template(c)
    if t:
        return get_fields(t)
    if has_kinds(c.get_children(), clang.CursorKind.FIELD_DECL):
        return get_kinds(c.get_children(), clang.CursorKind.FIELD_DECL)
    else:
        return get_base_fields(c)

def print_diagnostics(diags, depth=0):
    for diag in diags:
        margin = ''
        for x in range(depth):
            margin = margin + '*'
        print(margin, diag.format())
        print_diagnostics(diag.children, depth+1)

class ClangRefactor:
    def __init__(self, compiler_path, build_dir=None, fixit_dir=None):
        self.compiler_path = compiler_path
        clang.Config.set_library_file('{}/lib/libclang.so'.format(self.compiler_path))
        self.default_includes = self.get_default_includes()
        self.build_dir = build_dir or os.getcwd()
        self.fixit_dir = fixit_dir or os.path.join(self.build_dir, 'fixits')
        self.cdb = clang.CompilationDatabase.fromDirectory(self.build_dir)
        self.index = clang.Index.create()
        self.add_functions()

    def add_function(self, *args):
        clang.register_function(clang.conf.lib, args, True)

    def add_functions(self):
        self.add_function('clang_findIncludesInFile', [clang.TranslationUnit, clang.File, cursor_and_range_visit])

    def get_default_includes(self):
        includes = []
        clang_exe = '{}/bin/clang++'.format(self.compiler_path)
        output = subprocess.check_output([clang_exe, '-E', '-x', 'c++', '-', '-v'], input='', stderr=subprocess.STDOUT)
        capture = False
        for line in output.decode('utf-8').split('\n'):
            if 'End of search' in line:
                break
            if capture:
                includes.append(line.strip())
            else:
                if '#include <...> search starts here' in line:
                    capture = True
        return includes

    def parse(self, filename):
        file_args = self.cdb.getCompileCommands(filename)[0].arguments
        args = []
        skip = True
        for arg in file_args:
            if skip:
                skip = False
                continue
            if arg in ['-c', '-o']:
                skip = True
                continue
            args.append(arg)
        for include in self.default_includes:
            args.append('-isystem')
            args.append(include)
        tu = self.index.parse(filename, args)
        print(filename, ' '.join(args))
        print_diagnostics(tu.diagnostics)
        return tu

    def get_files(self):
        cmds = self.cdb.getAllCompileCommands()
        return [str(c.filename) for c in cmds]

    def walk_file(self, f, v):
        tu = self.parse(f)
        r = ClangReplacements(f, self.fixit_dir)
        walk_cursors(tu.cursor, v, r)
        r.save()

    def walk(self, v):
        p = Pool()
        fv = ((f, v) for f in self.get_files())
        p.starmap(self.walk_file, fv)

    def __getstate__(self):
        attributes = self.__dict__.copy()
        del attributes['cdb']
        del attributes['index']
        return attributes

    def __setstate__(self, state):
        self.__dict__ = state
        self.cdb = clang.CompilationDatabase.fromDirectory(self.build_dir)
        self.index = clang.Index.create()

# clang_findIncludesInFile doesn't seem to work
def find_includes(tu, f):
    includes = []
    def visitor(ctx, c, r):
        includes.append((c, r))
    clang.conf.lib.clang_findIncludesInFile(tu, f, cursor_and_range_visit(visitor))
    return includes