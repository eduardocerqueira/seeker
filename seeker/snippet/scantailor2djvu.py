#date: 2024-04-19T16:44:22Z
#url: https://api.github.com/gists/e4dca9fa4c01051ef0f429e7f22471e2
#owner: https://api.github.com/users/srtlg

"""
Use ScanTailor split output to generate djvu files

https://diybookscanner.org/forum/viewtopic.php?t=3601

see also https://github.com/strider1551/djvubind

requires DJVU Libra, ImageMagick, libtiff
"""
import os
import shutil
from io import BytesIO
from enum import Enum, auto
from subprocess import check_call, check_output
from multiprocessing.dummy import Pool

MINIDJVU = False
LANGUAGE = 'deu'
OCR = None;'tesseract'


class Format(Enum):
    BW = auto()
    COLOR = auto()


class TiffFile:
    def __init__(self, root, name):
        self.root = root
        self._name = name
        self._basename = os.path.splitext(name)[0]
        self._ocr_base = None
        self.resolution = 0
        self.format = Format.BW

    @property
    def path(self):
        return os.path.join(self.root, self._name)

    def intermediary(self, extension):
        return os.path.join(self.root, '{}.{}'.format(self._basename, extension))

    @property
    def foreground(self):
        return os.path.join(self.root, 'foreground', self._name)

    @property
    def background(self):
        return os.path.join(self.root, 'background', self._name)

    @property
    def is_separated(self):
        return os.path.exists(self.foreground)

    def _attr_from_tiffinfo(self, stdout: str):
        for line in stdout.splitlines(keepends=False):
            if line.lstrip().startswith('Resolution:'):
                a, v = line.split(':')
                r1 = v.split(',')
                self.resolution = int(r1[0].strip())
            elif line.lstrip().startswith('Bits/Sample:'):
                a, v = line.split(':')
                res = v.strip()
                if res == '1':
                    self.format = Format.BW
                else:
                    self.format = Format.COLOR

    def _cjb2(self, src, dst):
        if MINIDJVU:
            check_call(['minidjvu', '-d', str(self.resolution), src, dst])
        else:
            check_call(['cjb2', '-dpi', str(self.resolution), src, dst])

    def _to_csep(self):
        self._cjb2(self.foreground, self.intermediary('foreground.djvu'))
        self._ocr_base = self.foreground
        check_call(['ddjvu', '-format=rle', self.intermediary('foreground.djvu'), self.intermediary('rle')])
        with open(self.intermediary('mix'), 'wb') as fout, open(self.intermediary('rle'), 'rb') as frle:
            shutil.copyfileobj(frle, fout)
            bkg = check_output(['convert', self.background, 'ppm:-'])
            shutil.copyfileobj(BytesIO(bkg), fout)
        check_call(['csepdjvu', '-d', str(self.resolution), self.intermediary('mix'), self.intermediary('djvu')])

    def _to_djvu(self):
        if self.is_separated:
            self._to_csep()
        elif self.format == Format.BW:
            self._cjb2(self.path, self.intermediary('djvu'))
            self._ocr_base = self.path
        elif self.format == Format.COLOR:
            check_call(['convert', self.path, self.intermediary('jpg')])
            check_call(['c44', '-dpi', str(self.resolution), self.intermediary('jpg'), self.intermediary('djvu') ])

    def _append_text(self, txt):
        raise RuntimeError('set-txt requires a specific format')
        check_call(['djvused', '-e', "select 1; remove-txt; set-txt '{}'; save".format(txt),
                   self.intermediary('djvu')])

    def _ocr(self):
        if self._ocr_base is None:
            return
        if OCR is None:
            return
        if OCR == 'tesseract':
            env = dict(OMP_THREAD_LIMIT='1')
            check_call(['tesseract', self._ocr_base, self.intermediary('txt'), '-l', LANGUAGE], env=env)
            self._append_text(self.intermediary('txt') + '.txt')

    def convert(self):
        tiffinfo = check_output(['tiffinfo', self.path])
        self._attr_from_tiffinfo(tiffinfo.decode('utf8'))
        self._to_djvu()
        self._ocr()
        print(self._name, self.format, self.resolution, 'S' if self.is_separated else ' ')


def _convert_page(page: TiffFile):
    page.convert()


def _convert(pages, output_file):
    with Pool() as pool:
        pool.map(_convert_page, pages)
    args = ['djvm', '-c', output_file]
    args += [i.intermediary('djvu') for i in pages]
    check_call(args)
    print('output written to', output_file)


def main(input_directory, output_file):
    pages = []
    for f in sorted(os.listdir(input_directory)):
        if f.endswith('.tif') and not f.startswith('.'):
            obj = TiffFile(input_directory, f)
            pages.append(obj)
    _convert(pages, output_file)


if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
