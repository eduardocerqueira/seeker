#date: 2023-02-15T16:58:03Z
#url: https://api.github.com/gists/51175224ebd52bf8315d908ffc3710e4
#owner: https://api.github.com/users/tomowarkar

class _aabb:
    def __init__(self, _from, _to):
        self._from = tuple(map(min, zip(_from, _to)))
        self._to = tuple(map(max, zip(_from, _to)))
        
    def __iter__(self):
        return iter((*self._from, *self._to))
    
    @property
    def width(self):
        return tuple(map(lambda x: x[0]-x[1], zip(self._to, self._from)))
    
    @property
    def half(self):
        return tuple(map(lambda x: x*.5, self.width))
        
    @property
    def center(self):
        return tuple(map(lambda x: x[0]+x[1], zip(self._from, self.half)))

    def contains(self, other):
        pair1 = map(lambda x: x[0]+x[1], zip(self.half, other.half))
        pair2 = map(lambda x: x[0]-x[1], zip(self.center, other.center))
        return not any(map(lambda x: x[0]<x[1], zip(pair1, pair2)))
        

if __name__ == '__main__':
    from PIL import Image, ImageDraw


    a = _aabb((100, 100), (300, 200))
    b = _aabb((10, 10), (90, 90))
    c = _aabb((290, 190), (390, 290))
    cyan, magenta, yellow = (0, 255, 255), (255, 0, 255), (255, 255, 0)
    black, gray = (0, 0, 0), (240, 240, 240)

    im = Image.new('RGB', (400, 300), gray)
    draw = ImageDraw.Draw(im)

    draw.rectangle(list(a), outline=cyan, width=10)
    draw.text(a.center, 'A', black)

    draw.rectangle(list(b), outline=magenta, width=10)
    draw.text(b.center, 'B', black)

    draw.rectangle(list(c), outline=yellow, width=10)
    draw.text(c.center, 'C', black)

    print('A contains B:', a.contains(b)) #> A contains B: False
    print('A contains C:', a.contains(c)) #> A contains C: True
