#date: 2025-02-19T16:59:31Z
#url: https://api.github.com/gists/a74c201730af3e4c5f3bbe48ab221f56
#owner: https://api.github.com/users/Erotemic

import kwimage
import kwplot
import numpy as np


def padded_slice_v1(imdata, sl):
    # Shift imdata and slice if negative slice start value
    if sl[0].start < 0:
        shift = -sl[0].start
        imdata = np.roll(imdata, shift, axis=0)
        imdata[:shift] = np.zeros((imdata[:shift].shape))
        sl = (slice(0, sl[0].stop - sl[0].start), sl[1])
    if sl[1].start < 0:
        shift = -sl[1].start
        imdata = np.roll(imdata, shift, axis=1)
        imdata[:, :shift] = np.zeros((imdata[:, :shift].shape))
        sl = (sl[0], slice(0, sl[1].stop - sl[1].start))
    relative_submask = imdata[sl]
    return relative_submask


def padded_slice_v2(imdata, sl):
    import delayed_image
    delayed = delayed_image.DelayedIdentity(imdata)
    padded_crop = delayed.crop(sl, clip=False, wrap=False)
    subdata = padded_crop.finalize()
    return subdata


def main():
    imdata = kwimage.grab_test_image('astro')

    box1 = kwimage.Box.coerce([-10, -20, 100, 200], format='xywh')
    box2 = kwimage.Box.coerce([400, 400, 200, 300], format='xywh')
    box3 = kwimage.Box.coerce([-100, -100, 700, 700], format='xywh')

    plt = kwplot.autoplt()
    pnum_ = kwplot.PlotNums(nRows=3, nCols=4)
    kwplot.imshow(imdata, pnum=pnum_(), doclf=1, title='original')
    ax = plt.gca()
    box1.draw()
    box2.draw()
    box3.draw()
    for c in ax.collections:
        c.set_clip_on(False)

    sl = box1.to_slice()

    import ubelt as ub
    with ub.Timer('pad slice v1'):
        subdata1 = padded_slice_v1(imdata, sl)
    with ub.Timer('pad slice v2'):
        subdata2 = padded_slice_v2(imdata, sl)

    kwplot.imshow(subdata1, pnum=pnum_(), title='left top submask v1')
    kwplot.imshow(subdata2, pnum=pnum_(), title='left top submask v2')

    diff = np.abs(kwimage.ensure_float01(subdata2) - kwimage.ensure_float01(subdata1))
    assert diff.sum() == 0
    kwplot.imshow(diff, pnum=pnum_(), title='diff')

    # Test bottom right case:
    pnum_()  # skip first figure
    sl = box2.to_slice()
    subdata1 = padded_slice_v1(imdata, sl)
    subdata2 = padded_slice_v2(imdata, sl)
    kwplot.imshow(subdata1, pnum=pnum_(), title='bottom right submask v1')
    kwplot.imshow(subdata2, pnum=pnum_(), title='bottom right submask v2')
    pnum_()  # skip diff figure

    # Test bigger than everything case
    pnum_()  # skip first figure
    sl = box3.to_slice()
    subdata1 = padded_slice_v1(imdata, sl)
    subdata2 = padded_slice_v2(imdata, sl)
    kwplot.imshow(subdata1, pnum=pnum_(), title='big submask v1')
    kwplot.imshow(subdata2, pnum=pnum_(), title='big submask v2')
    pnum_()  # skip diff figure


if __name__ == '__main__':
    main()
