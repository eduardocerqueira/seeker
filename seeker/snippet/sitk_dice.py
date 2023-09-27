#date: 2023-09-27T17:03:01Z
#url: https://api.github.com/gists/194ec7bdc939482215d569dfde4c998d
#owner: https://api.github.com/users/jpeoples

import SimpleITK as sitk
import numpy

def load_ct(p):
    """Load CT from path p"""
    return sitk.ReadImage(p)
  
def load_segment(p, ct, bgval=0):
    """Load mask from path p and ensure same size of underlying array as CT image ct"""
    m = sitk.ReadImage(p) != bgval
    size = ct.GetSize()
    
    assert numpy.allclose(m.GetDirection(), ct.GetDirection(), atol=1e-3)
    assert numpy.allclose(m.GetSpacing(), ct.GetSpacing(), atol=1e-3)

    if ct.GetSize() == m.GetSize(): 
        # The images already are in same coord system and same number of slices, so we are fine
        assert numpy.allclose(ct.GetOrigin(), m.GetOrigin())
        return m
    
    # Mask volume is cropped, within CT volume, so we need to create our own image with the same size
    arr = numpy.zeros(size[::-1], dtype=numpy.uint8)
    mor = m.GetOrigin()
    mask_origin_index = ct.TransformPhysicalPointToIndex(mor)
    mask_bound = numpy.add(mor, numpy.multiply(m.GetSize(), m.GetSpacing()))
    mask_bound_index = ct.TransformPhysicalPointToIndex(mask_bound)
    
    # Copy the loaded mask into the CT.
    arr[mask_origin_index[2]:mask_bound_index[2], mask_origin_index[1]:mask_bound_index[1], mask_origin_index[0]:mask_bound_index[0]] = sitk.GetArrayViewFromImage(m)

    mout = sitk.GetImageFromArray(arr)
    mout.CopyInformation(ct)

    # Now mout matches ct
    return mout
  
def dice(m1, m2):
    a1 = sitk.GetArrayViewFromImage(m1)
    a2 = sitk.GetArrayViewFromImage(m2)
    s1 = a1.sum()
    s2 = a2.sum()
    sinter = (a1 & a2).sum()
    
    return 2 * sinter / (s1 + s2)