#date: 2021-10-13T16:51:20Z
#url: https://api.github.com/gists/098c4718187593d253532f4fc6d61217
#owner: https://api.github.com/users/Kolupsy

import bpy
import mathutils
from mathutils.geometry import interpolate_bezier

def evaluate_spline( spline, resolution ):

    points = spline.bezier_points
    length = len( points )
    frac = 1 / ( length * resolution )
    ev = []
    for i in range( len( points )):

        knot1 = points[ i ]
        handle1 = knot1.handle_right
        knot2 = points[ (i+1)%len( points )]
        handle2 = knot2.handle_left
        new_points = interpolate_bezier( knot1.co, handle1, handle2, knot2.co, resolution )[:-1]
        for p in new_points:
            l = len( ev )
            ev.append(( l * frac, p ))

    return ev

def nearest_on_curve( obj, curve_data, resolution ):

    ref_loc = obj.matrix_world.to_translation( )
    spline = curve_data.splines.active

    ev = evaluate_spline( spline, resolution )
    nearest = ev[0]
    for e in ev[1:]:
        p = e[1]

        nearest_dis = ( ref_loc - nearest[1] ).length
        new_dis = ( ref_loc - p ).length
        if new_dis < nearest_dis:
            nearest = e

    return nearest

def NOC( obj, curve_data, resolution ):

    return nearest_on_curve( obj, curve_data, resolution )[1]

bpy.app.driver_namespace['NOC'] = NOC