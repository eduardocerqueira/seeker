#date: 2025-06-13T17:01:02Z
#url: https://api.github.com/gists/b8e018dcc58d60ffe1963e598c26b246
#owner: https://api.github.com/users/SeanPesce

#!/usr/bin/env python3
# Author: Sean Pesce
#
# Converted from original JavaScript code by Seanghay Yath:
#   https://github.com/seanghay/vector-drawable-svg


import math
import sys

from xml.etree.ElementTree import Element, fromstring, tostring
from xml.dom import minidom


path_transformers = [
    lambda vdAttrs: {'d': vdAttrs.get('android:pathData')},
    lambda vdAttrs: {
        'fill': convert_hex_color(vdAttrs.get('android:fillColor'), vdAttrs.get('android:fillAlpha'))[0],
        'fill-opacity': convert_hex_color(vdAttrs.get('android:fillColor'), vdAttrs.get('android:fillAlpha'))[1] if convert_hex_color(vdAttrs.get('android:fillColor'), vdAttrs.get('android:fillAlpha'))[1] != 1 else None
    },
    lambda vdAttrs: {'stroke-linejoin': vdAttrs.get('android:strokeLineJoin')},
    lambda vdAttrs: {'stroke-linecap': vdAttrs.get('android:strokeLineCap')},
    lambda vdAttrs: {'stroke-miterlimit': vdAttrs.get('android:strokeMiterLimit')},
    lambda vdAttrs: convert_vector_drawable_stroke_attributes(vdAttrs),
    lambda vdAttrs: {'stroke-width': vdAttrs.get('android:strokeWidth')},
    lambda vdAttrs: {
        'fill-rule': vdAttrs['android:fillType'].lower() if (('android:fillType' in vdAttrs) and vdAttrs['android:fillType']) else None
    },
]

group_transformers = [
    lambda vdAttrs: {'id': vdAttrs.get('android:name')},
    lambda vdAttrs: {
        'transform': ' '.join([
            f'translate({vdAttrs.get("android:translateX", 0)}, {vdAttrs.get("android:translateY", 0)})' if (vdAttrs.get("android:translateX", 0) != 0 or vdAttrs.get("android:translateY", 0) != 0) else '',
            f'rotate({vdAttrs.get("android:rotation", 0)})' if vdAttrs.get("android:rotation", 0) != 0 else '',
            f'scale({vdAttrs.get("android:scaleX", 1)}, {vdAttrs.get("android:scaleY", 1)})' if (vdAttrs.get("android:scaleX", 1) != 1 or vdAttrs.get("android:scaleY", 1) != 1) else ''
            # @TODO: Handle 'android:pivotX' and 'android:pivotY'?
        ]).strip() or None
    },
]

gradient_transformers = [
    lambda vdAttrs: {'x1': vdAttrs.get('android:startX')},
    lambda vdAttrs: {'y1': vdAttrs.get('android:startY')},
    lambda vdAttrs: {'x2': vdAttrs.get('android:endX')},
    lambda vdAttrs: {'y2': vdAttrs.get('android:endY')},
    lambda vdAttrs: {'cx': vdAttrs.get('android:centerX')},
    lambda vdAttrs: {'cy': vdAttrs.get('android:centerY')},
    lambda vdAttrs: {'r': vdAttrs.get('android:gradientRadius')},
]

gradient_item_transformers = [
    lambda vdAttrs: {
        'stop-color': convert_hex_color(vdAttrs.get('android:color'))[0],
        'stop-opacity': convert_hex_color(vdAttrs.get('android:color'))[1] if convert_hex_color(vdAttrs['android:color'])[1] != 1 else None
    },
    lambda vdAttrs: {'offset': vdAttrs.get('android:offset')},
]


def parse_android_resource(value):
    """
    Parse Android XML Resources and returns an object.

    :param {string | undefined} value
    :returns {Object.<string, string>}
    """
    if not isinstance(value, str):
        return
    doc = minidom.parseString(value)
    resources_node = doc.getElementsByTagName('resources')[0]
    if not resources_node:
        return
    resource_map = {}
    for node in resources_node.childNodes:
        if node.nodeType != node.ELEMENT_NODE:
            continue
        if node.firstChild.nodeType != node.TEXT_NODE:
            continue
        key = f"@{node.tagName}/{node.getAttribute('name')}"
        value = node.firstChild.nodeValue
        resource_map[key] = value
    for key, value in resource_map.items():
        if re.match(r'@\w+/\w+', value):
            if value in resource_map:
                resource_map[key] = resource_map[value]
    return resource_map


def transform_attributes(vd_node, svg_node, transformers):
    if not vd_node.attributes:
        return

    #vd_attrs = {attr.name: attr.value for attr in vd_node.attributes}
    vd_attrs = {attr: vd_node.attributes.values().mapping[attr].value for attr in vd_node.attributes.values().mapping}
    for transformer in transformers:
        svg_attrs = transformer(vd_attrs)
        for name, value in svg_attrs.items():
            if value is not None:
                svg_node.setAttribute(name, value)
    return


def parse_path(root, path_node):
    svg_path = root.createElement('path')
    svg_path.setAttribute('fill', 'none')
    transform_attributes(path_node, svg_path, path_transformers)
    return svg_path


def parse_gradient(root, gradient_node):
    gradient_type = gradient_node.getAttribute('android:type')
    
    def svg_gradient(gradient_type):
        if gradient_type == 'linear':
            return root.createElement('linearGradient')
        elif gradient_type == 'radial':
            return root.createElement('radialGradient')
        elif gradient_type == 'sweep':
            raise ValueError('Sweep gradient is not compatible by SVG')
        else:
            raise ValueError('invalid gradient type')
    
    svg_gradient = svg_gradient(gradient_type)
    svg_gradient.setAttribute('gradientUnits', 'userSpaceOnUse')
    transform_attributes(gradient_node, svg_gradient, gradient_transformers)
    
    for it in gradient_node.childNodes:
        if it.tagName == 'item':
            svg_gradient_stop = root.createElement('stop')
            transform_attributes(it, svg_gradient_stop, gradient_item_transformers)
            svg_gradient.appendChild(svg_gradient_stop)
    
    return svg_gradient


def transform_node(node, parent, root, defs):
    if type(node) == minidom.Text:
        #print(node, file=sys.stderr)
        return None
    if node.tagName == 'path':
        svg_path = parse_path(root, node)
        
        for it in node.childNodes:
            if it.tagName == 'aapt:attr':
                attr_name = it.getAttribute('name')
                if attr_name in ['android:fillColor', 'android:strokeColor']:
                    for child_node in it.childNodes:
                        if child_node.tagName == 'gradient':
                            svg_gradient = parse_gradient(root, child_node)
                            if svg_gradient:
                                size = len(defs.childNodes)
                                gradient_id = f'gradient_{size}'
                                svg_gradient.setAttribute('id', gradient_id)
                                defs.appendChild(svg_gradient)
                                svg_attr_name = 'fill' if attr_name == 'android:fillColor' else 'stroke'
                                svg_path.setAttribute(svg_attr_name, f'url(#{gradient_id})')
                else:
                    continue
    
        return svg_path
    
    if node.tagName == 'group':
        group_node = root.createElement('g')
        transform_attributes(node, group_node, group_transformers)
        prev_clip_path_id = None
        
        for it in node.childNodes:
            child_path = transform_node(it, node, root, defs)
            if child_path:
                clip_path_node = getattr(child_path, 'clipPathNode', None)
                if clip_path_node is None and type(child_path) == dict:
                    clip_path_node = child_path.get('clipPathNode')
                if clip_path_node:
                    if defs is not None:
                        size = len(list(defs))
                        prev_clip_path_id = f'clip_path_{size}'
                        clip_path_node.setAttribute('id', prev_clip_path_id)
                        defs.append(minidom_to_elementtree(clip_path_node))
                    return
                
                if prev_clip_path_id:
                    child_path.setAttribute('clip-path', f'url(#{prev_clip_path_id})')
                    prev_clip_path_id = None
                if type(child_path) == dict:
                    #print(f'TESTTESTTEST', file=sys.stderr)
                    group_node.appendChild(child_path['clipPathNode'])
                else:
                    group_node.appendChild(child_path)
        
        return group_node
    
    if node.tagName == 'clip-path':
        path_data = node.getAttribute('android:pathData')
        svg_clip_path_node = root.createElement('clipPath')
        path = root.createElement('path')
        path.setAttribute('d', path_data)
        svg_clip_path_node.appendChild(path)
        #return {'clipPathNode': svg_clip_path_node}
        return svg_clip_path_node  # @TODO: Something is wrong with the clip path logic - fix this
    
    #print(f'RETURNING None ', file=sys.stderr)
    return None


def remove_dimen_suffix(dimen):
    dimen = dimen.strip()
    if not dimen:
        return dimen
    if dimen.isnumeric():
        return dimen
    if isinstance(dimen, str):
        return dimen[:-2]
    return dimen


def convert_hex_color(argb, opacity_str='1'):
    if argb is None:
        return [None, None,]
    if opacity_str is None:
        opacity_str = '1'
    if type(opacity_str) not in (str,):
        opacity_str = f'{int(opacity_str)}'
    digits = argb and argb.lstrip('#')
    opacity = float(opacity_str)
    if not digits or (len(digits) not in [4, 8]):
        return [argb, opacity]
    
    if len(digits) == 4:
        alpha = int(digits[0] * 2, 16) / 255
        red = digits[1]
        green = digits[2]
        blue = digits[3]
    else:
        alpha = int(digits[:2], 16) / 255
        red = digits[2:4]
        green = digits[4:6]
        blue = digits[6:8]
    
    return [
        '#' + red + green + blue,
        (alpha if isinstance(alpha, float) and math.isfinite(alpha) else 1) * opacity,
    ]


def convert_vector_drawable_stroke_attributes(vdAttrs):
    color = vdAttrs.get('android:strokeColor')
    alpha = vdAttrs.get('android:strokeAlpha')
    stroke_color = None
    stroke_opacity = None
    if (color is not None) and (alpha is not None):
        stroke_color, stroke_opacity = convert_hex_color(color, alpha)
    elif color is not None:
        stroke_color, stroke_opacity = convert_hex_color(color)
    return {
        'stroke': stroke_color,
        'stroke-opacity': stroke_opacity if ((stroke_opacity is not None) and stroke_opacity != 1) else None
    }


def minidom_to_elementtree(minidom_element):
    """Converts a minidom Element to an ElementTree Element."""
    xml_string = minidom_element.toxml()
    return fromstring(xml_string)


def transform(content, options=None):
    if options is None:
        options = {}
    
    override = options.get('override')
    doc = minidom.parseString(content)
    
    if override and isinstance(override, dict):
        def traverse(node, callback):
            callback(node)
            for child in node.childNodes:
                traverse(child, callback)
        
        traverse(doc, lambda node: None if not node.attributes else [
            node.setAttribute(attr.name, override[node.getAttribute(attr.name)]) 
            for attr in node.attributes.values() if node.getAttribute(attr.name) in override
        ])
    
    vector_drawables = doc.getElementsByTagName('vector')
    if len(vector_drawables) != 1:
        raise ValueError('VectorDrawable is invalid')
    
    vector_drawable = vector_drawables[0]
    viewport_width = vector_drawable.getAttribute('android:viewportWidth')
    viewport_height = vector_drawable.getAttribute('android:viewportHeight')
    output_width = remove_dimen_suffix(vector_drawable.getAttribute('android:width'))
    output_height = remove_dimen_suffix(vector_drawable.getAttribute('android:height'))
    
    svg_node = Element('svg')
    svg_node.set('id', 'vector')
    svg_node.set('xmlns', 'http://www.w3.org/2000/svg')
    svg_node.set('width', output_width or viewport_width)
    svg_node.set('height', output_height or viewport_height)
    svg_node.set('viewBox', f'0 0 {viewport_width} {viewport_height}')
    
    for child in doc.documentElement.childNodes:
        if type(child) == minidom.Text:
            assert not (child.wholeText.strip()), f'[ERROR] Found Unexpected textual data: "{child.wholeText}"'
    children_nodes = [child for child in doc.documentElement.childNodes if type(child) != minidom.Text and child.tagName]
    defs_node = Element('defs')
    nodes = [transform_node(it, doc.documentElement, doc, defs_node) for it in children_nodes]
    
    if list(defs_node): #defs_node.childNodes:
        svg_node.append(defs_node)
    
    node_indices = {
        'g': 0,
        'path': 0,
    }
    
    for node in nodes:
        # if node is None:
        #     continue
        id_ = node.getAttribute('id')
        current_id = node_indices.get(node.tagName)
        if isinstance(current_id, int):
            node_indices[node.tagName] += 1
        node.setAttribute('id', id_ or f'{node.tagName}_{current_id}')
        svg_node.append(minidom_to_elementtree(node))
    
    svg_string = tostring(svg_node).decode()
    
    if options:
        if options.get('pretty'):
            import xml.dom.minidom
            return xml.dom.minidom.parseString(svg_string).toprettyxml()
    
    return svg_string


if __name__ == '__main__':
    vd_data = None

    if len(sys.argv) < 2:
        print(f'Usage:\n\tpython3 {sys.argv[0]} <Android vector drawable XML file>', file=sys.stderr)
        sys.exit(1)
    
    with open(sys.argv[1], 'r') as f:
        vd_data = f.read()
    
    # Test data
    #vd_data = '<vector xmlns:android="http://schemas.android.com/apk/res/android" android:height="48.000000dip" android:width="48.000000dip" android:viewportWidth="48.000000" android:viewportHeight="48.000000">\n  <group>\n    <clip-path android:pathData="M0,0h48v48h-48z M 0,0"/>\n    <path android:fillColor="#FFFFFFFF" android:pathData="M24.53,22.621m-17.396,0a17.396,17.396 0,1 1,34.792 0a17.396,17.396 0,1 1,-34.792 0"/>\n    <group>\n      <clip-path android:pathData="M7.593,33.276l10.709,6.975 15.558,-0.223L33.86,11.092L7.593,11.092Z M 0,0"/>\n      <path android:fillColor="#FF0A0909" android:pathData="M28.906,15.808a30.48,30.48 0,0 1,-10.081 3.284c-5.721,0.783 -9.159,2.366 -9.085,8.811 0.025,2.2 1.159,5.092 0.872,7.23 -0.2,1.5 -2.146,3.24 -3.061,5.114 -1.735,3.554 -1.046,8.134 -0.672,11.97 3.335,-0.4 2.627,-0.456 3.024,-3.823a32.659,32.659 0,0 1,4.166 -12.265c1.185,5.421 0.542,10.621 -0.682,15.975 3.6,-1.58 4.256,-3.637 5.206,-7.631 1.213,-5.1 -2.339,-7.965 -2.774,-12.631 -0.18,-1.928 2.086,-7.2 3.276,-8.733 1.21,-1.557 2.061,-1.211 4.247,-1.709a15.516,15.516 0,0 0,7.439 -4.154,13.7 13.7,0 0,0 -1.877,-1.436"/>\n      <path android:fillColor="#FF0A0909" android:pathData="M28.906,15.018s0.355,-0.739 0.618,-0.312 0.183,0.629 0.183,0.629a10.6,10.6 0,0 1,2.229 -0.454c0.624,0.033 0.424,0.493 0.424,0.493s0.533,-0.483 0.562,0.118c0.01,0.208 0.312,-0.394 0.542,-0.1s0.04,0.178 0.138,0.473 0.259,0.645 -1.056,0.317 -3.641,-1.167 -3.641,-1.167"/>\n      <path android:fillColor="#FF979898" android:pathData="M12.711,23.804c-0.24,-0.12 -7.21,-0.9 -7.991,-1.142l-0.781,-0.24 -0.9,7.991 9.794,-0.12Z"/>\n      <path android:fillColor="#FFB8B8B9" android:pathData="M13.386,25.177c-0.226,-0.145 -7.07,-1.675 -7.821,-2l-0.75,-0.323 -1.76,7.847 9.749,0.939Z"/>\n      <path android:fillColor="#FFE1E1E2" android:pathData="M13.626,26.078c-0.226,-0.146 -7.07,-1.675 -7.821,-2l-0.75,-0.324 -1.76,7.847 9.749,0.939Z"/>\n      <path android:fillColor="#FF0A0909" android:pathData="M13.695,52.559c0.3,0.422 1.384,2.51 2.686,3.312 1.121,0.69 2.269,0.791 1.631,-1.055a24.3,24.3 0,0 0,-1.569 -3.5Z"/>\n      <path android:fillColor="#FF0A0909" android:pathData="M9.71,52.378c-0.665,0.1 -3.828,0.248 -5.339,1.053 -1.3,0.693 -1.82,1.623 0.868,1.761a48.514,48.514 0,0 0,5.216 -0.041Z"/>\n      <path android:fillColor="#FF0A0909" android:pathData="M19.128,14.844c0,2.072 -1.031,3.752 -2.3,3.752s-2.3,-1.68 -2.3,-3.752 1.031,-3.752 2.3,-3.752 2.3,1.68 2.3,3.752"/>\n    </group>\n    <path android:fillColor="#FFFFFFFF" android:pathData="M22.149,25.813a0.2,0.2 0,0 1,-0.1 -0.2l1.576,-10.171a0.2,0.2 0,0 1,0.091 -0.138,0.2 0.2,0 0,1 0.163,-0.02l13.827,4.5a0.192,0.192 0,0 1,0.131 0.215L36.261,30.174a0.2,0.2 0,0 1,-0.091 0.138,0.194 0.194,0 0,1 -0.163,0.02l-13.827,-4.5a0.2,0.2 0,0 1,-0.035 -0.016"/>\n    <path android:fillColor="#FF232322" android:pathData="M37.843,19.632a0.4,0.4 0,0 0,-0.071 -0.031l-13.827,-4.5a0.389,0.389 0,0 0,-0.326 0.04,0.4 0.4,0 0,0 -0.182,0.276L21.861,25.582a0.389,0.389 0,0 0,0.192 0.4,0.38 0.38,0 0,0 0.071,0.031l13.827,4.5a0.389,0.389 0,0 0,0.326 -0.04,0.4 0.4,0 0,0 0.182,-0.276l1.576,-10.172a0.389,0.389 0,0 0,-0.192 -0.4m-0.194,0.342L36.078,30.146l-13.827,-4.5 1.576,-10.171Z"/>\n    <path android:fillColor="#FFFFFFFF" android:pathData="M22.149,25.813a0.2,0.2 0,0 1,-0.09 -0.227l3.159,-10.4a0.2,0.2 0,0 1,0.1 -0.118,0.2 0.2,0 0,1 0.151,-0.013l13.833,4.5a0.2,0.2 0,0 1,0.035 0.016,0.2 0.2,0 0,1 0.09,0.227l-3.159,10.4a0.2,0.2 0,0 1,-0.1 0.118,0.2 0.2,0 0,1 -0.151,0.013l-13.833,-4.5a0.2,0.2 0,0 1,-0.035 -0.016"/>\n    <path android:fillColor="#FF232322" android:pathData="M39.432,19.403a0.41,0.41 0,0 0,-0.071 -0.031l-13.833,-4.5a0.389,0.389 0,0 0,-0.3 0.026,0.4 0.4,0 0,0 -0.194,0.235l-3.159,10.4a0.39,0.39 0,0 0,0.179 0.454,0.409 0.409,0 0,0 0.071,0.031l13.833,4.5a0.391,0.391 0,0 0,0.3 -0.026,0.4 0.4,0 0,0 0.194,-0.235l3.159,-10.4a0.391,0.391 0,0 0,-0.179 -0.454m-0.194,0.342L36.076,30.148l-13.833,-4.5 3.159,-10.4Z"/>\n    <path android:fillColor="#FFFFFFFF" android:pathData="M22.15,25.812a0.193,0.193 0,0 1,-0.081 -0.09,0.2 0.2,0 0,1 0,-0.158l4.74,-10.917a0.2,0.2 0,0 1,0.24 -0.109l6.074,1.978a9.6,9.6 0,0 1,1.718 0.768c2.689,1.521 3.058,3.338 2.87,4.628a3.237,3.237 0,0 1,0.74 2.6,5.537 5.537,0 0,1 -0.355,1.473l-1.84,4.24a0.2,0.2 0,0 1,-0.24 0.11l-13.829,-4.5a0.209,0.209 0,0 1,-0.035 -0.016"/>\n    <path android:fillColor="#FF232322" android:pathData="M34.937,17.112l-0.194,0.342 0.194,-0.342a9.888,9.888 0,0 0,-1.756 -0.784l-6.071,-1.977a0.393,0.393 0,0 0,-0.48 0.219l-4.74,10.917a0.394,0.394 0,0 0,0 0.317,0.385 0.385,0 0,0 0.163,0.18 0.376,0.376 0,0 0,0.07 0.031l13.829,4.5a0.392,0.392 0,0 0,0.48 -0.219l1.84,-4.238c0.047,-0.11 1.067,-2.578 -0.358,-4.215a3.917,3.917 0,0 0,-0.806 -2.915,6.944 6.944,0 0,0 -2.172,-1.818m-0.194,0.342c1.413,0.8 3.144,2.263 2.759,4.524 1.457,1.464 0.413,3.929 0.413,3.929l-1.84,4.238 -13.829,-4.5 4.74,-10.917 6.074,1.978a9.5,9.5 0,0 1,1.683 0.752"/>\n    <path android:fillColor="#FF1A00C4" android:pathData="M37.957,22.871l-11.186,-3.623 0.991,-2.326 9.665,3.2Z"/>\n    <path android:fillColor="#FF1A00C4" android:pathData="M28.805,23.856l-3.4,-1.178 1.036,-2.308 3.4,1.178Z"/>\n    <path android:fillColor="#FF0A0909" android:pathData="M35.585,29.154a0.2,0.2 0,0 1,-0.062 -0.01l-6.84,-2.27a0.2,0.2 0,1 1,0.125 -0.376l6.84,2.27a0.2,0.2 0,0 1,-0.062 0.386ZM35.93,28.323a0.2,0.2 0,0 1,-0.062 -0.01l-6.84,-2.27a0.2,0.2 0,0 1,0.125 -0.376l6.84,2.27a0.2,0.2 0,0 1,-0.062 0.386ZM36.275,27.492a0.2,0.2 0,0 1,-0.062 -0.01l-6.84,-2.27a0.2,0.2 0,0 1,0.125 -0.376l6.84,2.27a0.2,0.2 0,0 1,-0.062 0.386ZM36.62,26.661a0.2,0.2 0,0 1,-0.062 -0.01l-6.84,-2.27a0.2,0.2 0,1 1,0.125 -0.376l6.84,2.27a0.2,0.2 0,0 1,-0.062 0.386ZM27.588,26.539a0.2,0.2 0,0 1,-0.064 -0.011l-3.307,-1.136a0.2,0.2 0,1 1,0.128 -0.374l3.307,1.136a0.2,0.2 0,0 1,-0.064 0.385ZM36.965,25.831a0.194,0.194 0,0 1,-0.062 -0.01l-6.84,-2.27a0.2,0.2 0,1 1,0.125 -0.375l6.84,2.27a0.2,0.2 0,0 1,-0.062 0.386ZM28.018,25.779a0.2,0.2 0,0 1,-0.064 -0.011l-3.307,-1.136a0.2,0.2 0,0 1,0.129 -0.374l3.307,1.136a0.2,0.2 0,0 1,-0.064 0.385ZM37.31,25.001a0.194,0.194 0,0 1,-0.062 -0.01l-6.84,-2.27a0.2,0.2 0,1 1,0.125 -0.376l6.84,2.27a0.2,0.2 0,0 1,-0.062 0.386ZM28.325,24.948a0.2,0.2 0,0 1,-0.063 -0.01l-3.184,-1.066a0.2,0.2 0,1 1,0.126 -0.375l3.184,1.066a0.2,0.2 0,0 1,-0.063 0.385ZM37.654,24.17a0.2,0.2 0,0 1,-0.062 -0.01l-6.84,-2.27a0.2,0.2 0,0 1,0.125 -0.375l6.84,2.27a0.2,0.2 0,0 1,-0.062 0.386Z"/>\n    <path android:fillColor="#FFFFFFFF" android:pathData="M37.952,26.154a41.711,41.711 0,0 0,-3.553 -4.4,6.59 6.59,0 0,0 -0.969,-5.024 9.218,9.218 0,0 1,1.994 0.816,5.812 5.812,0 0,1 3.062,5.343 10.383,10.383 0,0 1,-0.533 3.264"/>\n    <path android:fillColor="#FF232322" android:pathData="M37.993,26.613l-0.01,0.024c0.011,0 0.014,-0.007 0.01,-0.024m-2.472,-9.237a9.908,9.908 0,0 0,-2.646 -0.992s1.941,1.74 1.316,5.439a34.984,34.984 0,0 1,3.8 4.791c0.173,-0.427 2.486,-6.432 -2.472,-9.237m-0.193,0.342a5.629,5.629 0,0 1,2.963 5.175,10.025 10.025,0 0,1 -0.4,2.818c-0.29,-0.41 -0.665,-0.9 -1,-1.334 -0.413,-0.529 -1.429,-1.808 -2.277,-2.691a7.022,7.022 0,0 0,-0.722 -4.607,8.561 8.561,0 0,1 1.442,0.639"/>\n    <path android:fillColor="#FF1A00C4" android:pathData="M24.242,46.537l10.725,-6.07 -21.686,-0.5Z"/>\n    <path android:fillColor="#FF1A00C4" android:pathData="M24.58,43.307a20.643,20.643 0,0 1,-14.6 -35.241A20.643,20.643 0,1 1,39.178 37.26,20.509 20.509,0 0,1 24.58,43.307ZM24.58,5.97A16.694,16.694 0,1 0,41.274 22.663,16.713 16.713,0 0,0 24.58,5.97Z"/>\n  </group>\n</vector>\n'

    svg_data = transform(vd_data)
    print(svg_data)
