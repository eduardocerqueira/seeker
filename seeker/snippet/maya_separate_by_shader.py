#date: 2022-10-11T17:25:10Z
#url: https://api.github.com/gists/ce8dbffbc8b00b2b8d0a05874d457a50
#owner: https://api.github.com/users/paulwinex

from pymel.core import *


def separate_by_shaders(obj):
    shaders = []
    for sg in obj.getShape().outputs(type='shadingEngine'):
        shaders.extend(sg.surfaceShader.inputs())
    shaders = list(set(shaders))
    
    for shader in shaders:
        members = get_shader_members(shader)
        if members:
            select(members)
            mel.DetachComponent()
    select(obj)
    objs = polySeparate(ch=0)
    return objs


def get_shader_members(mat):
    sg = listConnections(mat+'.outColor')
    if sg:
        members = sets(sg[0], q=True )
        return members

def separate_by_shader_selected():
    obj = selected()[0]
    separate_by_shaders(obj)
    
# separate_by_shader_selected()