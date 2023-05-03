#date: 2023-05-03T17:02:45Z
#url: https://api.github.com/gists/2146747599cfba0765dc5d9f77abd5e8
#owner: https://api.github.com/users/cetchebarne

import c4d
import math

replaceShaders = True
targetShaders = ['RedshiftMaterial', 'RedshiftMaterialBlender']

# mapping ['from', 'to']
mappingAiStandardSurface = [    ['base', 'diffuse_weight'],
    ['baseColor', 'diffuse_color'],
    ['diffuseRoughness', 'diffuse_roughness'],
    ['specular', 'refl_weight'],
    ['specularColor', 'refl_color'],
    ['specularRoughness', 'refl_roughness'],
    ['specularIOR', 'refl_ior'],
    ['specularAnisotropy', 'refl_aniso'],
    ['specularRotation', 'refl_aniso_rotation'],
    ['metalness', 'refl_metalness'],
    ['transmission', 'refr_weight'],
    ['transmissionColor', 'refr_color'],
    ['transmissionDepth', 'refr_depth'],
    ['transmissionDispersion', 'refr_abbe'],
    ['transmissionExtraRoughness', 'refr_roughness'],
    ['transmissionScatter', 'refr_absorbtion_scale'],
    #['transmissionScatterAnisotropy', 'refr_'],
    ['coat', 'coat_weight'],
    ['coatColor', 'coat_color'],
    ['coatRoughness', 'coat_roughness'],
    ['coatIOR', 'coat_ior'],
    ['coatNormal', 'coat_bump_input'],
    ['emission', 'emission_weight'],
    ['emissionColor', 'emission_color'],
    ['opacity', 'opacity_color'],

    ['useFresnel', 'refr_use_base_IOR'],

    #['subsurface', 'transl_weight'],
    #['subsurfaceColor', 'transl_color'],
    #['subsurfaceRadius', 'ms_radius'],
    #['subsurfaceScale', 'ss_amount'],
    ['thinWalled', 'refr_thin_walled'],

    ['fogColor', 'refr_transmittance_color'],
    ['normalCamera', 'bump_input']
]

mappingAiMixShader = [    ['mix', 'blendColor1'],
    ['shader1', 'baseColor'],
    ['shader2', 'layerColor1']
]

def convertUi():
    ret = c4d.gui.MessageDialog('Convert all shaders in scene, or selected shaders?')
    if ret == c4d.IDYES:
        convertAllShaders()
    elif ret == c4d.IDNO:
        convertSelection()

    setupOpacities()
    # convertOptions()


def convertSelection():
    """
    Loops through the selection and attempts to create Redshift shaders on whatever it finds
    """

    sel = doc.GetActiveObjects(c4d.GETACTIVEOBJECTFLAGS_SELECTIONORDER)
    if sel:
        for s in sel:
            ret = doMapping(s)

def convertAllShaders():
    """
    Converts each (in-use) material in the scene
    """

    for shdType in targetShaders:
        shaderColl = doc.GetMaterialsByType(getattr(c4d, shdType))
        if shaderColl:
            for x in shaderColl:
                # query the objects assigned to the shader
                # only convert things with members
                shdGroup = x.GetFirstShader()
                setMem = shdGroup[c4d.SLA_SGROUP_MEMBERS]
                if setMem:
                    ret = doMapping(x)

def doMapping(inShd):
    """
    Figures out which attribute mapping to use, and does the thing.
    @param inShd: Shader name
    @type inShd: String
    """
    ret = None

    shaderType = inShd.GetTypeName()
    if 'Standard' in shaderType :
        ret = shaderToRsMaterial(inShd, 'RedshiftMaterial', mappingStandard)

    elif 'Shader' in shaderType :
        ret = shaderToRsMaterial(inShd, 'RedshiftMaterialBlender', mappingShader)

    if ret:
        # assign objects to the new shader
        assignToNewShader(inShd, ret)

def assignToNewShader(oldShd, newShd):
    """
    Creates a shading group for the new shader, and assigns members of the old shader to it
    @param oldShd: Old shader to upgrade
    @type oldShd: String
    @param newShd: New shader
    @type newShd: String
    """

    retVal = False

    mat = newShd.GetFirstMaterial()
    if mat:
        if replaceShaders:
            mat[c4d.MATERIAL_COLOR_SHADER] = newShd
            doc = c4d.documents.GetActiveDocument()
            doc.RemoveObject(oldShd)
        else:
            mat[c4d.RENDERENGINE_SHADER] = newShd
        retVal = True

    return retVal

def setupConnections(inShd, fromAttr, outShd, toAttr):
    input = inShd.GetFirstShader()
    while input:
        if input.GetDescription() == fromAttr:
            output = outShd.GetFirstShader()
            while output:
                if output.GetDescription() == toAttr:
                    input.ConnectTo(output)
                    return True
                output = output.GetNext()
        input = input.GetNext()

    return False

def shaderToRsMaterial(inShd, nodeType, mapping):
    """
    'Converts' a shader to Redshift, using a mapping table.
    @param inShd: Shader to convert
    @type inShd: c4d.BaseMaterial
    @param nodeType: Redshift shader type to create
    @type nodeType: str
    @param mapping: List of attributes to map from old to new
    @type mapping: List
    """

    # print('Converting material:', inShd.GetName())

    if ':' in inShd.GetName():
        rsName = inShd.GetName().rsplit(':')[-1] + '_rs'
    else:
        rsName = inShd.GetName() + '_rs'

    rsNode = c4d.BaseShader(c4d.Xredshift)
    rsNode.SetName(rsName)

    for chan in mapping:
        fromAttr = chan[0]
        toAttr = chan[1]

        if inShd[fromAttr].GetType() != c4d.NOTOK:
            # print('\t', fromAttr, ' -> ', toAttr)

            fromData = inShd[fromAttr]
            toData = rsNode[toAttr]

            # Connect the attribute
            toData.ConnectFrom(fromData)
        else:
            # print('Attribute not found:', fromAttr)
            pass

    # print('Done. New shader is', rsNode.GetName())

    return rsNode


def setValue(attr, value):
    """Simplified set attribute function.
    @param attr: Attribute to set. Type will be queried dynamically
    @param value: Value to set to. Should be compatible with the attr type.
    """

    if not isinstance(attr, c4d.BaseContainer):
        return

    # Set the attribute value
    attrData = attr[c4d.DESC_CYCLE].GetType()
    if attrData == c4d.DTYPE_LONG or attrData == c4d.DTYPE_REAL or attrData == c4d.DTYPE_VECTOR:
        attr[c4d.DESC_DEFAULT] = value

    elif attrData == c4d.DTYPE_BOOL:
        attr[c4d.DESC_DEFAULT] = bool(value)

    elif attrData == c4d.DTYPE_STRING:
        attr[c4d.DESC_DEFAULT] = str(value)

    elif attrData == c4d.DTYPE_FILENAME:
        attr[c4d.DESC_DEFAULT] = c4d.storage.LoadPath(str(value))

    elif attrData == c4d.DTYPE_TIME:
        attr[c4d.DESC_DEFAULT] = c4d.BaseTime(float(value))

    elif attrData == c4d.DTYPE_BASETIME:
        attr[c4d.DESC_DEFAULT] = c4d.BaseTime(float(value))

    elif attrData == c4d.DTYPE_COLOR:
        attr[c4d.DESC_DEFAULT] = c4d.Vector(float(value[0]), float(value[1]), float(value[2]))

    elif attrData == c4d.DTYPE_LONGVECTOR:
        attr[c4d.DESC_DEFAULT] = c4d.Vector(float(value[0]), float(value[1]), float(value[2]), float(value[3]))

    elif attrData == c4d.DTYPE_NIL:
        # TODO: handle unknown data types
        pass

def transparencyToOpacity(inShd, outShd):
    transpMap = outShd[c4d.MATERIAL_USE_TRANSPARENCY]
    if transpMap is not None:
        # map is connected, argh...
        # need to add a reverse node in the shading tree

        # create reverse
        invertNode = c4d.BaseList2D(c4d.Xbitmap)
        invertNode.SetName(outShd.GetName() + '_rev')
        outShd.InsertShader(invertNode)

        #connect transparency Map to reverse 'input'
        transpMap.Connect(invertNode[c4d.BITMAPSHADER_FILENAME], c4d.SELECTION_NEW)

        #connect reverse to opacity
        outShd[c4d.MATERIAL_USE_ALPHA] = True
        invertNode[c4d.BITMAPSHADER_ALPHA_MODE] = c4d.ALPHA_MODE_STRAIGHT
        invertNode[c4d.BITMAPSHADER_ALPHA_COLOR] = c4d.Vector(1, 1, 1)
        outShd[c4d.MATERIAL_ALPHA_SHADER] = invertNode
    else:
        transparency = inShd[c4d.MATERIAL_TRANSPARENCY]
        opacity = [(1.0 - transparency.x), (1.0 - transparency.y), (1.0 - transparency.z)]

        #print opacity
        setValue(outShd + '.opacity', opacity)


def convertAiStandardSurface(inShd, outShd):

    #anisotropy from -1:1 to 0:1
    anisotropy = inShd[c4d.AIMATERIAL_ANISOTROPY]
    anisotropy = (anisotropy * 0.5) + 0.5
    setValue(outShd + '.specularAnisotropy', anisotropy)

    # do we need to check lockFresnelIORToRefractionIOR
    # or is fresnelIOR modified automatically when refractionIOR changes ?
    ior = 1.0
    if inShd[c4d.AIMATERIAL_LOCK_FRESNEL_IOR_TO_REFRACTION_IOR]:
        ior = inShd[c4d.AIMATERIAL_REFRACTION_IOR]
    else:
        ior = inShd[c4d.AIMATERIAL_FRESNEL_IOR]


    reflectivity = 1.0
    connReflectivity = outShd[c4d.MATERIAL_COLOR_SHADER].GetFirstConnection()
    if not connReflectivity:
        reflectivity = outShd[c4d.MATERIAL_COLOR_BRIGHTNESS]

    frontRefl = (ior - 1.0) / (ior + 1.0)
    frontRefl *= frontRefl

    setValue(outShd +'.Ksn', frontRefl * reflectivity)

    reflGloss = inShd[c4d.AIMATERIAL_REFLECTION_GLOSSINESS]
    setValue(outShd + '.specularRoughness', 1.0 - reflGloss)

    refrGloss = inShd[c4d.AIMATERIAL_REFRACTION_GLOSSINESS]
    setValue(outShd + '.refractionRoughness', 1.0 - refrGloss)


    #bumpMap, bumpMult, bumpMapType ?

    if inShd[c4d.AIMATERIAL_SUB_SURFACE_SCATTERING]:
        setValue(outShd + '.Ksss', 1.0)

    #selfIllumination is missing  but I need to know the exact attribute name in maya or this will fail

def convertOptions():
    doc = c4d.documents.GetActiveDocument()
    renderData = doc.GetActiveRenderData()
    arnoldData = renderData.GetFirstVideoPost().GetMain()
    arnoldData[c4d.AI_GI_TRANSMISSION_DEPTH] = 10


def isOpaque(shapeName):
    doc = c4d.documents.GetActiveDocument()

    # get shading engine
    mySGs = []
    for tag in doc.GetTags():
        if tag.GetType() == c4d.Tpolygonselection:
            if tag.GetName() == shapeName + "_SG":
                mySGs.append(tag.GetBaseSelect())
    if not mySGs:
        return 1

    # get surface shader
    surfaceShader = None
    for mat in doc.GetMaterials():
        if mat[c4d.MATERIAL_USE_REFLECTION] == False and mat[c4d.MATERIAL_USE_ALPHA] == False:
            for tag in mat.GetTags():
                if tag.GetType() == c4d.Tbase:
                    if tag[c4d.ID_BASELIST_NAME] == shapeName + "_SG":
                        surfaceShader = mat
                        break
        if surfaceShader:
            break
    if surfaceShader == None:
        return 1

    # check opacity
    if surfaceShader[c4d.MATERIAL_ALPHA_SHADER] == None:
        return 1
    shaderData = surfaceShader[c4d.MATERIAL_ALPHA_SHADER].GetData()
    if shaderData.GetOpCount() == 0:
        return 1
    opacity = shaderData.GetOpContainerInstance(0)
    if opacity[c4d.GV_REDSHIFT_SHADER_OPACITY] < 1.0 or opacity[c4d.GV_REDSHIFT_SHADER_OPACITY + 1] < 1.0 or opacity[c4d.GV_REDSHIFT_SHADER_OPACITY + 2] < 1.0:
        return 0
    return 1


def setupOpacities():
    doc = c4d.documents.GetActiveDocument()
    for obj in doc.GetObjects():
        if obj.GetType() != c4d.Opolygon:
            continue
        shapeName = obj.GetName()
        if isOpaque(shapeName) == 0:
            #print shapeName + ' is transparent'
            obj[c4d.AIOBJECT_OPACITY] = 1.0
            obj[c4d.AIOBJECT_OPAQUE] = 0

if not c4d.plugins.FindPlugin(1029983, c4d.PLUGINTYPE_SCENELOADER):
    c4d.plugins.GeLoadPlugin("c4dtoa")
if not c4d.plugins.FindPlugin(1036342, c4d.PLUGINTYPE_SHADER):
    c4d.plugins.GeLoadPlugin("c4d_Redshift")
    
convertUi()