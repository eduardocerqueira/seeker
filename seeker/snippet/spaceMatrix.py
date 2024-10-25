#date: 2024-10-25T16:53:11Z
#url: https://api.github.com/gists/b87a182991434ecfb6b8351821d46062
#owner: https://api.github.com/users/gregoiredehame

import maya.api.OpenMaya as om2

def maya_useNewAPI():
    pass
    

class spaceMatrix(om2.MPxNode):
    id_ = om2.MTypeId(0x00124dee)

    def __init__(self) -> None:
        super(spaceMatrix, self).__init__()

    @classmethod
    def creator(cls) -> None:
        return cls()

    @classmethod
    def initialize(cls) -> None:
        fn_matrix = om2.MFnMatrixAttribute()
        fn_numeric = om2.MFnNumericAttribute()
        fn_compound = om2.MFnCompoundAttribute()

        # INPUTS
        cls.space = fn_numeric.create("space", "s", om2.MFnNumericData.kInt, 0)
        fn_numeric.storable = True
        fn_numeric.writable = True
        fn_numeric.keyable = True
        fn_numeric.setMin(0)
        cls.addAttribute(cls.space)
        
        cls.input_matrix = fn_matrix.create("inputMatrix", "imat", om2.MFnMatrixAttribute.kDouble)
        fn_matrix.storable = True
        fn_matrix.writable = True
        fn_matrix.readable = True
        fn_matrix.keyable = False
        cls.addAttribute(cls.input_matrix)
        
        cls.target_matrix = fn_matrix.create("targetMatrix", "tmat", om2.MFnMatrixAttribute.kDouble)
        fn_matrix.storable = True
        fn_matrix.writable = True
        fn_matrix.readable = True
        fn_matrix.keyable = False
        cls.addAttribute(cls.target_matrix)
        
        cls.offset_matrix = fn_matrix.create("offsetMatrix", "omat", om2.MFnMatrixAttribute.kDouble)
        fn_matrix.storable = True
        fn_matrix.writable = True
        fn_matrix.readable = True
        fn_matrix.keyable = False
        cls.addAttribute(cls.offset_matrix)
        
        cls.point_weight = fn_numeric.create("pointWeight", "pw", om2.MFnNumericData.kFloat, 1.0)
        fn_numeric.storable = True
        fn_numeric.writable = True
        fn_numeric.keyable = True
        fn_numeric.setMin(0.0)
        fn_numeric.setMax(1.0)
        cls.addAttribute(cls.point_weight)
        
        cls.orient_weight = fn_numeric.create("orientWeight", "ow", om2.MFnNumericData.kFloat, 1.0)
        fn_numeric.storable = True
        fn_numeric.writable = True
        fn_numeric.keyable = True
        fn_numeric.setMin(0.0)
        fn_numeric.setMax(1.0)
        cls.addAttribute(cls.orient_weight)
        

        cls.target = fn_compound.create("target", "tgt")
        fn_compound.addChild(cls.target_matrix)
        fn_compound.addChild(cls.offset_matrix)
        fn_compound.addChild(cls.point_weight)
        fn_compound.addChild(cls.orient_weight)
        fn_compound.array = True
        fn_compound.usesArrayDataBuilder = True
        fn_compound.storable = True
        fn_compound.writable = True
        cls.addAttribute(cls.target)
        
        # OUPUTS
        cls.output_matrix = fn_matrix.create("outputMatrix", "om")
        fn_matrix.storable = False
        fn_matrix.keyable = False
        fn_matrix.connectable = True
        cls.addAttribute(cls.output_matrix)

        cls.attributeAffects(cls.space, cls.output_matrix)
        cls.attributeAffects(cls.input_matrix, cls.output_matrix)
        cls.attributeAffects(cls.target, cls.output_matrix)
        

    def compute(self, plug:om2.MPlug, data:om2.MDataBlock) -> None:
        if plug == self.output_matrix:
            
            space = data.inputValue(self.space).asInt()
            input_matrix = om2.MTransformationMatrix(data.inputValue(self.input_matrix).asMatrix())
  
            target_array_handle = data.inputArrayValue(self.target)

            if target_array_handle.jumpToLogicalElement(space):
                target_element_handle = target_array_handle.inputValue()
            
                target_matrix = target_element_handle.child(self.target_matrix).asMatrix()
                offset_matrix = target_element_handle.child(self.offset_matrix).asMatrix()
                output_matrix = om2.MTransformationMatrix(offset_matrix * target_matrix) 
                                
                point_weight = target_element_handle.child(self.point_weight).asFloat()
                if point_weight != 1.0:
                    trans_input = input_matrix.translation(om2.MSpace.kWorld)
                    trans_target = output_matrix.translation(om2.MSpace.kWorld)
                    trans_blended = (1 - point_weight) * trans_input + point_weight * trans_target
                    output_matrix.setTranslation(trans_blended, om2.MSpace.kWorld)
                            
                orient_weight = target_element_handle.child(self.orient_weight).asFloat()
                if orient_weight != 1.0:
                    quat_input = input_matrix.rotation(asQuaternion=True)
                    quat_target = output_matrix.rotation(asQuaternion=True)
                    quat_slerp = om2.MQuaternion.slerp(quat_input, quat_target, orient_weight)
                    output_matrix.setRotation(quat_slerp.asEulerRotation())

                data.outputValue(self.output_matrix).setMMatrix(output_matrix.asMatrix())
                
            else:
                data.outputValue(self.output_matrix).setMMatrix(input_matrix.asMatrix())
            data.setClean(plug)


def initializePlugin(obj:om2.MObject) -> None:
    fn_plugin = om2.MFnPlugin(obj)
    fn_plugin.registerNode(spaceMatrix.__name__, spaceMatrix.id_, spaceMatrix.creator, spaceMatrix.initialize)


def uninitializePlugin(obj:om2.MObject) -> None:
    fn_plugin = om2.MFnPlugin(obj)
    fn_plugin.deregisterNode(spaceMatrix.id_)