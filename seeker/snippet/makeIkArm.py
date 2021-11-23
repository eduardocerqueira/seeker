#date: 2021-11-23T17:09:13Z
#url: https://api.github.com/gists/ebb5f27bb90d52bb3309147d48043a7c
#owner: https://api.github.com/users/LukeCY

from maya import cmds
from maya import OpenMaya as om


class IkArm(object):

    def __init__(self):
        self.elbow = None
        self.wrist = None
        self.shoulder = None
        self.ik = None
        self.pv_position = None
        self.prefix = None

    # Create an IK handle from the joint above to the joint below the currently selected 'elbow' joint
    def create_ik_ctrl(self):

        solver = cmds.ls(type="ikRPsolver")
        if not solver:
            cmds.error("No solver")

        self.elbow = cmds.ls(selection=True)[0]
        wrist_jnt = cmds.listRelatives(self.elbow, children=True)
        if not wrist_jnt:
            cmds.error("No child joint to use - select the middle joint of a 3-joint chain")
        self.wrist = wrist_jnt[0]
        shoulder_jnt = cmds.listRelatives(self.elbow, parent=True)
        if not shoulder_jnt:
            cmds.error("No parent joint to use - select the middle joint of a 3-joint chain")
        self.shoulder = shoulder_jnt[0]

        self.get_component_name_prefix()

        ik_handle = cmds.ikHandle(name=f"{self.prefix}_ik", startJoint=self.shoulder, endEffector=self.wrist, solver="ikRPsolver")
        cmds.rename(ik_handle[-1], f"{self.prefix}_ik_eff")
        self.ik = ik_handle[0]
        cmds.setAttr(f"{self.ik}.visibility", 0)

        ctrl = cmds.circle(name=f"{self.prefix}_ik_ctrl", normal=(1, 0, 0), radius=1.5)[0]
        cmds.matchTransform(ctrl, self.wrist)
        grp = cmds.group(name=f"{self.prefix}_ik_ctrl_grp", empty=True)
        cmds.matchTransform(grp, ctrl)
        cmds.parent(ctrl, grp)
        cmds.pointConstraint(ctrl, self.ik)

    # This relies on a component naming convention of 'componentType_sideOfBody_additionalDetail...'
    def get_component_name_prefix(self):

        example_name = self.elbow

        rig_component = example_name.split("_")[0]
        body_side = example_name.split("_")[1]

        self.prefix = f"{rig_component}_{body_side}"

    # Calculate the vector of the pole vector position from the origin
    def find_pv_position(self):

        shoulder_pos = cmds.xform(self.shoulder, query=True, worldSpace=True, translation=True)
        elbow_pos = cmds.xform(self.elbow, query=True, worldSpace=True, translation=True)
        wrist_pos = cmds.xform(self.wrist, query=True, worldSpace=True, translation=True)

        shoulder_vec = om.MVector(shoulder_pos[0], shoulder_pos[1], shoulder_pos[2])
        elbow_vec = om.MVector(elbow_pos[0], elbow_pos[1], elbow_pos[2])
        wrist_vec = om.MVector(wrist_pos[0], wrist_pos[1], wrist_pos[2])

        shoulder_to_wrist_vec = (wrist_vec - shoulder_vec)
        shoulder_to_elbow_vec = (elbow_vec - shoulder_vec)
        elbow_to_wrist_vec = (wrist_vec - elbow_vec)

        # Calculate fraction of vector between shoulder and wrist that ends closest to the elbow
        scale_value = (shoulder_to_wrist_vec * shoulder_to_elbow_vec) / (shoulder_to_wrist_vec * shoulder_to_wrist_vec)

        closest_point_to_elbow_vec = shoulder_vec + shoulder_to_wrist_vec * scale_value

        pv_vec = elbow_vec - closest_point_to_elbow_vec

        shoulder_to_elbow_length = shoulder_to_elbow_vec.length()
        elbow_to_wrist_length = elbow_to_wrist_vec.length()
        total_length = shoulder_to_elbow_length + elbow_to_wrist_length

        self.pv_position = elbow_vec + pv_vec.normal() * total_length

    # Create a pole vector control at the calculated location
    def create_pv_ctrl(self):

        loc = cmds.spaceLocator(name=f"{self.prefix}_pv_ctrl")
        cmds.move(self.pv_position.x, self.pv_position.y, self.pv_position.z, loc)
        grp = cmds.group(name=f"{self.prefix}_pv_ctrl_grp", empty=True)
        cmds.matchTransform(grp, loc)
        cmds.parent(loc, grp)
        cmds.poleVectorConstraint(loc, self.ik)
