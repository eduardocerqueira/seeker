#date: 2022-02-09T17:12:53Z
#url: https://api.github.com/gists/3668e7e5b6988bf6e6f7fa3d8d33eb2d
#owner: https://api.github.com/users/BigRoy

from ragdoll.vendor import cmdx
from ragdoll import commands, interactive


def fix_marker_in_place(marker):
    source = marker["sourceTransform"].input()
    
    solver = commands._find_solver(marker)
    start_time = solver["_startTime"].asTime()
    start_frame = cmdx.frame(start_time)
    
    # Create a new but duplicate marker
    # Ragdoll doesn't allow creating a duplicate marker
    # with assign_marker. So we assign a temporary marker
    # and then reassign it to the one we want - it will allow that.
    tmp_cube = cmds.polyCube(ch=False, name="lock_marker")
    new_marker = commands.assign_marker(cmdx.ls(tmp_cube)[0], 
                                        solver)
    interactive.reassign_marker(selection=[new_marker, source])
    cmds.delete(tmp_cube)
                                        
    # We don't care about shape, size, since we'll disable collisions
    with cmdx.DGModifier() as dgmod:
        dgmod.set_attr(new_marker["inputType"], 2)            # kinematic
        dgmod.set_attr(new_marker["collide"], 0)
        
        dgmod.set_attr(new_marker["shapeType"], 0)            # box
        dgmod.set_attr(marker["shapeExtents"], [0.1, 0.1, 0.1]) # todo: make this size based on something logical
        dgmod.set_attr(marker["shapeOffset"], [0, 0, 0])
        dgmod.set_attr(marker["shapeRotation"], [0, 0, 0])
        dgmod.set_attr(new_marker["densityType"], 0)          # off
        
        # Dont' record this marker
        dgmod.set_attr(new_marker["recordRotation"], 0)
        dgmod.set_attr(new_marker["recordTranslation"], 0)
        
    # Force a refresh so that the marker has had the time to retrieve sim data
    # otherwise it will fail to generate a constraint
    cmds.currentTime(start_frame)
    cmds.refresh(force=True) 
        
    # Now generate a fixed constraint (== weld)
    commands.create_fixed_constraint(new_marker, marker)
        

for marker in cmdx.ls(selection=True, type="rdMarker"):
    fix_marker_in_place(marker)