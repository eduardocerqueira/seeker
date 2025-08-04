#date: 2025-08-04T16:49:49Z
#url: https://api.github.com/gists/563fd9afee7a3bbf370c269dd9d63ca5
#owner: https://api.github.com/users/Birdacious

import bpy
import leap
from mathutils import Vector, Quaternion

blones = bpy.data.objects['FingerArmatureObj'].pose.bones

class ModalTimerOperator(bpy.types.Operator):
    """Operator which runs itself from a timer"""
    bl_idname = "wm.modal_timer_operator"
    bl_label = "Modal Timer Operator"

    _timer = None

    def modal(self, context, event):
        global connection
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            wm = context.window_manager
            wm.event_timer_remove(self._timer)
            bpy.types.VIEW3D_MT_view.remove(menu_func)
            connection.disconnect()
            return {'CANCELLED'}

        if event.type == 'TIMER':
            ev = connection.wait_for(leap.enums.EventType.Tracking)
            render_hands(ev)

        return {'PASS_THROUGH'}

    def execute(self, context):
        global connection
        global open_connection
        connection = leap.Connection()
        open_connection = connection.connect()
        connection.set_tracking_mode(leap.TrackingMode.Desktop)

        wm = context.window_manager
        self._timer = wm.event_timer_add(.02, window=context.window)
        wm.modal_handler_add(self)

        return {'RUNNING_MODAL'}

def render_hands(event):
    if len(event.hands) == 0: return
    for h in range(0, len(event.hands)):
        hand = event.hands[h]
        h = 0 if str(hand.type) == "HandType.Left" else 1
        for d in range(0, 5):
            digit = hand.digits[d]
            for b in range(0, 4):
                bone = digit.bones[b]
                q = bone.rotation
                q = Quaternion((q.w,q.x,q.z,q.y))
                q.invert()
                p = bone.prev_joint
                p = Vector((p.x,p.z,p.y))/100
                n = bone.next_joint
                n = Vector((n.x,n.z,n.y))/100
                l = (n-p).length

                blone = blones[str(h)+str(d)+str(b)]
                blone.rotation_quaternion = q
                blone.location = p
                blone.scale.y = -l
                
                bpy.data.objects[f"{h}{d}{b}p"].location = p
                bpy.data.objects[f"{h}{d}{b}n"].location = n
        
        bone = hand.arm
        q = bone.rotation
        q = Quaternion((q.w,q.x,q.z,q.y))
        q.invert()
        p = bone.prev_joint
        p = Vector((p.x,p.z,p.y))/100
        n = bone.next_joint
        n = Vector((n.x,n.z,n.y))/100
        l = (n-p).length
        blone = blones[str(h)+'a']
        blone.rotation_quaternion = q
        blone.location = p
        blone.scale.y = -l
        bpy.data.objects[f"{h}ap"].location = p
        bpy.data.objects[f"{h}an"].location = n

def menu_func(self, context):
    self.layout.operator(ModalTimerOperator.bl_idname, text=ModalTimerOperator.bl_label)

def register():
    bpy.utils.register_class(ModalTimerOperator)
    bpy.types.VIEW3D_MT_view.append(menu_func)

# Register and add to the "view" menu (to use F3 search "Modal Timer Operator" for quick access).
def unregister():
    bpy.utils.unregister_class(ModalTimerOperator)
    bpy.types.VIEW3D_MT_view.remove(menu_func)

if __name__ == "__main__":
    register()