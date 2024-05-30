#date: 2024-05-30T16:56:00Z
#url: https://api.github.com/gists/6ba5f27b9e023c5a8d389f84621736b7
#owner: https://api.github.com/users/kephale

from psygnal.qt import start_emitting_from_queue
from qtpy.QtCore import QCoreApplication

from pulser._xtouch import XTouchMini

# other setup code

def setup_xtouch(viewer, widget):
    def handle_knob(knob, value):
        knob = int(knob)
        value = int(value)
        if knob == 1:
            # Brush size
            viewer.layers[painting_layer_name].brush_size = value            
            print(f"setting brush size to {value}")
        elif knob == 2:
            # Labels
            new_label = viewer.layers[painting_layer_name].painting_labels[value % len(viewer.layers[painting_layer_name].painting_labels)]
            widget.activate_label(new_label)

    def handle_button(button):
        button = int(button)
        if button == 1:
            viewer.layers[painting_layer_name].visible = not viewer.layers[painting_layer_name].visible
        elif button == 2:
            viewer.layers[reference_layer_name].visible = not viewer.layers[reference_layer_name].visible
        elif button == 3:
            viewer.layers[painting_layer_name].mode = "pan_zoom"
        elif button == 4:
            viewer.layers[painting_layer_name].mode = "paint"
        elif button == 5:
            viewer.layers[painting_layer_name].mode = "erase"
            
    
    # connect signals to be emitted in the main thread
    d.knob.changed.connect(handle_knob, thread="main")
    d.button.pressed.connect(handle_button, thread="main")

    # start the midi thread
    d.watch()
    # start QTimer emitting signals from the main thread
    start_emitting_from_queue()