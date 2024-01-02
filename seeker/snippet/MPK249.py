#date: 2024-01-02T16:50:13Z
#url: https://api.github.com/gists/7b6ff610aa29a3055ced808e9139539e
#owner: https://api.github.com/users/asselstine

# Extends the default MPK249 MIDI script for Ableton 11 Suite
# In this script the encoders effect the selection device instead of panning. Device selection follows track selection
# Directions:
# 1. Backup the original MPK249 directory:
#   Mac: /Applications/Ableton Live 11 Suite.app/Contents/App-Resources/MIDI Remote Scripts/MPK249
#   Windows: C:\ProgramData\Ableton\Live 11 Suite\Resources\MIDI Remote Scripts\MPK249
# 2. Delete the "MPK249.pyc" file in the original directory
# 3. Copy this file to the original directory and name it "MPK249.py"
# 4. Restart Ableton
from __future__ import absolute_import, print_function, unicode_literals
from _Framework.ControlSurface import ControlSurface
from _Framework.Layer import Layer
from _Framework.DrumRackComponent import DrumRackComponent
from _Framework.TransportComponent import TransportComponent
from _Framework.DeviceComponent import DeviceComponent
from _Framework.MixerComponent import MixerComponent
from _Framework.MidiMap import MidiMap as MidiMapBase
from _Framework.MidiMap import make_button, make_encoder, make_slider
from _Framework.InputControlElement import MIDI_CC_TYPE, MIDI_NOTE_TYPE

class MidiMap(MidiMapBase):

    def __init__(self, *a, **k):
        super(MidiMap, self).__init__(*a, **k)
        self.add_button(u'Play', 0, 118, MIDI_CC_TYPE)
        self.add_button(u'Record', 0, 119, MIDI_CC_TYPE)
        self.add_button(u'Stop', 0, 117, MIDI_CC_TYPE)
        self.add_button(u'Loop', 0, 114, MIDI_CC_TYPE)
        self.add_button(u'Forward', 0, 116, MIDI_CC_TYPE)
        self.add_button(u'Backward', 0, 115, MIDI_CC_TYPE)
        self.add_matrix(u'Sliders', make_slider, 0, [[12, 13, 14, 15, 16, 17, 18, 19]], MIDI_CC_TYPE)
        self.add_matrix(u'Encoders', make_encoder, 0, [[22, 23, 24, 25, 26, 27, 28, 29]], MIDI_CC_TYPE)
        self.add_matrix(u'Arm_Buttons', make_button, 0, [
         [
          32, 33, 34, 35, 36, 37, 38, 39]], MIDI_CC_TYPE)
        self.add_matrix(u'Drum_Pads', make_button, 1, [
         [
          81, 83, 84, 86], [74, 76, 77, 79], [67, 69, 71, 72], [60, 62, 64, 65]], MIDI_NOTE_TYPE)


class MPK249(ControlSurface):

    def __init__(self, *a, **k):
        super(MPK249, self).__init__(*a, **k)
        with self.component_guard():
            midimap = MidiMap()
            drum_rack = DrumRackComponent(name='Drum_Rack',
              is_enabled=False,
              layer=Layer(pads=(midimap['Drum_Pads'])))
            drum_rack.set_enabled(True)
            transport = TransportComponent(name='Transport',
              is_enabled=False,
              layer=Layer(play_button=(midimap['Play']),
              record_button=(midimap['Record']),
              stop_button=(midimap['Stop']),
              seek_forward_button=(midimap['Forward']),
              seek_backward_button=(midimap['Backward']),
              loop_button=(midimap['Loop'])))
            transport.set_enabled(True)
            mixer_size = len(midimap['Sliders'])
            mixer = MixerComponent(mixer_size,
              name='Mixer',
              is_enabled=False,
              layer=Layer(volume_controls=(midimap['Sliders']),
              # pan_controls=(midimap['Encoders']),
              arm_buttons=(midimap['Arm_Buttons'])))
            mixer.set_enabled(True)
            # NEW STUFF BELOW ====================================================
            device = DeviceComponent(
                name='Device',
                is_enabled=False,
                layer=Layer(parameter_controls=midimap['Encoders']),
                device_selection_follows_track_selection=True)
            device.set_enabled(True)
            self.set_device_component(device)
