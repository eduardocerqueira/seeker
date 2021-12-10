#date: 2021-12-10T16:55:39Z
#url: https://api.github.com/gists/fcbafd5cf748c9b11e64a4dd37ec8e9a
#owner: https://api.github.com/users/papr

from calibration_choreography.base_plugin import CalibrationChoreographyPlugin


class ExternalCalibrationChoreography(CalibrationChoreographyPlugin):
    is_user_selectable = False
    shows_action_buttons = False
    is_session_persistent = False
    label = "External Calibration"

    def __init__(self, *args, **kwargs):
        type(self).is_user_selectable = True
        super().__init__(*args, **kwargs)

    def cleanup(self):
        type(self).is_user_selectable = False
        super().cleanup()

    @classmethod
    def _choreography_description_text(cls) -> str:
        return "Choreography that collects reference data from an external client."

    def recent_events(self, events):
        super().recent_events(events)

        if self.is_active:
            self.pupil_list.extend(events["pupil"])

    def on_notify(self, note_dict):
        if note_dict["subject"] == "calibration.add_ref_data":
            self.ref_list += note_dict["ref_data"]
        super().on_notify(note_dict)
