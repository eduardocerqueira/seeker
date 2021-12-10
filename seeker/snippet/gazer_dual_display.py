#date: 2021-12-10T16:55:39Z
#url: https://api.github.com/gists/fcbafd5cf748c9b11e64a4dd37ec8e9a
#owner: https://api.github.com/users/papr

import logging

from gaze_mapping import Gazer2D
from gaze_mapping.gazer_base import (
    NotEnoughPupilDataError,
    NotEnoughReferenceDataError,
    NotEnoughDataError,
)
from gaze_mapping.gazer_2d import Model2D_Monocular
from gaze_mapping.utils import closest_matches_monocular_batch

logger = logging.getLogger(__name__)


class GazerDualDisplay2D(Gazer2D):
    label = "Dual-Display 2D"

    def init_matcher(self):
        self.matcher = DummyMatcher()

    def fit_on_calib_data(self, calib_data):
        # extract reference data
        ref_data = calib_data["ref_list"]
        # extract and filter pupil data
        pupil_data = calib_data["pupil_list"]
        pupil_data = self.filter_pupil_data(
            pupil_data, self.g_pool.min_calibration_confidence
        )
        if not pupil_data:
            raise NotEnoughPupilDataError
        if not ref_data:
            raise NotEnoughReferenceDataError
        # match pupil to reference data for each eye separately
        matches_right = self.match_pupil_to_ref(pupil_data, ref_data, eye_id=0)
        matches_left = self.match_pupil_to_ref(pupil_data, ref_data, eye_id=1)
        if matches_right[0]:
            self._fit_monocular_model(self.right_model, matches_right)
        else:
            logger.warning("Not enough matching data to fit right model")
        if matches_left[0]:
            self._fit_monocular_model(self.left_model, matches_left)
        else:
            logger.warning("Not enough matching data to fit left model")
        if not self.right_model.is_fitted and not self.left_model.is_fitted:
            raise NotEnoughDataError

    def match_pupil_to_ref(self, pupil_data, ref_data, eye_id):
        ref_data = [ref for ref in ref_data if ref["eye_id"] == eye_id]
        pupil_data = [datum for datum in pupil_data if datum["id"] == eye_id]
        return closest_matches_monocular_batch(ref_data, pupil_data)

    # overwrite model init functions in case frame size is not available,
    # e.g. external HMD use case

    def _init_left_model(self):
        return NoOutlierRemoval_Model2D_Monocular()

    def _init_right_model(self):
        return NoOutlierRemoval_Model2D_Monocular()

    def _init_binocular_model(self):
        """Just used for code compatibility with base classes"""
        return NoOutlierRemoval_Model2D_Monocular()


class NoOutlierRemoval_Model2D_Monocular(Model2D_Monocular):
    def fit(self, X, Y):
        assert X.shape[0] == Y.shape[0], "Required shape: (n_samples, n_features)"
        self._validate_feature_dimensionality(X)
        self._validate_reference_dimensionality(Y)

        if X.shape[0] == 0:
            raise NotEnoughDataError

        polynomial_features = self._polynomial_features(X)
        self._regressor.fit(polynomial_features, Y)
        self._is_fitted = True


class DummyMatcher:
    """Dummy matcher that simply returns the input pupil datum.

    Matching is only required if you want to build binocular pairs.
    """

    def map_batch(self, pupil_list):
        results = []
        for p in pupil_list:
            results.extend(self.on_pupil_datum(p))
        return results

    def on_pupil_datum(self, p):
        yield [p]
