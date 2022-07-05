#date: 2022-07-05T16:49:47Z
#url: https://api.github.com/gists/b0bd274a7c9824551da63ba73f5318f7
#owner: https://api.github.com/users/Shoeboxam


from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class MockMeasurement(object):
    input_domain: Any
    output_domain: Any
    function: Callable
    input_metric: Any
    output_measure: Any
    privacy_map: Callable

    def __call__(self, arg):
        return self.function(arg)

    def map(self, d_X, d_in, d_Y):
        return self.privacy_map(d_X, d_in, d_Y)

def make_ci(rho):
    function = lambda x: x + 1
    privacy_map = lambda d_in: d_in * rho
    return MockMeasurement(
        input_domain="VectorDomain[AllDomain[f64]]", 
        output_domain="PairDomain[AllDomain[f64], AllDomain[f64]]", 
        function=function, 
        input_metric="SymmetricDistance", 
        output_measure="ZeroConcentratedDivergence[f64]", 
        privacy_map=privacy_map)


meas = make_ci()
print(meas(23))
print(meas.map(12))