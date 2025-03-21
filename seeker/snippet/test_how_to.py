#date: 2025-03-21T17:00:52Z
#url: https://api.github.com/gists/05a26912120062b4dcb5e8af00f1af56
#owner: https://api.github.com/users/niltonfrederico

from django.test import TestCase

class TestPayment(TestCase, metaclass=ParametrizedTestCaseMeta):
    @parametrize(
        "amount",
        [1,2,3, ...]
    )
    def test_sum(self, amount: int)
        self.assertEqual(0 + amount, amount, msg="Dumb assertion.")
    
    @parametrize(
        "amount, expected_result", # Can also be a list
        [
            (1,2),
            (2,3),
            (3,4)
        ]
    )
    def test_sum_and_result(self, amount: int, expected_result: int):
        self.assertEqual(1 + amount, expected_result, msg="Dumb assertion.")