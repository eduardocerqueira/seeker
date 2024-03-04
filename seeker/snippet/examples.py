#date: 2024-03-04T17:00:40Z
#url: https://api.github.com/gists/7a4e574406cc8942346ff81f9d3fc825
#owner: https://api.github.com/users/danielgregkraken

class TestGetExpectedThroughputForScheduleObject:
    # To me this is a really nice example of a static method on a class. By putting it inside the class (and not outside) we
    # associate the function with the test class, and by decorating it with `staticmethod` we emphasise that this method
    # does not actually depend on any attribute defined on the class itself or an instance of the class. (i.e. there's no 
    # reference to self or cls <=> i.e. the return value is a function of the inputs only and not any global values defined
    # on the class/object)
    
    @staticmethod
    def generate_schedule(
        service_type,
        throughput_low: int = 1_000,
        throughput_high: int = 2_000,
        max_power: int = 2_000_000,
        min_power: int = -2_000_000,
    ) -> DeviceServiceSchedule:
      pass