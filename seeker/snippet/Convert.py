#date: 2024-05-13T16:45:17Z
#url: https://api.github.com/gists/4fcc5b6ae51464919dc0e46c07715381
#owner: https://api.github.com/users/Alastor15

import unittest
import re
from functools import reduce

metric = {
  'length': {
    'units': ['km', 'hm', 'dam', 'm', 'dm', 'cm', 'mm'],
    'updater': {
      'inc': lambda a, _: a * 10,
      'dec': lambda a, _: a / 10,
    },
  },
  'weight': {
    'units': ['kg', 'hg', 'dkg', 'g', 'dg', 'cg', 'mg'],
    'updater': {
      'inc': lambda a, _: a * 10,
      'dec': lambda a, _: a / 10,
    },
  },
}

imperial = {
  'length': {
    'units': ['mi', 1760, 'yd', 3, 'ft', 12, 'in'],
    'updater': {
      'inc': lambda a, b: a if type(b) == str else a * b,
      'dec': lambda a, b: a if type(b) == str else a / b,
    },
  },
  'weight': {
    'units': ['st', 14, 'lb', 16, 'oz'],
    'updater': {
      'inc': lambda a, b: a if type(b) == str else a * b,
      'dec': lambda a, b: a if type(b) == str else a / b,
    },
  },
}

measure_system = {
  'imperial': imperial,
  'metric': metric,
}

sys_to_sys = {
  'length': {
    'imperial': 'in',
    'metric': 'cm',
    'by': 2.54,
    'imperial_to_metric': lambda a, b: a * b,
    'metric_to_imperial': lambda a, b: float(a) / float(b),
  },
  'weight': {
    'imperial': 'oz',
    'metric': 'kg',
    'by': 35.274,
    'imperial_to_metric': lambda a, b: float(a) / float(b),
    'metric_to_imperial': lambda a, b: a * b,
  },
}

def get_distance(system, from_measure, to_measure):
  if (from_measure == to_measure):
    return 1

  units = system['units'] # ['km', 'hm', ...]
  [from_indx, to_indx] = [units.index(a) for a in [from_measure, to_measure]]
  [start, end] = [fn(from_indx, to_indx) + 1 for fn in [min, max]]
  operation_key = 'dec' if from_indx > to_indx else 'inc'
  operation = system['updater'][operation_key]

  return reduce(lambda acc, a: operation(acc, a), units[start:end], 1)


def find_system(measure):
  for system_name, system_types in measure_system.items():
    # system_types = measure_system.imperial, measure_system.metric
    for system_type, system_value in system_types.items():
      # system_value = imperial.length, imperial.weight, metric.length, etc
      if measure in system_value['units']:
        return {
          'name': system_name,
          'type': system_type,
          'value': system_value,
        }

  raise TypeError("Something's wrong.. and it's not me")


def convert(from_data, to_measure):
  [from_value_str, from_measure] = re.findall(r'[^A-Za-z]+|[A-Za-z]+', from_data)
  # from_measure = 'km'
  from_value = float(from_value_str) # 1.0
  from_system = find_system(from_measure) # {'name': 'metric', 'type': 'length', 'value': { 'unit': [...] } }
  to_system = find_system(to_measure)
  if (from_system['type'] != to_system['type']):
    raise TypeError("Not cool! you don't mix potatos and chocolate!")
    #return "Not cool! you don't mix potatos and chocolate!"

  if from_system['name'] == to_system['name']:
    distance = get_distance(to_system['value'], from_measure, to_measure)
    result = from_value * distance
  else:
    strategy = sys_to_sys[to_system['type']]
    operation = strategy.get(f'{from_system['name']}_to_{to_system['name']}')
    from_scale = strategy[from_system['name']]
    to_scale = strategy[to_system['name']]
    from_distance = get_distance(from_system['value'], from_measure, from_scale)
    to_distance = get_distance(to_system['value'], to_measure, to_scale)
    result = operation(from_value * from_distance / to_distance, strategy['by'])

  return f'{round(result, 5)}{to_measure}'


class TestConvert(unittest.TestCase):
    def test_km_to_m(self):
        self.assertEqual(convert('1km', 'm'), '1000.0m')

    def test_m_to_km(self):
        self.assertEqual(convert('1m', 'km'), '0.001km')

    def test_in_to_yd(self):
        self.assertEqual(convert('1in', 'yd'), '0.02778yd')

    def test_yd_to_in(self):
        self.assertEqual(convert('1yd', 'in'), '36.0in')

    def test_in_to_cm(self):
        self.assertEqual(convert('1in', 'cm'), '2.54cm')

    def test_mi_to_yd(self):
        self.assertEqual(convert('1mi', 'yd'), '1760.0yd')

    def test_cm_to_in(self):
        self.assertEqual(convert('1cm', 'in'), '0.3937in')

    def test_mi_to_km(self):
        self.assertEqual(convert('1mi', 'km'), '1.60934km')
    
    def test_different_system_of_units(self):
        # Check if a TypeError is raised when arguments are not supported
        with self.assertRaises(TypeError):
            convert('1km','g')
        
        with self.assertRaises(TypeError):
            convert('1kg', 'm')
if __name__ == '__main__':
    unittest.main()