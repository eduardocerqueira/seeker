#date: 2022-05-20T17:08:36Z
#url: https://api.github.com/gists/e10bdb462a98dc79e1df57c0bafbea7a
#owner: https://api.github.com/users/seba-ban

from typing import Callable, Union, List, TypeVar

T = TypeVar("T")
P = TypeVar("P")
ListOrValue = List[Union[T, 'ListOrValue[T]']]

def map_recursive(
  l: ListOrValue[T], 
  func: Callable[[T], P]
) -> ListOrValue[P]:
  mapped = []
  
  for el in l:
    if isinstance(el, list):
      mapped.append(map_recursive(el, func))
    else:
      mapped.append(func(el))

  return mapped

list_of_strings = [
  'aaa',
  'a',
  [
    [
      'vsdvsdvs',
      'verveve',
      [
        'popopopop',
        [
          'geevwre'
        ],
        'grewgerw'
      ]
    ],
    'vrvrvr'
  ],
  'ewfe'
]

string_lengths = map_recursive(list_of_strings, lambda s: len(s))
print(string_lengths)
# [3, 1, [[8, 7, [9, [7], 8]], 6], 4]