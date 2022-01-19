#date: 2022-01-19T17:04:15Z
#url: https://api.github.com/gists/94bc1483a3efa399b2f5052ff3827f24
#owner: https://api.github.com/users/mgaitan

### WORK IN PROGRESS . NOT WORKING


from dataclasses import dataclass, replace


@dataclass(frozen=True)
class A:
	name: str

@dataclass(frozen=True)
class B:
	a: A
	other: int

some_b = B(a=A(name="tin"), other=12)
other_b = replace(some_b, other=1)


other_a = replace(some_b.a, name="nati")
b_with_new_a = replace(some_b, a=other_a)


assert other_b.a.name == "tin"
assert b_with_new_a.a.name == "nati"


def deep_replace(obj, /, **kwargs):
	
	for k, v in kwargs.items():		
		k = k.replace("__", ".")
		while "." in k:
			print("")
			prefix, _, attr = k.rpartition(".")
			print()
			v = dataclasses.replace(attrgetter(prefix)(obj), **{attr: v})
			k = prefix
	return v
