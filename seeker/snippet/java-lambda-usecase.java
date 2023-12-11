//date: 2023-12-11T17:04:42Z
//url: https://api.github.com/gists/2a6dffe5962049010125ce9052212202
//owner: https://api.github.com/users/jcohen66

// Use Case					Examples

A boolean expression		(List<String> list) -> list.isEmpty()

  Creating Objects			() -> new Apple(10)
  
  
Consuming an object			(Apple a) -> {
  								System.out.println(a.getWeight());
							}

Select/extract from an object	(String s) -> s.length()
  
  Combine two values		(int a, int b) -> a * b
  
  Compare two objects		(Apple a1, Apple a2) -> a1.getWeight().compareTo(a2.getWeight())
  
  
 