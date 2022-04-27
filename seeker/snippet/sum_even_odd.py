#date: 2022-04-27T17:10:31Z
#url: https://api.github.com/gists/639672b58b8c702687f9a1f919f8852e
#owner: https://api.github.com/users/oze4

def run():
  i = input("enter 20 numbers separated by spaces: ")
  iList = i.split(" ")
  validate_input(iList, 20)
  nums = list_str_to_list_int(iList)
  return calculate_sum(nums)

def validate_input(strList, expectedTotal):
  l = len(strList)
  if l < expectedTotal or l > expectedTotal:
    raise Exception("expected %s got %s" % expectedTotal, len(n))

def list_str_to_list_int(nl):
  return list(map(int, nl))

def calculate_sum(numList):
  sum = { "evenTotal": 0, "oddTotal": 0 }
  for n in numList:
    if is_even(n):
      sum["evenTotal"] += n
    else:
      sum["oddTotal"] += n
  return sum
    
def is_even(n):
  return n % 2 == 0

results = run()
print("sum of even nums: %s" % results["evenTotal"])
print("sum of odd nums: %s" % results["oddTotal"])