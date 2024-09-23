#date: 2024-09-23T16:55:00Z
#url: https://api.github.com/gists/3aae90e5e09f6ec0dad6084019d608ec
#owner: https://api.github.com/users/r12habh

from types import prepare_class
from typing import List

class NestedInteger:
    def __init__(self, value=None):
        if value is None:
            self.is_int = False
            self.list = []
        else:
            self.is_int = True
            self.value = value

    def isInteger(self):
        return self.is_int

    def add(self, elem):
        if not self.is_int:
            self.list.append(elem)

    def setInteger(self, value):
        self.is_int = True
        self.value = value
        self.list = []  # clear the list if it was previously a list

    def getInteger(self):
        return self.value if self.is_int else None

    def getList(self):
        return self.list if not self.is_int else None


class Solution:
    def depthSum(self, nestedList: List[NestedInteger]) -> int:
        # Helper funciton to perform depth-first search
        def dfs(nested_list, depth):
            # Initialize a variable to keep track of the depth sum
            depth_sum = 0

            # Iterate through each item in the current nested list
            for item in nested_list:
                # Check if the item is an integer
                if item.isInteger():
                    # If it's integer, add its value multiplied by the current depth
                    depth_sum += item.getInteger() * depth
                else:
                    # If it's a nested list, recurslively call dfs for that list, increasing the depth by 1
                    depth_sum += dfs(item.getList(), depth + 1)

            # Return the accumulated depth sum for this level of nesting
            return depth_sum
        
        # Start the DFS with the initial depth of 1
        return dfs(nestedList, 1)



# Test cases
def test_depth_sum():
    # Example 1
    nested_list1 = [
        NestedInteger(),  # Representing the list [1, 1]
        NestedInteger(2),  # Representing the integer 2
        NestedInteger()   # Representing the list [1, 1]
    ]
    nested_list1[0].add(NestedInteger(1))
    nested_list1[0].add(NestedInteger(1))
    nested_list1[2].add(NestedInteger(1))
    nested_list1[2].add(NestedInteger(1))

    sol1 = Solution()
    assert sol1.depthSum(nested_list1) == 10  # 1*2 + 1*2 + 2*1 + 1*2 + 1*2

    # Example 2
    nested_list2 = [
        NestedInteger(1),
        NestedInteger()
    ]
    nested_list2[1].add(NestedInteger(4))
    nested_list2[1].add(NestedInteger())
    nested_list2[1].getList()[1].add(NestedInteger(6))

    sol2 = Solution()
    assert sol2.depthSum(nested_list2) == 27  # 1*1 + 4*2 + 6*3

    # Example 3
    nested_list3 = [NestedInteger(0)]
    sol3 = Solution()
    assert sol3.depthSum(nested_list3) == 0  # 0 at depth 1

    # Additional Test Case 1: Multiple integers at various depths
    nested_list4 = [
        NestedInteger(),  # Representing the list [1, 2]
        NestedInteger(3)
    ]
    nested_list4[0].add(NestedInteger(1))
    nested_list4[0].add(NestedInteger(2))

    sol4 = Solution()
    assert sol4.depthSum(nested_list4) == 9  # (1*2 + 2*2 + 3*1)

    # Additional Test Case 2: Deeply nested structure
    nested_list5 = [
        NestedInteger()
    ]
    nested_list5[0].add(NestedInteger())
    nested_list5[0].getList()[0].add(NestedInteger(1))

    sol5 = Solution()
    assert sol5.depthSum(nested_list5) == 3  # 1*3

    # Additional Test Case 3: Mixed nesting and integers
    nested_list6 = [
        NestedInteger(1),  # Depth 1
        NestedInteger()    # Nested list
    ]
    nested_list6[1].add(NestedInteger(2))  # Depth 2
    nested_list6[1].add(NestedInteger())   # Another nested list
    nested_list6[1].getList()[1].add(NestedInteger(3))  # Depth 3

    sol6 = Solution()
    assert sol6.depthSum(nested_list6) == 14  # 1*1 + 2*2 + 3*3

    # Additional Test Case 4: Empty nested list
    nested_list7 = []
    sol7 = Solution()
    assert sol7.depthSum(nested_list7) == 0  # No integers

    # Additional Test Case 5: Complex nesting with zeros
    nested_list8 = [
        NestedInteger(),
        NestedInteger(2)
    ]
    nested_list8[0].add(NestedInteger(0))
    nested_list8[0].add(NestedInteger(1))

    sol8 = Solution()
    assert sol8.depthSum(nested_list8) == 4  # (0*2 + 1*2 + 2*1)

    # Additional Test Case 6: All nested integers are zeros
    nested_list9 = [
        NestedInteger()
    ]
    nested_list9[0].add(NestedInteger(0))
    nested_list9[0].add(NestedInteger())
    nested_list9[0].getList()[1].add(NestedInteger(0))

    sol9 = Solution()
    assert sol9.depthSum(nested_list9) == 0  # All zeros

    # Additional Test Case 7: Very deep nesting with one integer
    nested_list10 = [
        NestedInteger()
    ]
    nested_list10[0].add(NestedInteger())
    nested_list10[0].getList()[0].add(NestedInteger())
    nested_list10[0].getList()[0].getList()[0].add(NestedInteger(5))

    sol10 = Solution()
    assert sol10.depthSum(nested_list10) == 20  # 5*4

    print("All test cases passed!")


# Run tests
test_depth_sum()
