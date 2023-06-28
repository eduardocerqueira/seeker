#date: 2023-06-28T16:50:14Z
#url: https://api.github.com/gists/b5aedecf09ab51ed614cf5259cb80b13
#owner: https://api.github.com/users/elicharlese

# USING BINARY SEARCH
class RecentCounter:
    def __init__(self):
        self.arr = []

    def ping(self, t: int) -> int:
        self.arr.append(t)
        start = t - 3000
        if(t<=0):
            return len(self.arr)
      # find t which is >= start in arr
        def binSearch(start,arr):
            i = 0
            j = len(arr)
            while(i<=j):
                mid = (i+j)//2
                if(arr[mid] > start):
                    j = mid - 1
                elif(arr[mid] < start):
                    i = mid + 1
                else:
                    return mid
            return i
        
        indx = binSearch(start,self.arr)
        return len(self.arr) - indx