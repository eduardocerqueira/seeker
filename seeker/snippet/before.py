#date: 2023-10-24T16:53:25Z
#url: https://api.github.com/gists/fcbb2136e5e3e803849a594a9b375bf2
#owner: https://api.github.com/users/xNatthapol

class Calculator:
    def divide(self, numerator, denominator):
        try:
            result = numerator / denominator
        except ZeroDivisionError:
            result = "Cannot divide by zero"
        return result