#date: 2022-10-21T17:08:31Z
#url: https://api.github.com/gists/55e671d82ed6114682b090b02b0d0358
#owner: https://api.github.com/users/merve-ozturk

countryCode1 = {
    1 : "United States",
    44 : "United Kingdom",
    971 : "United Arab Emirates",
    7 : "Russia"
}

countryCode2 = {
    90 : "Turkiye",
    47 : "Norway",
    49 : "Germany",
    420 : "Czech Republic"
}

new_dict = {**countryCode1, **countryCode2}

print(new_dict)
#{1: 'United States', 44: 'United Kingdom', 971: 'United Arab Emirates', 7: 'Russia', 90: 'Turkiye', 47: 'Norway', 49: 'Germany', 420: 'Czech Republic'}