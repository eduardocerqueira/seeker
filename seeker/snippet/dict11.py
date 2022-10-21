#date: 2022-10-21T17:03:41Z
#url: https://api.github.com/gists/7f527b983a70679f9b9a4d6838edb7bf
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

countryCode1.update(countryCode2)

print(countryCode1)
#{1: 'United States', 44: 'United Kingdom', 971: 'United Arab Emirates', 7: 'Russia', 90: 'Turkiye', 47: 'Norway', 49: 'Germany', 420: 'Czech Republic'}

print(countryCode2)
#{90: 'Turkiye', 47: 'Norway', 49: 'Germany', 420: 'Czech Republic'}