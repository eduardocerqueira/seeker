#date: 2022-02-09T17:05:26Z
#url: https://api.github.com/gists/5ca7d10c2e9efb12d5fa5ce121f9faf1
#owner: https://api.github.com/users/valeriacarballo

def getWavelengthCounts (file_path: str):
    wavelength = pd.read_csv(file_path, usecols=['wavelength'])
    counts = pd.read_csv(file_path, usecols=['counts'])
    return[wavelength, counts]