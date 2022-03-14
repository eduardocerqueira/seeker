#date: 2022-03-14T16:54:28Z
#url: https://api.github.com/gists/a652b93276b8b1c3bc5dca03c17153fd
#owner: https://api.github.com/users/ihorCholiy

coinName: str = 'ethereum'  # ethereum-classic, zcash, ravencoin etc.
interval: str = 'd1'  # m1, m5, m15, m30, h1, h2, h6, h12, d1
startInMillis: int = 1615740722000
endInMillis: int = 1647276645491
url: str = f"https://api.coincap.io/v2/assets/{coinName}/history?interval={interval}&start={startInMillis}&end={endInMillis}"