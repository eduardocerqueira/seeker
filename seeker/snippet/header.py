#date: 2022-04-25T17:09:30Z
#url: https://api.github.com/gists/74fd3c4aed2febf2e438b7f6db581165
#owner: https://api.github.com/users/decentralfarm

headers = {
  "Content-Type": "application/json",
  "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.77 Safari/537.36" }
web3 = Web3(Web3.HTTPProvider('https://proxy.roninchain.com/free-gas-rpc',
    request_kwargs={ "headers": headers }))