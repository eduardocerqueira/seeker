#date: 2021-12-03T17:02:47Z
#url: https://api.github.com/gists/a62c2170b4e650298f1e9f9c22f49b1a
#owner: https://api.github.com/users/dgtlctzn

async with aiohttp.ClientSession(
    headers=headers_dict,
    cookies=cookies_dict,
    trace_configs=[trace_config],
) as session:
    ...