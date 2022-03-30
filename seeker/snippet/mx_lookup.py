#date: 2022-03-30T16:59:46Z
#url: https://api.github.com/gists/7e151c639f6cbd9b13223781421cb60c
#owner: https://api.github.com/users/lesiki

import dns.resolver

input_domains = [
  "1.com",
  "2.com",
  "3.com",
  "etc.com",
]

for domain in input_domains:
    answer = None
    try:
        answers = dns.resolver.resolve(domain, 'MX')
        answers = sorted(answers, key = lambda a: a.preference)
        answer = answers[0] if len(answers) > 0 else None
    except Exception as e:
        answer = None
    if answer is None:
        print(f"{domain},,")
    else:
        best_guess = ""
        exchange = str(answer.exchange)
        if "google" in exchange:
            best_guess = 'google'
        elif "outlook.com" in exchange:
            best_guess = "outlook"
        elif "zoho" in exchange:
            best_guess = "zoho"
        print(f"{domain}, {exchange}, {best_guess}")
