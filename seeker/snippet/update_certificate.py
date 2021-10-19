#date: 2021-10-19T16:55:45Z
#url: https://api.github.com/gists/d99d832ff9d7c28c6d7abb69d0fa4225
#owner: https://api.github.com/users/simranjaising30

import datetime

from eshares.issuables.certificates.models import Certificate

Certificate.objects.filter(id=8337236).update(
  eshares_issued_paper_certificate=False,
  eshares_issued_paper_certificate_tracking=" 03/26/2021 FedEx: 773280116514 Case # 00474004  then 10/18/21- Remove paper tag 00516613",
  eshares_issued_paper_certificate_date=None,
)