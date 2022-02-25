#date: 2022-02-25T16:56:10Z
#url: https://api.github.com/gists/f12f7f5be78826b1e6ac416bd8f78a88
#owner: https://api.github.com/users/ssime-git

if employeeid:
        query += ' EmployeeId=? AND'
        to_filter.append(employeeid)
if lastname:
        query += ' LastName=? AND'
        to_filter.append(lastname)
if city:
        query += ' City=? AND'
        to_filter.append(city)