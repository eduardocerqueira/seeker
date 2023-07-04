#date: 2023-07-04T17:06:44Z
#url: https://api.github.com/gists/84a8aee26e21ce67b8313e1579d4a9c4
#owner: https://api.github.com/users/victorbrca

{
  printf '%-51s%s\n' 'PACKAGE NAME' 'DATE'
  printf '%0.s=' $(seq 1 50)
  printf ' '
  printf '%0.s=' $(seq 1 32)
  printf '\n'
  rpm -qa --qf '%{INSTALLTIME} %-50{NAME} %{INSTALLTIME:date}\n' | sort -nr | cut -d' ' -f2- 
} | less