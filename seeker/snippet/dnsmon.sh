#date: 2022-02-22T16:58:46Z
#url: https://api.github.com/gists/4ced671e2d4a7cc28a427446c9ddb3ac
#owner: https://api.github.com/users/deoxykev

# dns servers, comma separated                                         test query                                  send email on failure                                                                                                          email client       notification addr
echo -n "1.1.1.1,1.0.0.1" | xargs -d',' -I{} sh -c 'dig +timeout=5 @{} google.com | grep -q NOERROR || echo -e "To:me@smtp.local \nSubject:{} DNS server is unresponsive\n\nAs of $(date), {} is not responding to DNS queries for google.com." | /usr/sbin/sendmail me@smtp.local'