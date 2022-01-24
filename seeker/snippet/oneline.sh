#date: 2022-01-24T16:55:53Z
#url: https://api.github.com/gists/f1b3fe9b0ec868fd4be33b53ba319cae
#owner: https://api.github.com/users/jul

#!/usr/bin/env bash
export LANG=C
function count() { python3 -c'(s,d)=(__import__("sys"),__import__("archery").mdict);print(sorted(sum((d({l[:-1]:1}) for l in s.stdin.readlines()),{}).items(),key=lambda k:k[1],reverse=True)[:10])'; }
cfl() { python3 -c'(s,t,d,f)=(__import__("sys"),__import__("tld"),__import__("archery").mdict,lambda l:t.get_fld(l[:-1],fix_protocol=True,fail_silently=True));print(sorted(sum((d({f(l):1}) for l in s.stdin.readlines()),{}).items(),key=lambda k:k[1],reverse=True)[:10])'; }
test_time() {
    func="$1"
    TIMEFORMAT="%U"
    printf " %.5f" $( echo "$( (time $func < this.sample 1> /dev/null)  2>&1 ) / $l" | bc -l );
}
(echo line_count  simple_count count_w_fld;
for((l=100; l<8000; l+=200)); do
    echo -n "$l"; head -n $l this.list > this.sample ;
    test_time count;test_time cfl;echo; 
done ) | (
    cat > /dev/shm/mytempfile && \
    trap 'rm /dev/shm/mytempfile' EXIT && \
    gnuplot -e "set terminal png; set output 'out.png'; plot for[col=2:3] '/dev/shm/mytempfile' using 1:col title columnheader(col) with lines")
#!/usr/bin/env bash
export LANG=C

curl -s https://raw.githubusercontent.com/justdomains/blocklists/master/lists/easyprivacy-justdomains.txt > this.list

function count() { python3 -c'(s,d)=(__import__("sys"),__import__("archery").mdict);print(sorted(sum((d({l[:-1]:1}) for l in s.stdin.readlines()),{}).items(),key=lambda k:k[1],reverse=True)[:10])'; }
cfl() { python3 -c'(s,t,d,f)=(__import__("sys"),__import__("tld"),__import__("archery").mdict,lambda l:t.get_fld(l[:-1],fix_protocol=True,fail_silently=True));print(sorted(sum((d({f(l):1}) for l in s.stdin.readlines()),{}).items(),key=lambda k:k[1],reverse=True)[:10])'; }
test_time() {
    func="$1"
    TIMEFORMAT="%U"
    printf " %.5f" $( echo "$( (time $func < this.sample 1> /dev/null)  2>&1 ) / $l" | bc -l );
}
(echo line_count  simple_count count_w_fld;
for((l=100; l<8000; l+=200)); do
    echo -n "$l"; head -n $l this.list > this.sample ;
    test_time count;test_time cfl;echo; 
done ) | (
    cat > /dev/shm/mytempfile && \
    trap 'rm /dev/shm/mytempfile' EXIT && \
    gnuplot -e "set terminal png; set output 'out.png'; plot for[col=2:3] '/dev/shm/mytempfile' using 1:col title columnheader(col) with lines")
echo result should look like this
cat <<EOF
   0.003 +-----------------------------------------------------------------+   
         |      +     +      +     +      +      +     +      +     +    ##|   
         |                                            simple_count ****##**|   
  0.0025 |-+                                           count_w_fld #######-|   
         |                                                       ####      |   
         |                                                  #####**        |   
         |                                             ### #**             |   
   0.002 |-+                                          #***#              +-|   
         |                                       #####                     |   
         |                                    ###***                       |   
  0.0015 |-+                                ##***                        +-|   
         |                            ######**                             |   
         |                       #####***                                  |   
   0.001 |#+                  ###   **                                   +-|   
         | #                ##******                                       |   
         |  #          #####**                                             |   
         |  #       ###**                                                  |   
  0.0005 |-+ #######***                                                  +-|   
         |*******                                                          |   
         |      +     +      +     +      +      +     +      +     +      |   
       0 +-----------------------------------------------------------------+   
         0     500   1000   1500  2000   2500   3000  3500   4000  4500   5000 
EOF