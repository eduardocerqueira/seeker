#date: 2021-10-07T16:55:48Z
#url: https://api.github.com/gists/4714cf4849749ae3d48473bc02ef1788
#owner: https://api.github.com/users/mtholder

#!/bin/bash
if ! test -d cruft ; then
    mkdir cruft
fi
for i in ott*.tre ; do
    j=`echo ${i} | sed -E 's/^ott//' | sed -E 's/\.tre//'`
    echo $j >&2
    tail -n1 ${i} > cruft/tax.tre
    ntax=`otc-degree-distribution cruft/tax.tre 2>&1 | tail -n2 | head -n1 | awk '{print $1}'`
    nsoln=`otc-degree-distribution ../subproblem_solutions/${i} 2>&1 | tail -n2 | head -n1 | awk '{print $1}'`
    echo -e "$j\thttps://tree.opentreeoflife.org/opentree/argus/opentree13.4@ott${j}\t${ntax}\t${nsoln}"
done
