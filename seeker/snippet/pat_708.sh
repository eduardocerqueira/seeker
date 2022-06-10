#date: 2022-06-10T16:57:49Z
#url: https://api.github.com/gists/e806768fb5431cb78a469bf1b4a99530
#owner: https://api.github.com/users/jameskyle

#!/bin/bash
export TS_SLOTS=25

root=s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11
source_root=s3://spectus-truthsets/weekly-patterns-v2/release-2022-01-11-with-carriers-fix
monthly=${root/"weekly-"/}
monthly_source=${source_root/"weekly-"/}

function path {
    echo "y=2022/m=$1/d=$2"
}

function subs {
    echo $(aws s3 ls  $1/ | awk '/PRE/{gsub(/\//, "");print $2}')
}

for p in $(path 4 25) $(path 5 2) $(path 5 9);do
    for d in $(subs $root);do 
        echo ts aws s3 sync $root/$d/$p $root-bad-carrier-backup-PAT-708/$d/$p
        echo ts -d aws s3 rm --recursive $root/$d/$p
        echo ts -d aws s3 sync $source_root/$d/$p $root/$d/$p
        echo ""
    done
done

for d in $(subs $monthly);do 
    p=$d/y=2022/m=4
    echo ts aws s3 sync $monthly/$p $monthly-bad-carrier-backup-PAT-708/$p
    echo ts -d aws s3 rm --recursive $monthly/$p
    echo ts -d aws s3 sync $monthly_source/$p $monthly/$p
    echo ""
done


## OUTPUT
# ts aws s3 sync s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/home_panel_summary/y=2022/m=4/d=25 s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11-bad-carrier-backup-PAT-708/home_panel_summary/y=2022/m=4/d=25
# ts -d aws s3 rm --recursive s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/home_panel_summary/y=2022/m=4/d=25
# ts -d aws s3 sync s3://spectus-truthsets/weekly-patterns-v2/release-2022-01-11-with-carriers-fix/home_panel_summary/y=2022/m=4/d=25 s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/home_panel_summary/y=2022/m=4/d=25

# ts aws s3 sync s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/normalization_stats/y=2022/m=4/d=25 s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11-bad-carrier-backup-PAT-708/normalization_stats/y=2022/m=4/d=25
# ts -d aws s3 rm --recursive s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/normalization_stats/y=2022/m=4/d=25
# ts -d aws s3 sync s3://spectus-truthsets/weekly-patterns-v2/release-2022-01-11-with-carriers-fix/normalization_stats/y=2022/m=4/d=25 s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/normalization_stats/y=2022/m=4/d=25

# ts aws s3 sync s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/patterns/y=2022/m=4/d=25 s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11-bad-carrier-backup-PAT-708/patterns/y=2022/m=4/d=25
# ts -d aws s3 rm --recursive s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/patterns/y=2022/m=4/d=25
# ts -d aws s3 sync s3://spectus-truthsets/weekly-patterns-v2/release-2022-01-11-with-carriers-fix/patterns/y=2022/m=4/d=25 s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/patterns/y=2022/m=4/d=25

# ts aws s3 sync s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/release_metadata/y=2022/m=4/d=25 s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11-bad-carrier-backup-PAT-708/release_metadata/y=2022/m=4/d=25
# ts -d aws s3 rm --recursive s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/release_metadata/y=2022/m=4/d=25
# ts -d aws s3 sync s3://spectus-truthsets/weekly-patterns-v2/release-2022-01-11-with-carriers-fix/release_metadata/y=2022/m=4/d=25 s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/release_metadata/y=2022/m=4/d=25

# ts aws s3 sync s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/visit_panel_summary/y=2022/m=4/d=25 s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11-bad-carrier-backup-PAT-708/visit_panel_summary/y=2022/m=4/d=25
# ts -d aws s3 rm --recursive s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/visit_panel_summary/y=2022/m=4/d=25
# ts -d aws s3 sync s3://spectus-truthsets/weekly-patterns-v2/release-2022-01-11-with-carriers-fix/visit_panel_summary/y=2022/m=4/d=25 s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/visit_panel_summary/y=2022/m=4/d=25

# ts aws s3 sync s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/home_panel_summary/y=2022/m=5/d=2 s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11-bad-carrier-backup-PAT-708/home_panel_summary/y=2022/m=5/d=2
# ts -d aws s3 rm --recursive s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/home_panel_summary/y=2022/m=5/d=2
# ts -d aws s3 sync s3://spectus-truthsets/weekly-patterns-v2/release-2022-01-11-with-carriers-fix/home_panel_summary/y=2022/m=5/d=2 s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/home_panel_summary/y=2022/m=5/d=2

# ts aws s3 sync s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/normalization_stats/y=2022/m=5/d=2 s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11-bad-carrier-backup-PAT-708/normalization_stats/y=2022/m=5/d=2
# ts -d aws s3 rm --recursive s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/normalization_stats/y=2022/m=5/d=2
# ts -d aws s3 sync s3://spectus-truthsets/weekly-patterns-v2/release-2022-01-11-with-carriers-fix/normalization_stats/y=2022/m=5/d=2 s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/normalization_stats/y=2022/m=5/d=2

# ts aws s3 sync s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/patterns/y=2022/m=5/d=2 s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11-bad-carrier-backup-PAT-708/patterns/y=2022/m=5/d=2
# ts -d aws s3 rm --recursive s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/patterns/y=2022/m=5/d=2
# ts -d aws s3 sync s3://spectus-truthsets/weekly-patterns-v2/release-2022-01-11-with-carriers-fix/patterns/y=2022/m=5/d=2 s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/patterns/y=2022/m=5/d=2

# ts aws s3 sync s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/release_metadata/y=2022/m=5/d=2 s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11-bad-carrier-backup-PAT-708/release_metadata/y=2022/m=5/d=2
# ts -d aws s3 rm --recursive s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/release_metadata/y=2022/m=5/d=2
# ts -d aws s3 sync s3://spectus-truthsets/weekly-patterns-v2/release-2022-01-11-with-carriers-fix/release_metadata/y=2022/m=5/d=2 s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/release_metadata/y=2022/m=5/d=2

# ts aws s3 sync s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/visit_panel_summary/y=2022/m=5/d=2 s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11-bad-carrier-backup-PAT-708/visit_panel_summary/y=2022/m=5/d=2
# ts -d aws s3 rm --recursive s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/visit_panel_summary/y=2022/m=5/d=2
# ts -d aws s3 sync s3://spectus-truthsets/weekly-patterns-v2/release-2022-01-11-with-carriers-fix/visit_panel_summary/y=2022/m=5/d=2 s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/visit_panel_summary/y=2022/m=5/d=2

# ts aws s3 sync s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/home_panel_summary/y=2022/m=5/d=9 s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11-bad-carrier-backup-PAT-708/home_panel_summary/y=2022/m=5/d=9
# ts -d aws s3 rm --recursive s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/home_panel_summary/y=2022/m=5/d=9
# ts -d aws s3 sync s3://spectus-truthsets/weekly-patterns-v2/release-2022-01-11-with-carriers-fix/home_panel_summary/y=2022/m=5/d=9 s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/home_panel_summary/y=2022/m=5/d=9

# ts aws s3 sync s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/normalization_stats/y=2022/m=5/d=9 s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11-bad-carrier-backup-PAT-708/normalization_stats/y=2022/m=5/d=9
# ts -d aws s3 rm --recursive s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/normalization_stats/y=2022/m=5/d=9
# ts -d aws s3 sync s3://spectus-truthsets/weekly-patterns-v2/release-2022-01-11-with-carriers-fix/normalization_stats/y=2022/m=5/d=9 s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/normalization_stats/y=2022/m=5/d=9

# ts aws s3 sync s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/patterns/y=2022/m=5/d=9 s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11-bad-carrier-backup-PAT-708/patterns/y=2022/m=5/d=9
# ts -d aws s3 rm --recursive s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/patterns/y=2022/m=5/d=9
# ts -d aws s3 sync s3://spectus-truthsets/weekly-patterns-v2/release-2022-01-11-with-carriers-fix/patterns/y=2022/m=5/d=9 s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/patterns/y=2022/m=5/d=9

# ts aws s3 sync s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/release_metadata/y=2022/m=5/d=9 s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11-bad-carrier-backup-PAT-708/release_metadata/y=2022/m=5/d=9
# ts -d aws s3 rm --recursive s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/release_metadata/y=2022/m=5/d=9
# ts -d aws s3 sync s3://spectus-truthsets/weekly-patterns-v2/release-2022-01-11-with-carriers-fix/release_metadata/y=2022/m=5/d=9 s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/release_metadata/y=2022/m=5/d=9

# ts aws s3 sync s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/visit_panel_summary/y=2022/m=5/d=9 s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11-bad-carrier-backup-PAT-708/visit_panel_summary/y=2022/m=5/d=9
# ts -d aws s3 rm --recursive s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/visit_panel_summary/y=2022/m=5/d=9
# ts -d aws s3 sync s3://spectus-truthsets/weekly-patterns-v2/release-2022-01-11-with-carriers-fix/visit_panel_summary/y=2022/m=5/d=9 s3://safegraph-perm/weekly-patterns-v2/release-2022-01-11/visit_panel_summary/y=2022/m=5/d=9

# ts aws s3 sync s3://safegraph-perm/patterns-v2/release-2022-01-11/home_panel_summary/y=2022/m=4 s3://safegraph-perm/patterns-v2/release-2022-01-11-bad-carrier-backup-PAT-708/home_panel_summary/y=2022/m=4
# ts -d aws s3 rm --recursive s3://safegraph-perm/patterns-v2/release-2022-01-11/home_panel_summary/y=2022/m=4
# ts -d aws s3 sync s3://spectus-truthsets/patterns-v2/release-2022-01-11-with-carriers-fix/home_panel_summary/y=2022/m=4 s3://safegraph-perm/patterns-v2/release-2022-01-11/home_panel_summary/y=2022/m=4

# ts aws s3 sync s3://safegraph-perm/patterns-v2/release-2022-01-11/neighborhood_home_panel_summary/y=2022/m=4 s3://safegraph-perm/patterns-v2/release-2022-01-11-bad-carrier-backup-PAT-708/neighborhood_home_panel_summary/y=2022/m=4
# ts -d aws s3 rm --recursive s3://safegraph-perm/patterns-v2/release-2022-01-11/neighborhood_home_panel_summary/y=2022/m=4
# ts -d aws s3 sync s3://spectus-truthsets/patterns-v2/release-2022-01-11-with-carriers-fix/neighborhood_home_panel_summary/y=2022/m=4 s3://safegraph-perm/patterns-v2/release-2022-01-11/neighborhood_home_panel_summary/y=2022/m=4

# ts aws s3 sync s3://safegraph-perm/patterns-v2/release-2022-01-11/neighborhood_patterns/y=2022/m=4 s3://safegraph-perm/patterns-v2/release-2022-01-11-bad-carrier-backup-PAT-708/neighborhood_patterns/y=2022/m=4
# ts -d aws s3 rm --recursive s3://safegraph-perm/patterns-v2/release-2022-01-11/neighborhood_patterns/y=2022/m=4
# ts -d aws s3 sync s3://spectus-truthsets/patterns-v2/release-2022-01-11-with-carriers-fix/neighborhood_patterns/y=2022/m=4 s3://safegraph-perm/patterns-v2/release-2022-01-11/neighborhood_patterns/y=2022/m=4

# ts aws s3 sync s3://safegraph-perm/patterns-v2/release-2022-01-11/normalization_stats/y=2022/m=4 s3://safegraph-perm/patterns-v2/release-2022-01-11-bad-carrier-backup-PAT-708/normalization_stats/y=2022/m=4
# ts -d aws s3 rm --recursive s3://safegraph-perm/patterns-v2/release-2022-01-11/normalization_stats/y=2022/m=4
# ts -d aws s3 sync s3://spectus-truthsets/patterns-v2/release-2022-01-11-with-carriers-fix/normalization_stats/y=2022/m=4 s3://safegraph-perm/patterns-v2/release-2022-01-11/normalization_stats/y=2022/m=4

# ts aws s3 sync s3://safegraph-perm/patterns-v2/release-2022-01-11/patterns/y=2022/m=4 s3://safegraph-perm/patterns-v2/release-2022-01-11-bad-carrier-backup-PAT-708/patterns/y=2022/m=4
# ts -d aws s3 rm --recursive s3://safegraph-perm/patterns-v2/release-2022-01-11/patterns/y=2022/m=4
# ts -d aws s3 sync s3://spectus-truthsets/patterns-v2/release-2022-01-11-with-carriers-fix/patterns/y=2022/m=4 s3://safegraph-perm/patterns-v2/release-2022-01-11/patterns/y=2022/m=4

# ts aws s3 sync s3://safegraph-perm/patterns-v2/release-2022-01-11/visit_panel_summary/y=2022/m=4 s3://safegraph-perm/patterns-v2/release-2022-01-11-bad-carrier-backup-PAT-708/visit_panel_summary/y=2022/m=4
# ts -d aws s3 rm --recursive s3://safegraph-perm/patterns-v2/release-2022-01-11/visit_panel_summary/y=2022/m=4
# ts -d aws s3 sync s3://spectus-truthsets/patterns-v2/release-2022-01-11-with-carriers-fix/visit_panel_summary/y=2022/m=4 s3://safegraph-perm/patterns-v2/release-2022-01-11/visit_panel_summary/y=2022/m=4

# ts aws s3 sync s3://safegraph-perm/patterns-v2/release-2022-01-11/visitors_to_brand_summary/y=2022/m=4 s3://safegraph-perm/patterns-v2/release-2022-01-11-bad-carrier-backup-PAT-708/visitors_to_brand_summary/y=2022/m=4
# ts -d aws s3 rm --recursive s3://safegraph-perm/patterns-v2/release-2022-01-11/visitors_to_brand_summary/y=2022/m=4
# ts -d aws s3 sync s3://spectus-truthsets/patterns-v2/release-2022-01-11-with-carriers-fix/visitors_to_brand_summary/y=2022/m=4 s3://safegraph-perm/patterns-v2/release-2022-01-11/visitors_to_brand_summary/y=2022/m=4

