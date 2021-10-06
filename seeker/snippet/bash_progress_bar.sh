#date: 2021-10-06T16:59:20Z
#url: https://api.github.com/gists/a910f11e0dad94594d7f5b044b569d11
#owner: https://api.github.com/users/macsmax

# Function to print a progress barr, pass three arguments: total counter, current counter, start epoch
function progressbar {
        TOT_COUNT=${1}
        COUNTER=${2}
        STARTEPOCH=${3}
        CUREPOCH=`date +%s`
        # get the percentage
        let CUR_PERC=$COUNTER*100/$TOT_COUNT
        # get the elapsed seconds
        let DELTA_SECONDS=$CUREPOCH-$STARTEPOCH
        let DELTA_SECONDS_MOD=$DELTA_SECONDS%60
        let DELTA_MINUTES=$DELTA_SECONDS/60
        let DELTA_MINUTES_MOD=$DELTA_MINUTES%60
        let DELTA_HOURS=$DELTA_MINUTES/60
        let DELTA_HOURS_MOD=$DELTA_HOURS%60

        # get the estimated end time
        let EST_SECONDS=$DELTA_SECONDS*$TOT_COUNT/$COUNTER
        let REMAINING_SECONDS_MOD=$EST_SECONDS%60
        let REMAINING_MINUTES=$EST_SECONDS/60
        let REMAINING_MINUTES_MOD=$REMAINING_MINUTES%60
        let REMAINING_HOURS=$REMAINING_MINUTES/60
        let REMAINING_HOURS_MOD=$REMAINING_HOURS%60

        # get the eta time going to 0
        let ETA_SECONDS=$EST_SECONDS-$DELTA_SECONDS
        let ETA_SECONDS_MOD=$ETA_SECONDS%60
        let ETA_MINUTES=$ETA_SECONDS/60
        let ETA_MINUTES_MOD=$ETA_MINUTES%60
        let ETA_HOURS=$ETA_MINUTES/60
        let ETA_HOURS_MOD=$ETA_HOURS%60

        # how many dashes out of 20
        let DASHES_PERC=$CUR_PERC/5
        let DASHES=$(printf "%d" $DASHES_PERC)
        let WHITESPACES=21-$DASHES
        PROGRESSBARR_D=$(awk -v i=$DASHES 'BEGIN { OFS="#"; $i="#"; print }')
        PROGRESSBARR_W=$(awk -v i=$WHITESPACES 'BEGIN { OFS=" "; $i=" "; print }')
        PROGRESSBARR="${PROGRESSBARR_D}${PROGRESSBARR_W}"

        runtime=$(printf "Runtime: %02d:%02d:%02d" $DELTA_HOURS_MOD $DELTA_MINUTES_MOD $DELTA_SECONDS_MOD)
        completion=$(printf "Completion: %02d:%02d:%02d" $REMAINING_HOURS_MOD $REMAINING_MINUTES_MOD $REMAINING_SECONDS_MOD)
        etatime=$(printf "ETA: %02d:%02d:%02d" $ETA_HOURS_MOD $ETA_MINUTES_MOD $ETA_SECONDS_MOD)

        echo -ne "${PROGRESSBARR}(${CUR_PERC}%) ${runtime} ${completion} ${etatime}\r"
}