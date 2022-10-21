#date: 2022-10-21T17:15:04Z
#url: https://api.github.com/gists/9af4a42171144a0877288a45c5145c1b
#owner: https://api.github.com/users/jeyraof

function hds {
    arrs=("20221029" "20221030" "20221110" "20221112" "20221113")

    echo
    echo "  화담숲 체크: https://m.hwadamsup.com/reservation/checkTermsAgree.do"
    echo

    for value in "${arrs[@]}"; do
        result=$(curl -s -d "selDate=$value&itemCode=00001" -X POST https://hwadamsup.com/mReserve/reserveInfo.do | jq '.timeList | map(select(.reQuantity >= 2)) | map(.startTime) | join(", ")')
        echo "  $value: $result"
    done
}