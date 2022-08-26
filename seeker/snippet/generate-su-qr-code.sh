#date: 2022-08-26T16:46:01Z
#url: https://api.github.com/gists/eea14e8cd6e502d2b1a85f6f757c73af
#owner: https://api.github.com/users/NateCross

#!/bin/bash

# How SU does QR codes:
# Basically, it's a POST request to a certain link
# that was discovered through checking the source code of the website
# Emulating the values of these forms and using them with the curl command
# allows us to bypass the site and yoink the qr code image immediately.

qr_img_filename="qr-image.png"

link="https://my.su.edu.ph/mysilliman/public/"
qrlanding_link="qrlanding.php?area=1"
form_link="qrprocess.php"
full_link="${link}${form_link}"

# Change these values
firstname="Nathan Angelo"
middlename="Balubar"
lastname="Cruz"
idno="20-1-02114"
address="Vernon Hall, Silliman University, Dumaguete City"
contactno="09691800593"
email="nathanbcruz@su.edu.ph"
# fsavo is the form's name for Category
fsavo="D"
age="20"
gender="M"
department="College of Computer Studies"
temp="36.6"

# You probably shouldn't change these
# so that you won't face trouble with the people checking the code
fever="0"
cough="0"
losssmelltaste="0"
diarrhea="0"
nausea="0"
sorethroat="0"
ncongestion="0"
headache="0"
fatigue="0"
muscleache="0"
shortbreath="0"
vaccinated="1"

main() {
    local content=$(curl -X POST "${full_link}" \
        -d "firstname=${firstname}" \
        -d "middlename=${middlename}" \
        -d "lastname=${lastname}" \
        -d "idno=${idno}" \
        -d "address=${address}" \
        -d "contactno=${contactno}" \
        -d "email=${email}" \
        -d "fsavo=${fsavo}" \
        -d "age=${age}" \
        -d "gender=${gender}" \
        -d "department=${department}" \
        -d "temp=${temp}" \
        -d "fever=${fever}" \
        -d "cough=${cough}" \
        -d "losssmelltaste=${losssmelltaste}" \
        -d "diarrhea=${diarrhea}" \
        -d "nausea=${nausea}" \
        -d "sorethroat=${sorethroat}" \
        -d "ncongestion=${ncongestion}" \
        -d "headache=${headache}" \
        -d "fatigue=${fatigue}" \
        -d "muscleache=${muscleache}" \
        -d "shortbreath=${shortbreath}" \
        -d "vaccinated=${vaccinated}")

    local qr_img_src=$(echo "${content}" | pup '#qrimgholder json{}' | jq -r '.[0].src')

    echo "${link}${qr_img_src}"

}

cleanup() {
    rm qr-image
}

output=$(main)

# Saves to a file called qr-image.jpg
curl "${output}" > "${qr_img_filename}"

if [[ "${TERM}" = "xterm-kitty" ]]; then
    kitty +kitten icat "${qr_img_filename}"
fi

if [[ "${OSTYPE}" = "linux-android" ]]; then
    termimage "${qr_img_filename}"
fi
