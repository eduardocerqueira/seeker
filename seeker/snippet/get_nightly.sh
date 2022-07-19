#date: 2022-07-19T17:14:21Z
#url: https://api.github.com/gists/5a4de7e2b6043cf950856b330287eec8
#owner: https://api.github.com/users/eldersixpack13

#!/bin/sh


#	SCRIPT_VERSION="0.10";

#	Build Date 19.07.2022
#
#	author: JoeAverage
#
#	script is under GPL
#
########################################################################
#
#	What it does:
#	=============
#	mainly: downloads LibreELEC nightlies for supported platforms [*] from the download server
#	[*] (currently only) Generic, Generic-legacy, RPi2, RPi4 
#
#	- reads info from currently running nightly
#	- checks the download server for available nightlies images for an update/downgrade
#	- updates: to the newest (I hope this plays well when jumping over several nightlies in one step !)
#	- downgrades: one nightly before currently running (stepwise one image down)   
#	- checks for de-compression needed disk space when the LE updater runs (549 MB) and if okay:
#		- downloads a chosen nightly image to ~/.update/
#		- checksum image (OFF for now) and the update installer does this anyway
#	- 	offers to reboot
#
#	TODO
#	====
#	- test Generic-legacy, RPi4, RPi2, others ?
#	- test with lower disk space
#
########################################################################
#	VARs
########################################################################

SCRIPT_VERSION="0.10";

DEBUG=1;	# writes debug output; NO image will be downloaded (simulate it only) !
DEBUG=0;	# if un-commented: debug is OFF

LE_RELEASE="11.0";							# DON'T touch !!!

SERVER="https://test3.libreelec.tv/json";	# DON'T touch !!!
# 	SERVER="https://test.libreelec.tv/json";	# DON'T touch !!!

SUPPORTED_HW="Generic|Generic-legacy|RPi";	# DON'T touch !!!

PARTITION="/storage";
UPDATE_DIR="/storage/.update";
ETC_OS_RELEASE="/etc/os-release";

WORK_DIR="/storage/";
MY_TMPDIR="$(mktemp -d -p ${WORK_DIR})";
AVAILABLE_IMAGES="${MY_TMPDIR}/available_images";
POTENTIAL_IMAGES="${MY_TMPDIR}/potenial_images";

########################################################################
#
#	Functions
#
########################################################################

error_warning() {

	printf %b "\n\n\t ${1} \n\n";
	clean_up;
	exit 13;
}
########################################################################

print_message() {

	printf %b "\n ${1} \n";

	return 0;
}
########################################################################

clean_up() {

	[ ${DEBUG} -gt 0 ] && print_message "+++ in Function clean up +++";

	[ -d ${UPDATE_DIR} ] && rm -f ${UPDATE_DIR}/*;

	[ -d "${MY_TMPDIR}" ] && rm -rf "${MY_TMPDIR}";

	printf %b "\n";

	return 0;
}
########################################################################

get_HW_info(){

	if [ -s ${ETC_OS_RELEASE} ]; then
		PLATFORM="$(cat ${ETC_OS_RELEASE} | grep LIBREELEC_ARCH | sed 's/"/ /g' | cut -d " " -f2 | sed 's/ //g')";
		RUNNING_VERSION="$(cat ${ETC_OS_RELEASE} | grep ^VERSION | grep nightly | awk -F '"' '{print $2}' | sed 's/nightly-//g' | sed 's/ //g')";
		RUNNING_VERSION_DATE="$(echo "${RUNNING_VERSION}" | awk -F "-" '{print $1}' | sed 's/ //g')";
		GIT_HASH_INSTALLED="$(echo "${RUNNING_VERSION}" | awk -F "-" '{print $2}' | sed 's/ //g')";
		DEVICE="$(cat ${ETC_OS_RELEASE} | grep "LIBREELEC_DEVICE=" | awk -F '"' '{print $2}' | sed 's/ //g')";

		if [ ${DEBUG} -gt 0 ]; then
			print_message "+++ Debug Info +++";
			print_message "\t your Platform is:    ${PLATFORM}";
			print_message "\t your Device:         ${DEVICE}";
			print_message "\t build date is:       ${RUNNING_VERSION_DATE}";
			print_message "\t Git Hash is:         ${GIT_HASH_INSTALLED}";
			print_message "\t Version is:          ${RUNNING_VERSION}";
			print_message "\n";
		fi;

	else
		error_warning "can't find your ${ETC_OS_RELEASE} \n\n\t Aborting !!!";
	fi;

}
########################################################################

check_HW_Support() {

	# handle only SUPPORTED_HW (see section VARs), deny other variants
	RC="$(echo "${PLATFORM}" | grep -iE ${SUPPORTED_HW})";

	# -z STRING: the length of STRING is zero
	[ -z "${RC}" ] && error_warning "this script only supports Generic* and RPi's. \n\n\t Aborting !!!";

}
########################################################################

create_ServerURL() {
 
	# create the download URL for the running HW
	PROJECT="$(cat ${ETC_OS_RELEASE} | grep LIBREELEC_PROJECT | sed 's/"/ /g' | cut -d " " -f2 | sed 's/ //g')";

	DOWNLOAD_URL="${SERVER}/${LE_RELEASE}/${PROJECT}/${DEVICE}/";   # the last slash is important, otherwise NOTHING will be downloaded

	if [ ${DEBUG} -gt 0 ]; then
		print_message "+++ Debug Info +++";
		print_message "\t using download URL:\t ${DOWNLOAD_URL}";
	fi
}

########################################################################

check_available_Images() {

	if [ ${DEBUG} -gt 0 ]; then
		print_message "+++ Debug Info +++";
		print_message "\t in function check_available_Images";
	fi

	# get what's on the download server available for the found HW

	[ -e "${AVAILABLE_IMAGES}" ] && rm -f "${AVAILABLE_IMAGES}";

	curl -s "${DOWNLOAD_URL}" -o "${AVAILABLE_IMAGES}";

	RC=$?;   # return code from curl

	[ "${RC}" -gt "0" ] && error_warning "something went wrong during download from the Server: \n\n\t ${DOWNLOAD_URL} \n\n\t curl return code was: ${RC} \n\n\t Aborting !!!";

	if [ ${DEBUG} -gt 0 ]; then
		print_message "\t curl return code after downloading server list to ${AVAILABLE_IMAGES} was:\t ${RC} \n";
	fi

	# -s FILE: FILE exists and has a size greater than zero
	if [ ! -s "${AVAILABLE_IMAGES}" ]; then
		error_warning "Can't find anything on the download server: ${AVAILABLE_IMAGES} was empty ! \n\n\t Aborting !!!";
	else
		sed -i -e "/${PLATFORM}/!d" \
			-e '/^$/d' \
			-e '/.sha256/d' \
			-e '/\[/d' \
			-e '/\]/d' \
			-e 's/{//g' \
			-e 's/}//g' \
			-e 's/,//g' \
			-e 's/"name"://g' \
			-e 's/"type":"file"//g' \
			-e 's/"mtime"://g' \
			-e 's/"size"://g' \
			-e 's/^ //g' \
			-e 's/ $//g' "${AVAILABLE_IMAGES}";

		[ -e "${POTENTIAL_IMAGES}" ] && rm -rf "${POTENTIAL_IMAGES}";

		while read -r LINE; do
			# prepare for sortable dates the form YYMMDDHHMM
			DAY="$(echo "${LINE}" | awk -F " " '{print $3}' | sed 's/ //g')";
			MONTH_STRING="$(echo "${LINE}" | awk -F " " '{print $4}' | sed 's/ //g')";
			YEAR="$(echo "${LINE}" | awk -F " " '{print $5}' | sed 's/ //g')";
			TIME="$(echo "${LINE}" | awk -F " " '{print $6}' | cut -d: -f1,2 | sed 's/ //g' | sed 's/://g')";
			
			case ${MONTH_STRING} in

			Jan)
				MONTH=01;
				;;

			Feb)
				MONTH=02;
				;;

			Mar)
				MONTH=03;
				;;

			Apr)
				MONTH=04;
				;;

			May)
				MONTH=05;
				;;

			Jun)
				MONTH=06;
				;;

			Jul)
				MONTH=07;
				;;

			Aug)
				MONTH=08;
				;;

			Sep)
				MONTH=09;
				;;

			Oct)
				MONTH=10;
				;;

			Nov)
				MONTH=11;
				;;

			Dec)
				MONTH=12;
				;;

			esac

			MY_DATE="${YEAR}${MONTH}${DAY}${TIME}";

			echo "${MY_DATE}   ${LINE}" >> "${POTENTIAL_IMAGES}";

		done < "${AVAILABLE_IMAGES}";

		###############################################################
		#
		# possible cases in ${POTENTIAL_IMAGES}
		#
		# case 1.  nothing found cause ${POTENTIAL_IMAGES} is empty
		# case 2.  ${POTENTIAL_IMAGES} doesn't contain the currently running nightly => NO decision/comparsion possible
		# Case 3.  only one entry:  => is it a update or a downgrade  
		# Case 4.  more then 2 entries: => which one is the update or downgrade 
		#           
		#####################################################################################

		###
		### case 1.: ${POTENTIAL_IMAGES} is empty => nothing to download
		###

		# -s FILE: FILE exists and has a size greater than zero
		[ ! -s "${POTENTIAL_IMAGES}" ] && error_warning "NO nightlies found today ! \n\n\t Please try later again";

		# sort list downward to get the newest on top
		sort -ur "${POTENTIAL_IMAGES}" -o "${POTENTIAL_IMAGES}";

		#####################################################################################
		###
		### case 2.: if I can't find the currently running nightly in the list I got from the download server I can't make any decision
		###

		DATE_CURRENTLY_RUNNING="$(cat "${POTENTIAL_IMAGES}" | grep "${RUNNING_VERSION}" | head -1 | awk -F " " '{print $1}' | sed 's/ //g')";

		# -z STRING: the length of STRING is zero
		[ -z "${DATE_CURRENTLY_RUNNING}" ] && error_warning "I can't find your currently running nighty on the dowload server ! \n\n\t Aborting !!!";

		#####################################################################################
		###
		### case 3.: only one entry in ${POTENTIAL_IMAGES}
		###

		COUNTER="$(cat "${POTENTIAL_IMAGES}" | wc -l)";

		if [ "${COUNTER}" -eq 1 ]; then
			DATE_SINGLE_ENTRY="$(cat "${POTENTIAL_IMAGES}" | head -1 | awk -F " " '{print $1}' | sed 's/ //g')";

			if [ "${DATE_SINGLE_ENTRY}" -gt "${DATE_CURRENTLY_RUNNING}" ]; then
				UPDATE_IMAGE="$(cat "${POTENTIAL_IMAGES}" | grep "${DATE_SINGLE_ENTRY}" | head -1 | awk -F " " '{print $2}' | sed -e 's/"//g' -e 's/ //g')";
				UPDATE_VERSION="$(cat "${POTENTIAL_IMAGES}" | grep "${DATE_SINGLE_ENTRY}" | head -1 | awk -F " " '{print $2}' | sed -e 's/"//g' -e 's/.img.gz//g'| awk -F 'nightly-' '{print $2}')";
				DOWNGRADE_IMAGE="";
			else
				UPDATE_IMAGE="";
				DOWNGRADE_IMAGE="$(cat "${POTENTIAL_IMAGES}" | grep "${DATE_SINGLE_ENTRY}" | head -1 | awk -F " " '{print $2}' | sed -e 's/"//g' -e 's/ //g')";
				DOWNGRADE_VERSION="$(cat "${POTENTIAL_IMAGES}" | grep "${DATE_SINGLE_ENTRY}" | head -1 | awk -F " " '{print $2}' | sed -e 's/"//g' -e 's/.img.gz//g'| awk -F 'nightly-' '{print $2}')";
			fi
		fi

		#####################################################################################
		###
		### case 4.: more then 2 entries in ${POTENTIAL_IMAGES} => which one is the update or downgrade
		###

		# and cause of sorting with "-ur" the newest is at top
		sort -ur "${POTENTIAL_IMAGES}" -o "${POTENTIAL_IMAGES}";

		# remember top entry => possible update candidate
		DATE_UPDATE_CANDIDATE="$(cat "${POTENTIAL_IMAGES}" | head -1 | awk -F " " '{print $1}' | sed 's/ //g')";

		### corner case: NO update image in the list: the currently running is the same as the newest in the list

		if [ "${DATE_UPDATE_CANDIDATE}" -gt "${DATE_CURRENTLY_RUNNING}" ]; then
			UPDATE_IMAGE="$(cat "${POTENTIAL_IMAGES}" | grep "${DATE_UPDATE_CANDIDATE}" | head -1 | awk -F " " '{print $2}' | sed -e 's/"//g' -e 's/ //g')";
			UPDATE_VERSION="$(cat "${POTENTIAL_IMAGES}" | grep "${DATE_UPDATE_CANDIDATE}" | head -1 | awk -F " " '{print $2}' | sed -e 's/"//g' -e 's/.img.gz//g'| awk -F 'nightly-' '{print $2}')";
		else
			UPDATE_IMAGE="";	# needed, otherwise UPDATE_* is undefined
			UPDATE_VERSION="";	# needed, otherwise UPDATE_* is undefined
		fi

		# sort ascending to get the oldest at top and search the list until currently running => possible downgrade candidate
		sort -ui "${POTENTIAL_IMAGES}" -o "${POTENTIAL_IMAGES}";

		while read -r LINE; do

			[ "${LINE}" = " " ] && break;

			eval echo "${LINE}" | grep -q "${RUNNING_VERSION}"  && break;

			DATE_DOWNGRADE_CANDIDATE="$(echo "${LINE}" | awk -F " " '{print $1}' | sed 's/ //g')";

		done < "${POTENTIAL_IMAGES}";

		### corner case: NO downgrade image in the list: the currently running is the same as the oldest in the list
      
		# -n STRING: the length of STRING is nonzero
		if [ -n "${DATE_DOWNGRADE_CANDIDATE}" ]; then

			if [ "${DATE_DOWNGRADE_CANDIDATE}" -lt "${DATE_CURRENTLY_RUNNING}" ]; then
				DOWNGRADE_IMAGE="$(cat "${POTENTIAL_IMAGES}" | grep "${DATE_DOWNGRADE_CANDIDATE}" | head -1 | awk -F " " '{print $2}' | sed -e 's/"//g' -e 's/ //g')";
				DOWNGRADE_VERSION="$(cat "${POTENTIAL_IMAGES}" | grep "${DATE_DOWNGRADE_CANDIDATE}" | head -1 | awk -F " " '{print $2}' | sed -e 's/"//g' -e 's/.img.gz//g'| awk -F 'nightly-' '{print $2}')";
			fi

		else
			DOWNGRADE_IMAGE="";		# needed, otherwise DOWNGRADE_* is undefined
			DOWNGRADE_VERSION="";	# needed, otherwise DOWNGRADE_* is undefined
		fi

		if [ ${DEBUG} -gt 0 ]; then

			[ "${COUNTER}" ] &&                   print_message "\t the list contains     \t ${COUNTER} entries";

			[ "${DATE_SINGLE_ENTRY}" ] &&         print_message "\t single Entry in list: \t ${DATE_SINGLE_ENTRY}";

			[ "${DATE_UPDATE_CANDIDATE}" ] &&     print_message "\n\t Update candidate:   \t ${DATE_UPDATE_CANDIDATE}";
			[ "${UPDATE_VERSION}" ] &&            print_message "\t Update Version:       \t ${UPDATE_VERSION}";
			[ "${UPDATE_IMAGE}" ] &&              print_message "\t Update image:         \t ${UPDATE_IMAGE}";

			[ "${DATE_DOWNGRADE_CANDIDATE}" ] &&  print_message "\n\t downgrade candiate: \t ${DATE_DOWNGRADE_CANDIDATE}";
			[ "${DOWNGRADE_VERSION}" ] &&         print_message "\t Downgrade_Version:    \t ${DOWNGRADE_VERSION}";
			[ "${DOWNGRADE_IMAGE}" ] &&           print_message "\t Downgrade image:      \t ${DOWNGRADE_IMAGE}";

			print_message "\n\n";
		fi
	fi
}
########################################################################

check_disk_space() {

	if [ ${DEBUG} -gt 0 ]; then
		print_message "+++ Debug Info +++";
		print_message "\t in function check_disk_space";
	fi

	# all Images (Generic*, RPi*) decompresses to the same size: 549 MB

	SPACE_NEEDED="562176";  # 549 MB

	SPACE_AVAILABLE=$(df | grep ${PARTITION} | awk 'END {print $4}');

	[ "${SPACE_AVAILABLE}" -lt "${SPACE_NEEDED}" ] && error_warning "Sorry, NOT enough disk space to proceed ! \n\n\t The update installer needs roughly $((SPACE_NEEDED/1024)) MB of free disk space !\n\n\t Aborting !!!";

	if [ ${DEBUG} -gt 0 ]; then
		print_message "+++ Debug Info +++";
		print_message "\t needed disk space is:      ${SPACE_NEEDED} MB ";
		print_message "\t avaiable disk space is:    ${SPACE_AVAILABLE} MB ";
	fi

	return 0;
}
########################################################################

download_and_install() {

	if [ ${DEBUG} -gt 0 ]; then
		print_message "+++ Debug Info +++";
		print_message "\t in Function  download_and_install";
	fi

	IMAGE_TO_INSTALL=$1;

	[ ${DEBUG} -gt 0 ] && print_message "\t for download selected image is: \t ${IMAGE_TO_INSTALL}";

	# images are downloaded to ~/.update

	# -d FILE: FILE exists and is a directory
	[ ! -d "${UPDATE_DIR}" ] && mkdir -p ${UPDATE_DIR};

	# don't overread:  cd ... is TRUE when cd is successfully executed and only THEN "rm -f ./* " takes action !
	[ -d ${UPDATE_DIR} ] && cd ${UPDATE_DIR} && rm -f ./*;

	if [ ${DEBUG} -eq 0 ]; then
		print_message "\n\t downloading: \t ${IMAGE_TO_INSTALL} \n\n\t wait ...\n";			# and \n\t downloading: \t ${IMAGE_TO_INSTALL}.sha256 \n\n\t wait ...\n";

		curl "${DOWNLOAD_URL}/${IMAGE_TO_INSTALL}" -o "${IMAGE_TO_INSTALL}";				# && curl -# "${DOWNLOAD_URL}/${IMAGE_TO_INSTALL}.sha256" -o ${IMAGE_TO_INSTALL}.sha256;

		RC=$?;   # see: man curl

		[ "${RC}" -gt "0" ] && error_warning "something went wrong during image download ! \n curl return code is:\t ${RC} \n\n\t Aborting !!!";

		########################################################################
		#
		#	switch checksumming OFF for now, cause of damaged *.sha256 on the server
		#
		#      print_message "\n\t checksumming the downloaded image, wait...\n";
		#      sha256sum -c ${IMAGE_TO_INSTALL}.sha256 > /dev/null;
		#      RC=$?;
		#      [ "${RC}" -gt "0" ] && error_warning "Error during checksumming. The return code is: \t ${RC}  \n\n\t Aborting !!!";
		########################################################################

		# cause we downloaded to ~/.update remove the *.sha256 to not confuse the update installer during next boot
		[ -e "${IMAGE_TO_INSTALL}.sha256" ] && rm -f "${IMAGE_TO_INSTALL}.sha256";

	else

		print_message "+++ Debug Info +++";
		print_message "\t while in DEBUG mode nothing will be downloaded, but the command would be:";
		print_message "\t curl ${DOWNLOAD_URL}/${IMAGE_TO_INSTALL} -o ${IMAGE_TO_INSTALL}";  # && curl -# "${DOWNLOAD_URL}/${IMAGE_TO_INSTALL}.sha256" -o "${IMAGE_TO_INSTALL}.sha256" ";
	fi

	# check for zero length/empty nightlies !
	# -s FILE: FILE exists and has a size greater than zero
	if [ -s "${IMAGE_TO_INSTALL}" ]; then
		print_message "\t nightly image under: \t ${UPDATE_DIR} is ready ! \n\n\t reboot to install the nightly or maybe reboot via following command: \n\n\t sync && sync && systemctl reboot \n\n";
	else
		[ ${DEBUG} -eq 0 ] && error_warning "downloaded image seems broken or has zero length, please try later again And/Or check the website: \n\n\t ${DOWNLOAD_URL}";
	fi

	[ -d "${MY_TMPDIR}" ] && rm -rf "${MY_TMPDIR}";
}
########################################################################
#
#		MAIN
#
########################################################################

trap "error_warning 'script was aborted !'" INT TERM  # SIGINT SIGTERM

[ ${DEBUG} -eq 0 ] && clear;

get_HW_info;

check_HW_Support;

create_ServerURL;

check_available_Images;

print_message " *** script version: ${SCRIPT_VERSION}\n";
print_message "\t your Platform is:       ${PLATFORM}";
print_message "\t running nightly from:   ${RUNNING_VERSION_DATE}";
print_message "\t and has Git Hash:       ${GIT_HASH_INSTALLED}";
print_message "\t Version:                ${RUNNING_VERSION}";

print_message "\n *** the following images are currently available on the download server:";

if [ "${UPDATE_IMAGE}" ]; then
	print_message "\t * Update (to the most recent Image): \t ${UPDATE_IMAGE} ";
else
	print_message "\t *** No updates today *** ";
fi

if [ "${DOWNGRADE_IMAGE}" ]; then
	print_message "\t * Downgrade (to the previous Image):  \t ${DOWNGRADE_IMAGE} ";
else
	print_message "\t *** No downgrade today *** ";
fi

while true; do

	REPLY="";

	print_message "\n *** What do you want me to do ? ";

	print_message "\t u) \t update to the lastest nightly";
	print_message "\t d) \t downgrade to the previous nightly";
	print_message "\t q) \t quit (auto. selected in 20 seconds  !!!)";
	print_message;

	read -r -n 1 -t 20 -p "Enter selection [u, d, q] >  ";

	case ${REPLY} in

	u)
		[ ${DEBUG} -gt 0 ] && print_message "\t update was selected";

		# -n STRING: the length of STRING is nonzero
		if [ -n "${UPDATE_IMAGE}" ]; then
			check_disk_space && download_and_install "${UPDATE_IMAGE}";
		else
			error_warning "No Image for updates available \n\n\t Aborting !!!";
		fi;
		break;
		;;

	d)
		[ ${DEBUG} -gt 0 ] && print_message "\t downgrade was selected";

		# -n STRING: the length of STRING is nonzero
		if [ -n "${DOWNGRADE_IMAGE}" ]; then
			check_disk_space && download_and_install "${DOWNGRADE_IMAGE}";
		else
			error_warning "No Image for downgrades available \n\n\t Aborting !!!";
		fi;
		break;
		;;

	q | *)
		[ ${DEBUG} -gt 0 ] && print_message "quit selected";
		print_message "\n\n\t *** Sayonara *** \n\n";
		clean_up && exit 0;
		break;
		;;
	esac

done

exit 0;
