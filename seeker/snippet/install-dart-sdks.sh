#date: 2022-07-27T17:03:26Z
#url: https://api.github.com/gists/833b22f6c0fa9776dae5cf9bcc2fb295
#owner: https://api.github.com/users/theanurin

#
# To start this script from remote:
#
#   curl https://gist.githubusercontent.com/theanurin/833b22f6c0fa9776dae5cf9bcc2fb295/raw/install-dart-sdks.sh | /bin/bash
#

mkdir --parents ~/opt/dart
cd ~/opt/dart/
for DART_VERSION in 2.17.6 2.16.2 2.15.1 2.10.5; do
	echo
	echo "Installing Dart v${DART_VERSION} ..."
	EXPECTED_SHA256SUM=$(curl --fail --silent https://storage.googleapis.com/dart-archive/channels/stable/release/${DART_VERSION}/sdk/dartsdk-linux-x64-release.zip.sha256sum | cut -f 1 -d ' ')
	curl --fail --output /dev/shm/${DART_VERSION}-dartsdk-linux-x64-release.zip https://storage.googleapis.com/dart-archive/channels/stable/release/${DART_VERSION}/sdk/dartsdk-linux-x64-release.zip
	CURRENT_SHA256SUM=$(sha256sum /dev/shm/${DART_VERSION}-dartsdk-linux-x64-release.zip | cut -f 1 -d ' ')
	if [ "${EXPECTED_SHA256SUM}" != "${CURRENT_SHA256SUM}" ]; then
		echo "Expected hashsum ${EXPECTED_SHA256SUM} is not equal to actual hashsum ${CURRENT_SHA256SUM}" >&2
		break
	fi
	unzip -q /dev/shm/${DART_VERSION}-dartsdk-linux-x64-release.zip
	rm /dev/shm/${DART_VERSION}-dartsdk-linux-x64-release.zip
	mv dart-sdk ${DART_VERSION}-x64
done