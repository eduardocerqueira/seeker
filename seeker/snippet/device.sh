#date: 2023-01-05T16:45:57Z
#url: https://api.github.com/gists/8e2d4ed711b859e52b2fbc111fc09801
#owner: https://api.github.com/users/Amanse

git clone https://github.com/pjgowtham/android_device_oplus_RMX3461 device/oplus/RMX3461 --depth=1
git clone https://github.com/pjgowtham/android_device_oplus_sm8350-common device/oplus/sm8350-common -b bk
git clone https://github.com/pjgowtham/android_kernel_oplus_sm8350 kernel/oplus/sm8350 -b RMX3461 --depth=1
git clone https://github.com/pjgowtham/android_hardware_oplus hardware/oplus --depth=1
git clone https://gitlab.com/itsxrp/proprietary_vendor_oplus vendor/oplus --depth=1 -b elixir
git clone https://gitlab.com/itsxrp/vendor_oplus_rmx3461 vendor/oplus/RMX3461 --depth=1

cd device/oplus/sm8350-common
git reset a9fad439a433c05becd3bfa643e120a606ac1cf7 --hard
cd -

