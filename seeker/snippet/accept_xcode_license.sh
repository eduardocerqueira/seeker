#date: 2022-07-14T17:03:37Z
#url: https://api.github.com/gists/84ff5d36b01905d8d41ec3b8ef7185a9
#owner: https://api.github.com/users/smithw-adsk

# Run using "sudo accept_xcode_license.sh"
#
# Solving the OSX Yosemite Xcode Command Line Tools Licensing problem
# for multiple updates in order to script post-install tasks. 
# Typical error reads after running "xcode-select --install" when setting up 
# Homebrew is: "Agreeing to the Xcode/iOS license requires admin priviledges, 
# please re-run as root via sudo"
#
# CREDIT:
# Based on a tip found at http://krypted.com/mac-os-x/licensing-the-xcode-command-line-tools/
# Also using the code found here: http://stackoverflow.com/questions/26125036

!/usr/bin/expect
set timeout 5
spawn sudo xcodebuild -license
expect {
    "By typing 'agree' you are agreeing" {
        send "agree\r\n"
    }
    "Software License Agreements Press 'space' for more, or 'q' to quit" {
        send " ";
        exp_continue;
    }
    timeout {
        send_user "\nTimeout 2\n";
        exit 1
    }
}
expect {
    timeout {
        send_user "\nFailed\n";
        exit 1
    }
}