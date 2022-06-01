#date: 2022-06-01T17:22:44Z
#url: https://api.github.com/gists/f3a623e7e43847c6a0d881390658c43e
#owner: https://api.github.com/users/jthoms1

(export IONIC_CLOUD_VERSION=0.5.0; curl -sL https://ionic.io/get-ionic-cloud-cli | bash)

ionic-cloud live-update download --app-id 186b544f --channel-name production --zip-name myappname.zip

unzip myappname.zip -d ./myappname