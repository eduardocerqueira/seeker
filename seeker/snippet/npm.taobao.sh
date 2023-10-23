#date: 2023-10-23T17:06:15Z
#url: https://api.github.com/gists/b492e95632880688cdab48a2d38659a5
#owner: https://api.github.com/users/neocodegs

# ==========================================================
# NPM
# ==========================================================

npm set registry https://registry.npmmirror.com # 注册模块镜像
npm set disturl https://npmmirror.com/mirrors/node # node-gyp 编译依赖的 node 源码镜像

## 以下选择添加
npm set sass_binary_site https://registry.npmmirror.com/mirrors/node-sass # node-sass 二进制包镜像
npm set electron_mirror https://registry.npmmirror.com/mirrors/electron/ # electron 二进制包镜像
npm set puppeteer_download_host https://registry.npmmirror.com/mirrors # puppeteer 二进制包镜像
npm set chromedriver_cdnurl https://registry.npmmirror.com/mirrors/chromedriver # chromedriver 二进制包镜像
npm set operadriver_cdnurl https://registry.npmmirror.com/mirrors/operadriver # operadriver 二进制包镜像
npm set phantomjs_cdnurl https://registry.npmmirror.com/mirrors/phantomjs # phantomjs 二进制包镜像
npm set selenium_cdnurl https://registry.npmmirror.com/mirrors/selenium # selenium 二进制包镜像
npm set node_inspector_cdnurl https://registry.npmmirror.com/mirrors/node-inspector # node-inspector 二进制包镜像
npm set sentrycli_cdnurl https://registry.npmmirror.com/mirrors/sentry-cli # sentry-cli

npm cache clean --force # 清空缓存

# ==========================================================
# YARN
# ==========================================================

yarn config set registry https://registry.npmmirror.com # 注册模块镜像
yarn config set disturl https://npmmirror.com/mirrors/node # node-gyp 编译依赖的 node 源码镜像

## 以下选择添加
yarn config set sass_binary_site https://registry.npmmirror.com/mirrors/node-sass # node-sass 二进制包镜像
yarn config set electron_mirror https://registry.npmmirror.com/mirrors/electron/ # electron 二进制包镜像
yarn config set puppeteer_download_host https://registry.npmmirror.com/mirrors # puppeteer 二进制包镜像
yarn config set chromedriver_cdnurl https://registry.npmmirror.com/mirrors/chromedriver # chromedriver 二进制包镜像
yarn config set operadriver_cdnurl https://registry.npmmirror.com/mirrors/operadriver # operadriver 二进制包镜像
yarn config set phantomjs_cdnurl https://registry.npmmirror.com/mirrors/phantomjs # phantomjs 二进制包镜像
yarn config set selenium_cdnurl https://registry.npmmirror.com/mirrors/selenium # selenium 二进制包镜像
yarn config set node_inspector_cdnurl https://registry.npmmirror.com/mirrors/node-inspector # node-inspector 二进制包镜像
yarn config set sentrycli_cdnurl https://registry.npmmirror.com/mirrors/sentry-cli # sentry-cli

yarn cache clean # 清空缓存