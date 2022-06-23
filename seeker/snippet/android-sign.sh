#date: 2022-06-23T17:02:31Z
#url: https://api.github.com/gists/47468a35cded1a5631ec6785a8d9d77d
#owner: https://api.github.com/users/Marcotso

#!/bin/bash

# 私钥别名
ALIAS=alias_name

# 私钥存储密码
STOREPASS=password

# 要签名的 APK 文件位置
APK_FILE=my_application.apk

# 私钥存储的路径，注意私钥不能丢，否则以后无法更新
PRIVATE_KEY_FILE=my-release-key.keystore

# 签名完成后的文件存储路径
OUTPUT_FILE=your_project_name.apk

# 生成私钥
keytool \
  -genkey \
  -keystore $PRIVATE_KEY_FILE \
  -alias $ALIAS \
  -keyalg RSA \
  -keysize 2048 \
  -storepass $STOREPASS \
  -validity 10000

# 签名，使用 tsa 保证时间
jarsigner \
	-tsa http://timestamp.digicert.com \
  -sigalg SHA1withRSA \
  -digestalg SHA1 \
  -keystore $PRIVATE_KEY_PATH \
  -storepass $STOREPASS \
  $APK_FILE \
  $ALIAS

# 对齐，减小内存使用，必须在签名后完成
# 使用 google 提供的高压缩比 zlib
zipalign 4 \
  -z \
  $APK_FILE \
  $OUTPUT_FILE
