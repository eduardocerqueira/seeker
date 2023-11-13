#date: 2023-11-13T17:04:56Z
#url: https://api.github.com/gists/0c2d48da834f1e24225a0d3240bd2570
#owner: https://api.github.com/users/kyY00n

#!/bin/bash
# Amazon Linux 2023에서 기본 MySQL 서버 설치 및 설정

# 시스템 업데이트
sudo yum update -y

# MySQL 서버 설치
sudo yum install mysql-server -y

# MySQL 서비스 시작 및 부팅 시 자동 시작 설정
sudo systemctl start mysqld
sudo systemctl enable mysqld
