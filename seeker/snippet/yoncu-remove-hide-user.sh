#date: 2022-04-12T17:16:54Z
#url: https://api.github.com/gists/ec1eba7219babf52177c9bb5a5dc2eca
#owner: https://api.github.com/users/ertugrulturan

#zarari kullaniciyi ekrana yazdir
echo "zarari kullanici(varsa yazar yoksa bostur):"
cat /etc/passwd | grep psalics
sleep 3
#temiz kurulum plesk root cron unda tek olan veriyi tum icerigi silerek guncelliyoruz
echo '47	23	*	*	*	/usr/sbin/ntpdate -b -s 0.pool.ntp.org' > /var/spool/cron/root
#zararli kullaniciyi sistem uzerinden siliyoruz
userdel -r psalics
#gozden kacmis gecmiste olmus olabilicek her turlu isleme karsilik baglatıyı engelliyoruz
echo '127.0.0.1 plesktrial.yoncu.com' >> /etc/hosts
echo '127.0.0.1 yoncu.com' >> /etc/hosts
clear
echo "Islem Tamamlandi! -Layer.web.tr- Zehri bilmeyen panzehir uretemez."