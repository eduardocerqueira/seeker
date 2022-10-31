#date: 2022-10-31T17:20:50Z
#url: https://api.github.com/gists/aae755c01daf9501832116390ee810f3
#owner: https://api.github.com/users/badicev

my_list = ["Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül"]
f = open("aylar2.txt", "w")
for name in my_list:
    f.write(name + "\n")
f.close()