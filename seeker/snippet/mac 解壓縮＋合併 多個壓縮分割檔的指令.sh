#date: 2025-04-01T17:04:27Z
#url: https://api.github.com/gists/6e3fa5dcfdfa527dba0529171401fbc0
#owner: https://api.github.com/users/lazyjerry

# 首先在同一個目錄建立 combined 資料夾
# 儲存該 shell 編輯更改名稱，執行。
for archive in 分割檔案的檔案名稱-*.zip
do
    ditto -V -x -k --sequesterRsrc "$archive" combined
done