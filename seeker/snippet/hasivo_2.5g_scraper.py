#date: 2025-11-21T17:09:43Z
#url: https://api.github.com/gists/dfd3cd3015373e5792fdb2fd34b2a692
#owner: https://api.github.com/users/patrick3399

#!/usr/bin/env python3
"""
Hasivo Switch Scraper for Home Assistant
放置於: /config/scripts/hasivo_scraper.py
記得設定執行權限: chmod +x /config/scripts/hasivo_scraper.py
"""
import requests
import hashlib
import json
import sys
from datetime import datetime
from bs4 import BeautifulSoup

# 配置
BASE_URL = "http://192.168.10.8"
USERNAME = "admin"
PASSWORD = "**********"
TIMEOUT = 10

def md5_hash(text):
    """計算 MD5 雜湊值"""
    return hashlib.md5(text.encode()).hexdigest()

 "**********"d "**********"e "**********"f "**********"  "**********"l "**********"o "**********"g "**********"i "**********"n "**********"_ "**********"a "**********"n "**********"d "**********"_ "**********"f "**********"e "**********"t "**********"c "**********"h "**********"( "**********"u "**********"r "**********"l "**********", "**********"  "**********"u "**********"s "**********"e "**********"r "**********"n "**********"a "**********"m "**********"e "**********", "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********") "**********": "**********"
    """登入並獲取頁面內容"""
    try:
        session = requests.Session()
        
        # 計算 MD5 認證
        combined = "**********"
        response_hash = md5_hash(combined)
        
        # 登入資料
        login_data = {
            'username': username,
            'password': "**********"
            'Response': response_hash,
            'language': 'CN'
        }
        
        # 設置 Cookie
        session.cookies.set('admin', response_hash)
        
        # 登入
        login_url = f"{BASE_URL}/login.cgi"
        session.post(login_url, data=login_data, timeout=TIMEOUT)
        
        # 獲取目標頁面
        response = session.get(url, timeout=TIMEOUT)
        response.encoding = 'utf-8'
        
        return response.text
    
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return None

def parse_port_page(html):
    """解析 port.cgi 頁面"""
    soup = BeautifulSoup(html, 'html.parser')
    
    # 用於統計整體連線狀態
    connected_count = 0
    
    ports_data = {}
    
    try:
        # 找到狀態表格(最後一個 table)
        tables = soup.find_all('table', border="1")
        if not tables:
            raise Exception("找不到端口狀態表格")
        
        status_table = tables[-1]
        rows = status_table.find_all('tr')
        
        # 跳過前兩行標題,從第三行開始是實際數據
        for row in rows[2:]:
            cells = row.find_all('td')
            if len(cells) >= 6:
                # 解析端口資訊
                port_name = cells[0].text.strip()
                port_num = port_name.replace('端口', '').strip()
                
                state = cells[1].text.strip()
                actual_speed = cells[3].text.strip()
                
                # 判斷連線狀態(只用來決定整體 Connected/Disconnected)
                if actual_speed != "掉线" and actual_speed != "Down":
                    connected_count += 1
                    link_status = "up"
                else:
                    link_status = "down"
                
                # 直接儲存原始數據,不做轉換
                ports_data[f"port{port_num}"] = {
                    "state": state,
                    "actual_speed": actual_speed,
                    "link_status": link_status
                }
    
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        return {
            "status": "Error",
            "ports": {}
        }
    
    # 決定整體連線狀態
    if connected_count > 0:
        status = "Connected"
    else:
        status = "Disconnected"
    
    return {
        "status": status,
        "ports": ports_data
    }

def main():
    """主函數"""
    # 獲取頁面內容
    html = "**********"
    
    if not html:
        # 返回錯誤但保持 JSON 格式
        error_data = {
            "status": "Error",
            "ports": {}
        }
        print(json.dumps(error_data))
        sys.exit(0)
    
    # 解析並輸出 JSON
    data = parse_port_page(html)
    print(json.dumps(data, ensure_ascii=False))

if __name__ == "__main__":
    main()