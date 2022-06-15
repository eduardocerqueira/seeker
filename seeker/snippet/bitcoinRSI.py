#date: 2022-06-15T16:50:03Z
#url: https://api.github.com/gists/91bc598c8d424bd1786073be449dfbad
#owner: https://api.github.com/users/PayayamDev

from cryptopher.collect.get_binance_data import GetBinanceData
import btalib as btalib

# กำหนด timeframe
tf = '1h'
start_date = '5 day ago UTC'
gbd = GetBinanceData(tf, start_date)

# กำหนด เหรียญที่ต้องการดึงราคา
target_name = 'BTCUSDT'
df1 = gbd.get_data(target_name)

df1.columns = ['Date', 'Open', 'High', 'Low', 'Close','Volume']
df1 = df1.set_index('Date')

# คำนวณค่า RSI
rsi = btalib.rsi(df1['Close'])
print(rsi.df)


# ดึงค่า rsi 2 ค่าล่าสุดออกมาตาวจสอบ ไม่นับค่าสุดท้ายเพราะเป็นค่าจากเวลาปัจจุบัน ซึ่งค่าไม่นิ่งเพราะยังไม่แท่งเทียนยังไม่ปิด
# ดึงค่า rsi นับจากเวลาล่าสุดไป 2
rsi_before = rsi.df[117:118]['rsi'][0]
print(rsi_before)

# ดึงค่า rsi นับจากเวลาล่าสุดไป 1
rsi_after = rsi.df[118:119]['rsi'][0]
print(rsi_after)

# ตรวจสอบเงื่อนไข หาสัญญาณซื้อ - ขาย 
# (เอาอย่างง่ายก่อน ถ้าจะเอามาใช้จริง ต้องใช้หลาย indicator และหายปัจจัย)

if rsi_after <= 40 and rsi_after > rsi_before:
    if rsi_after > 30 and rsi_before <= 30:
        print('สัญญาณ ซื้อ สวยๆ RSI ตัดเส้น 30 ขึ้น')
    else:
        print('สัญญาณเตรียม ซื้อ')
elif rsi_after >= 60 and rsi_after < rsi_before:
    if rsi_after < 70 and rsi_before >= 70:
        print('สัญญาณ ขาย สวยๆ RSI ตัดเส้น 70 ลง')
    else:
        print('สัญญาณเตรียม ขาย')
else:
    print('ยังไม่มีสัญญาณ ซื้อ-ขาย จาก RSI')