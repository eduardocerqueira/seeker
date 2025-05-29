#date: 2025-05-29T17:14:21Z
#url: https://api.github.com/gists/d704bbbb4c6e26c938e9e90e797b2616
#owner: https://api.github.com/users/Hossein-aliian

import pandas as pd
import numpy as np
from playwright.sync_api import sync_playwright, TimeoutError
from datetime import datetime
import re
import os
import time
import subprocess
##############################################
# تنظیمات و توابع کمکی
##############################################

input_file = "symbol_links.csv"           # فایل ورودی لینک‌های نماد
output_file = "processed_symbols.csv"      # فایل خروجی اولیه (قبل از استانداردسازی)
standardized_output = "processed_symbols_standardized.csv"  # فایل خروجی نهایی استاندارد شده

# خواندن لینک‌ها (می‌توانید تعداد لینک‌ها را برای دیباگ محدود کنید)
df_symbols = pd.read_csv(input_file)
urls = df_symbols["لینک"]
# urls = df_symbols["لینک"].head(10)

def clean_symbol_name(raw_name):
    """حذف دو کلمه اول و نگه داشتن کلمه‌ی سوم"""
    words = raw_name.split()
    return words[2] if len(words) >= 3 else "نامعتبر"

def update_csv_with_record(record, csv_file):
    """
    اگر فایل CSV موجود باشد، رکورد جدید (بر اساس 'نام نماد' و 'تاریخ استخراج')
    رکوردهای قبلی را جایگزین می‌کند؛ در غیر این صورت، رکورد جدید را اضافه می‌کند.
    همچنین، پس از به‌روزرسانی، داده‌ها را بر اساس 'نام نماد' و 'تاریخ استخراج'
    (به صورت طولی و مرتب شده) مرتب می‌کند.
    """
    new_df = pd.DataFrame([record])
    # اگر فایل موجود باشد
    if os.path.exists(csv_file):
        old_df = pd.read_csv(csv_file, encoding="utf-8")
        # حذف رکوردهایی که نام نماد و تاریخ استخراج مشابه دارند
        updated_df = old_df[~((old_df["نام نماد"] == record["نام نماد"]) &
                              (old_df["تاریخ استخراج"] == record["تاریخ استخراج"]))]
        updated_df = pd.concat([updated_df, new_df], ignore_index=True)
        # تبدیل تاریخ به نوع datetime و مرتب‌سازی بر اساس نام نماد و تاریخ استخراج
        updated_df["تاریخ استخراج"] = pd.to_datetime(updated_df["تاریخ استخراج"])
        updated_df = updated_df.sort_values(by=["نام نماد", "تاریخ استخراج"]).reset_index(drop=True)
        updated_df.to_csv(csv_file, index=False, encoding="utf-8")
    else:
        new_df.to_csv(csv_file, index=False, encoding="utf-8")

def load_page(page, url, fixed_timeout=3000):
    """
    این تابع سعی می‌کند تا صفحه را با زمان حداکثر 3 ثانیه بارگذاری کند.
    اگر عنصر مورد نظر (با سلکتور مشخص شده) قبل از پایان 3 ثانیه ظاهر شود،
    ادامه می‌دهد؛ در غیر این صورت، صفحه رفرش شده و تلاش مجدد می‌شود.
    """
    selector = "#main-info > div:nth-child(3) > div > table:nth-child(6) > tbody"
    attempt = 1
    while True:
        try:
            print(f"✅ تلاش {attempt}: بارگذاری {url} با timeout {fixed_timeout}ms")
            page.goto(url, timeout=fixed_timeout)
            page.wait_for_selector(selector, timeout=fixed_timeout)
            print(f"✅ صفحه {url} در تلاش {attempt} با موفقیت بارگذاری شد.")
            return True
        except TimeoutError as te:
            print(f"🚨 تلاش {attempt} برای {url} در مدت {fixed_timeout}ms موفق نشد: {te}. صفحه رفرش می‌شود...")
            try:
                page.reload(timeout=fixed_timeout)
            except TimeoutError:
                print(f"🚨 تلاش {attempt}: خطا هنگام رفرش صفحه {url}.")
            attempt += 1

def extract_symbol_data(page, url):
    """
    پس از بارگذاری موفق صفحه (با استفاده از load_page)، یک انتظار ثابت 1 ثانیه جهت تکمیل بارگذاری
    داینامیک اعمال می‌شود. سپس داده‌های صفحه استخراج شده و در یک دیکشنری ذخیره می‌شوند.
    """
    if not load_page(page, url):
        raise Exception("صفحه در مدت زمان معین بارگذاری نشد.")
    
    # انتظار ثابت 1 ثانیه جهت تکمیل بارگذاری اسکریپت‌های داینامیک
    page.wait_for_timeout(1000)
    
    name_element = page.query_selector(".xtitle-span")
    raw_name = name_element.inner_text() if name_element else "ناموجود"
    cleaned_name = clean_symbol_name(raw_name)
    
    avg_volume_element = page.query_selector("#tenCap_state")
    avg_volume = avg_volume_element.inner_text() if avg_volume_element else "ناموجود"
    
    extracted_data = {
        "نام نماد": cleaned_name,
        "میانگین ۱۰ روزه حجم معاملات": avg_volume,
        "تاریخ استخراج": datetime.today().strftime("%Y-%m-%d")
    }
    
    # استخراج داده‌های جدول اصلی
    tbody = page.query_selector("#main-info > div:nth-child(3) > div > table:nth-child(6) > tbody")
    if tbody:
        rows = tbody.query_selector_all("tr")
        for row in rows:
            cells = row.query_selector_all("td")
            if len(cells) >= 2:
                column_name = cells[0].inner_text().strip()
                value = cells[1].inner_text().strip()
                extracted_data[column_name] = value

    # استخراج بازدهی (درصدها)
    returns_wrapper = page.query_selector("#main-info > div:nth-child(3) > div > div.return-wrapper")
    returns_text = returns_wrapper.inner_text().replace("\n", " ").strip() if returns_wrapper else ""
    numbers = re.findall(r"[-+]?\d*\.?\d+%", returns_text)
    extracted_returns = {
         "بازده ۱ ماهه": numbers[0] if len(numbers) > 0 else "ناموجود",
         "بازده ۳ ماهه": numbers[1] if len(numbers) > 1 else "ناموجود",
         "بازده ۶ ماهه": numbers[2] if len(numbers) > 2 else "ناموجود",
         "بازده ۱ ساله": numbers[3] if len(numbers) > 3 else "ناموجود"
    }
    extracted_data.update(extracted_returns)
    
    # استخراج داده‌های مربوط به سرانه
    sarane_buy_element = page.query_selector("#sarane_buy")
    sarane_sell_element = page.query_selector("#sarane_sell")
    extracted_data["سرانه تقاضا"] = sarane_buy_element.inner_text().strip() if sarane_buy_element and sarane_buy_element.inner_text().strip() else "ناموجود"
    extracted_data["سرانه عرضه"] = sarane_sell_element.inner_text().strip() if sarane_sell_element and sarane_sell_element.inner_text().strip() else "ناموجود"
    
    n_number_element = page.query_selector("#n_number")
    extracted_data["تعداد معاملات روزانه"] = n_number_element.inner_text().strip() if n_number_element and n_number_element.inner_text().strip() else "ناموجود"
    
    volume_selector = "#main-info > div:nth-child(2) > div > table:nth-child(2) > tbody > tr:nth-child(2) > td:nth-child(2) > span"
    volume_element = page.query_selector(volume_selector)
    extracted_data["حجم معاملات روزانه"] = volume_element.inner_text().strip() if volume_element and volume_element.inner_text().strip() else "ناموجود"
    
    n_value_element = page.query_selector("#n_value")
    extracted_data["ارزش معاملات روزانه"] = n_value_element.inner_text().strip() if n_value_element and n_value_element.inner_text().strip() else "ناموجود"
    
    ten_client_buy_element = page.query_selector("#tenClientBuy_state")
    extracted_data["ميانگين 10 روزه سرانه خريد هر حقيقي"] = ten_client_buy_element.inner_text().strip() if ten_client_buy_element and ten_client_buy_element.inner_text().strip() else "ناموجود"
    
    n_buyeachi_element = page.query_selector("#n_buyeachi")
    extracted_data["خريد هر حقيقي (ميليون ريال)"] = n_buyeachi_element.inner_text().strip() if n_buyeachi_element and n_buyeachi_element.inner_text().strip() else "ناموجود"
    
    n_selleachi_element = page.query_selector("#n_selleachi")
    extracted_data["فروش هر حقيقي (ميليون ريال)"] = n_selleachi_element.inner_text().strip() if n_selleachi_element and n_selleachi_element.inner_text().strip() else "ناموجود"
    
    n_buyselleachi_element = page.query_selector("#n_buyselleachi")
    extracted_data["قدرت خريدار به فروشنده"] = n_buyselleachi_element.inner_text().strip() if n_buyselleachi_element and n_buyselleachi_element.inner_text().strip() else "ناموجود"
    
    n_inout_element = page.query_selector("#n_inout")
    extracted_data["ورود/خروج پول حقيقي (ميليون ريال)"] = n_inout_element.inner_text().strip() if n_inout_element and n_inout_element.inner_text().strip() else "ناموجود"
    
    client_real_buyer = page.query_selector("#n_clients2 > tr:nth-child(5) > td:nth-child(2)")
    extracted_data["تعداد خریدار حقیقی"] = client_real_buyer.inner_text().strip() if client_real_buyer and client_real_buyer.inner_text().strip() else "ناموجود"
    client_real_seller = page.query_selector("#n_clients2 > tr:nth-child(5) > td:nth-child(3)")
    extracted_data["تعداد فروشنده حقیقی"] = client_real_seller.inner_text().strip() if client_real_seller and client_real_seller.inner_text().strip() else "ناموجود"
    
    real_buy_vol_elem = page.query_selector("#n_clients2 > tr:nth-child(2) > td:nth-child(2) > div > div.data.d-flex.justify-content-between.align-items-center > span")
    if real_buy_vol_elem and real_buy_vol_elem.inner_text().strip():
        real_buy_vol_text = real_buy_vol_elem.inner_text().strip()
        real_buy_vol_clean = re.sub(r"\(.*?\)", "", real_buy_vol_text).strip()
    else:
        real_buy_vol_clean = "ناموجود"
    extracted_data["حجم خرید حقیقی ها"] = real_buy_vol_clean

    real_sell_vol_elem = page.query_selector("#n_clients2 > tr:nth-child(2) > td:nth-child(3) > div > div.data.d-flex.justify-content-between.align-items-center > span")
    if real_sell_vol_elem and real_sell_vol_elem.inner_text().strip():
        real_sell_vol_text = real_sell_vol_elem.inner_text().strip()
        real_sell_vol_clean = re.sub(r"\(.*?\)", "", real_sell_vol_text).strip()
    else:
        real_sell_vol_clean = "ناموجود"
    extracted_data["حجم فروش حقیقی ها"] = real_sell_vol_clean
    
    client_legal_buyer = page.query_selector("#n_clients2 > tr:nth-child(6) > td:nth-child(2)")
    extracted_data["تعداد خریدار حقوقی"] = client_legal_buyer.inner_text().strip() if client_legal_buyer and client_legal_buyer.inner_text().strip() else "ناموجود"
    client_legal_seller = page.query_selector("#n_clients2 > tr:nth-child(6) > td:nth-child(3)")
    extracted_data["تعداد فروشنده حقوقی"] = client_legal_seller.inner_text().strip() if client_legal_seller and client_legal_seller.inner_text().strip() else "ناموجود"
    
    legal_buy_vol_elem = page.query_selector("#n_clients2 > tr:nth-child(3) > td:nth-child(2) > div > div.data.d-flex.justify-content-between.align-items-center > span")
    if legal_buy_vol_elem and legal_buy_vol_elem.inner_text().strip():
        legal_buy_vol_text = legal_buy_vol_elem.inner_text().strip()
        legal_buy_vol_clean = re.sub(r"\(.*?\)", "", legal_buy_vol_text).strip()
    else:
        legal_buy_vol_clean = "ناموجود"
    extracted_data["حجم خرید حقوقی ها"] = legal_buy_vol_clean

    legal_sell_vol_elem = page.query_selector("#n_clients2 > tr:nth-child(3) > td:nth-child(3) > div > div.data.d-flex.justify-content-between.align-items-center > span")
    if legal_sell_vol_elem and legal_sell_vol_elem.inner_text().strip():
        legal_sell_vol_text = legal_sell_vol_elem.inner_text().strip()
        legal_sell_vol_clean = re.sub(r"\(.*?\)", "", legal_sell_vol_text).strip()
    else:
        legal_sell_vol_clean = "ناموجود"
    extracted_data["حجم فروش حقوقی ها"] = legal_sell_vol_clean

    # استخراج 5 ستون جدید برای اطلاعات قیمتی روزانه:
    price_buy = page.query_selector("#n_buy")
    extracted_data["قیمت خرید"] = price_buy.inner_text().strip() if price_buy and price_buy.inner_text().strip() else "ناموجود"
    
    price_deal = page.query_selector("#main-info > div:nth-child(2) > div > table:nth-child(1) > tbody > tr:nth-child(2) > td.change_val > span.n_price")
    extracted_data["قیمت معامله"] = price_deal.inner_text().strip() if price_deal and price_deal.inner_text().strip() else "ناموجود"
    
    price_sell = page.query_selector("#n_sell")
    extracted_data["قیمت فروش"] = price_sell.inner_text().strip() if price_sell and price_sell.inner_text().strip() else "ناموجود"
    
    price_first = page.query_selector("#n_first")
    extracted_data["اولین قیمت"] = price_first.inner_text().strip() if price_first and price_first.inner_text().strip() else "ناموجود"
    
    price_last = page.query_selector("#main-info > div:nth-child(2) > div > table:nth-child(1) > tbody > tr:nth-child(4) > td.change_val2 > span.n_last")
    extracted_data["قیمت پایانی"] = price_last.inner_text().strip() if price_last and price_last.inner_text().strip() else "ناموجود"
    
    return extracted_data

##############################################
# تابع تبدیل مقادیر رشته‌ای به عدد
##############################################

def convert_value(val):
    """
    - حذف جداکننده‌های هزار،
    - حذف واحدهای اختصاری (B, M, K) به همراه ضرب در مقدار مناسب،
    - حذف علامت درصد (در صورت وجود)؛
    - تبدیل مقادیر خالی یا "ناموجود" به np.nan.
    """
    try:
        if isinstance(val, str):
            val = val.strip().replace('"', '')
            if val in ["", "ناموجود"]:
                return np.nan
            val = val.replace(",", "")
            multiplier = 1
            if val.endswith("B"):
                multiplier = 1e9
                val = val[:-1]
            elif val.endswith("M"):
                multiplier = 1e6
                val = val[:-1]
            elif val.endswith("K"):
                multiplier = 1e3
                val = val[:-1]
            if val.endswith("%"):
                val = val[:-1]
            return float(val) * multiplier
        else:
            return float(val)
    except:
        return np.nan

# لیست ستون‌های عددی جهت استانداردسازی (شامل ستون‌های قیمتی جدید)
numeric_cols = [
    "میانگین ۱۰ روزه حجم معاملات",
    "P/E - TTM",
    "P/S",
    "P/B",
    "بتا ۹۰ روز",
    "ROE",
    "NAV",
    "بازده ۱ ماهه",
    "بازده ۳ ماهه",
    "بازده ۶ ماهه",
    "بازده ۱ ساله",
    "سرانه تقاضا",
    "سرانه عرضه",
    "تعداد معاملات روزانه",
    "حجم معاملات روزانه",
    "ارزش معاملات روزانه",
    "ميانگين 10 روزه سرانه خريد هر حقيقي",
    "خريد هر حقيقي (ميليون ريال)",
    "فروش هر حقيقي (ميليون ريال)",
    "قدرت خريدار به فروشنده",
    "ورود/خروج پول حقيقي (ميليون ريال)",
    "تعداد خریدار حقیقی",
    "تعداد فروشنده حقیقی",
    "حجم خرید حقیقی ها",
    "حجم فروش حقیقی ها",
    "حجم خرید حقوقی ها",
    "حجم فروش حقوقی ها",
    "قیمت خرید",
    "قیمت معامله",
    "قیمت فروش",
    "اولین قیمت",
    "قیمت پایانی"
]

def standardize_new_data(csv_file, output_csv):
    """
    این تابع فایل CSV موجود را می‌خواند، رکوردهای مربوط به تاریخ امروز برای تمامی نمادها
    (که به‌صورت طولی درج شده‌اند) را استانداردسازی می‌کند.
    ستون‌های عددی با convert_value تبدیل شده و پس از انجام تبدیل‌های لازم،
    داده‌ها بر اساس 'نام نماد' و 'تاریخ استخراج' مرتب می‌شوند.
    سپس خروجی نهایی در همان فایل (output_csv) ذخیره می‌شود.
    """
    current_date = datetime.today().strftime("%Y-%m-%d")
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file, encoding="utf-8")
        # انتخاب رکوردهای امروز (یا اعمال تغییراتی در صورت نیاز)
        mask = df["تاریخ استخراج"] == current_date
        for col in numeric_cols:
            if col in df.columns:
                df.loc[mask, col] = df.loc[mask, col].apply(convert_value)
        # تبدیل ستون‌های بازده به عدد اعشاری (تقسیم بر 100)
        returns_cols = ["بازده ۱ ماهه", "بازده ۳ ماهه", "بازده ۶ ماهه", "بازده ۱ ساله"]
        for col in returns_cols:
            if col in df.columns:
                df.loc[mask, col] = df.loc[mask, col].apply(lambda x: x/100 if pd.notna(x) else x)
        # تبدیل 'تاریخ استخراج' به نوع datetime و مرتب‌سازی نهایی
        df["تاریخ استخراج"] = pd.to_datetime(df["تاریخ استخراج"])
        df = df.sort_values(by=["نام نماد", "تاریخ استخراج"]).reset_index(drop=True)
        df.to_csv(output_csv, index=False, encoding="utf-8")
        print(f"✅ استانداردسازی و نرمال‌سازی رکوردهای موجود تکمیل شد و در '{output_csv}' ذخیره شد.")
    else:
        print("❌ فایل CSV مبنای استخراج یافت نشد.")

##############################################
# اجرای استخراج و به‌روزرسانی CSV
##############################################

with sync_playwright() as p:
    browser = p.chromium.launch(
        executable_path=r"C:\Users\mrali\AppData\Local\Chromium\Application\chrome.exe",
        headless=True
    )
    page = browser.new_page()
    for url in urls:
        try:
            record = extract_symbol_data(page, url)
            update_csv_with_record(record, output_file)
            print(f"✅ رکورد {record['نام نماد']} برای تاریخ {record['تاریخ استخراج']} ذخیره شد.")
        except Exception as e:
            print(f"🚨 خطا در استخراج داده‌ها برای {url}: {e}")
            continue
    browser.close()

# استانداردسازی نهایی و ذخیره در یک فایل CSV واحد (یکپارچه)
standardize_new_data(output_file, standardized_output)
print("✅ استخراج، به‌روزرسانی و استانداردسازی داده‌ها تکمیل شد!")



# انتظار ۴ ثانیه قبل از اجرای پردازش داده‌ها
time.sleep(4)

# اجرای فایل پردازش داده پس از اتمام اسکریپینگ
subprocess.run(["python", r"C:\Users\mrali\OneDrive\Desktop\Robot\process_data\process_data_V1.0.1.py"])

# subprocess.run(["python", "process_data.py"])

