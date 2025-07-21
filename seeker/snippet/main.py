#date: 2025-07-21T16:48:23Z
#url: https://api.github.com/gists/93e6db82eb20b8e433be426dcc426970
#owner: https://api.github.com/users/SamanHaedar78

from flet import *
import sqlite3
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import datetime

# قاعدة البيانات
conn = sqlite3.connect('sam1978.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        code TEXT,
        department TEXT,
        sale TEXT,
        comment TEXT,
        stage1 INTEGER,
        stage2 INTEGER,
        stage3 INTEGER,
        stage4 INTEGER,
        total REAL
    )
""")
conn.commit()

def main(page: Page):
    page.title = 'Fine Arts'
    page.scroll = 'auto'
    page.window.top = 1
    page.window.left = 960
    page.window.width = 390
    page.window.height = 740
    page.theme_mode = ThemeMode.LIGHT
    page.window_icon = ("theater.png")

    table_name = 'students'
    cursor.execute(f'SELECT COUNT(*) FROM {table_name}')
    row_count = cursor.fetchone()[0]

    def calculate_total():
        try:
            s1 = float(stage1.value) if stage1.value else 0.0
            s2 = float(stage2.value) if stage2.value else 0.0
            s3 = float(stage3.value) if stage3.value else 0.0
            s4 = float(stage4.value) if stage4.value else 0.0
            total_score = s1 * 0.1 + s2 * 0.2 + s3 * 0.3 + s4 * 0.4
            total.value = str(round(total_score, 4))
        except:
            total.value = "0.000"
        page.update()

    def add(e):
        cursor.execute(
            "INSERT INTO students (name, code, department, sale, comment, stage1, stage2, stage3, stage4, total) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (tname.value, tcode.value, tdepartment.value, tsale.value, tcomment.value, stage1.value, stage2.value, stage3.value, stage4.value, total.value)
        )
        conn.commit()
        page.add(Text(f"قوتابی {tname.value} بەسەرکەوتوویی تۆمارکرا", color='green'))
        page.update()

    def clyear(e):
        tname.value = ""
        tcode.value = ""
        tdepartment.value = ""
        tsale.value = ""
        tcomment.value = ""
        stage1.value = ""
        stage2.value = ""
        stage3.value = ""
        stage4.value = ""
        total.value = ""
        page.add(Text("هەموو خانەکان پاککرا", color='orange'))
        page.update()

    def serch_student(e):
        if tname.value.strip():
            c = conn.cursor()
            c.execute("SELECT * FROM students WHERE name = ?", (tname.value.strip(),))
            user = c.fetchone()
            if user:
                tname.value = user[1]
                tcode.value = user[2]
                tdepartment.value = user[3]
                tsale.value = user[4]
                tcomment.value = user[5]
                stage1.value = str(user[6]) 
                stage2.value = str(user[7]) 
                stage3.value = str(user[8])  
                stage4.value = str(user[9])
                total.value = str(user[10]) 
                page.add(Text(f"✅ داتای قوتابی {tname.value} بارکرا", color='green'))
            else:
                page.add(Text(f"❌ قوتابی بەناوی '{tname.value}' نەدۆزرایەوە", color='red'))
        else:
            page.add(Text("⚠️ تکایە ناوی قوتابی بنووسە", color='red'))
        page.update()

    def show(e):
        c = conn.cursor()
        c.execute("SELECT * FROM students ORDER BY total DESC")
        users = c.fetchall()
        if users:
            keys = ['id', 'name', 'code', 'department', 'sale', 'comment', 'stage1', 'stage2', 'stage3', 'stage4', 'total']
            result = [dict(zip(keys, values)) for values in users]
            for x in result:
                page.add(
                    Card(
                        color='black',
                        content=Container(
                            content=Column([
                                ListTile(
                                    leading=Icon(Icons.PERSON, color='white'),
                                    title=Text(f"ناوی قوتابی: {x['name']}", color='white', size=12, rtl=True),
                                    subtitle=Column([
                                        Text(f"کۆدی قوتابی: {x['code']}", color='white', size=12, rtl=True),
                                        Text(f"بەش: {x['department']}", color='white', size=12, rtl=True),
                                        Text(f"ساڵی دەرچوون: {x['sale']}", color='white', size=12, rtl=True),
                                        Text(f"تێبینی: {x['comment']}", color='white', size=12, rtl=True),
                                        Text(f"قۆناغی یەکەم: {x['stage1']}", color='white', size=12, rtl=True),
                                        Text(f"قۆناغی دووەم: {x['stage2']}", color='white', size=12, rtl=True),
                                        Text(f"قۆناغی سێیەم: {x['stage3']}", color='white', size=12, rtl=True),
                                        Text(f"قۆناغی چوارەم: {x['stage4']}", color='white', size=12, rtl=True),
                                        Text(f"کۆنمرە : {x['total']}", color='white', size=12, rtl=True)
                                    ], alignment=MainAxisAlignment.CENTER, rtl=True),
                                )
                            ], spacing=10, alignment=MainAxisAlignment.START),
                            padding=padding.all(10),
                            width=300,
                            height=250
                        )
                    )
                )
            page.update()

    def remove(e):
        if tname.value:
            cursor.execute("DELETE FROM students WHERE name = ?", (tname.value,))
            conn.commit()
            page.add(Text(f"قوتابی بەناوی {tname.value} سڕایەوە", color='red'))
        else:
            page.add(Text("تکایە ناوی قوتابی بنووسە", color='red'))
        page.update()

    def update_student(e):
        if not tname.value.strip():
            page.add(Text("⚠️ تکایە ناوی قوتابی بنووسە بۆ گۆرانکاری", color="red"))
            page.update()
            return

        cursor.execute("""
            UPDATE students SET
                code = ?, department = ?, sale = ?, comment = ?,
                stage1 = ?, stage2 = ?, stage3 = ?, stage4 = ?, total = ?
            WHERE name = ?
        """, (
            tcode.value, tdepartment.value, tsale.value, tcomment.value,
            stage1.value, stage2.value, stage3.value, stage4.value, total.value,
            tname.value
        ))
        conn.commit()
        page.add(Text(f"✅ داتای قوتابی '{tname.value}' نوێکرایەوە", color="green"))
        page.update()

    def export_to_pdf(e):
        cursor.execute("SELECT * FROM students ORDER BY total DESC")
        students = cursor.fetchall()

        if not students:
            page.add(Text("هیچ داتایەک بۆ ناردن نییە", color="red"))
            page.update()
            return

        filename = f"students_report_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pdf"
        c = canvas.Canvas(filename, pagesize=letter)
        width, height = letter

        c.setFont("Helvetica-Bold", 14)
        c.drawString(200, height - 40, "ڕاپۆرتی قوتابیان - Fine Arts")

        c.setFont("Helvetica", 10)
        y = height - 70
        for student in students:
            text = f"👤 {student[1]} | Code: {student[2]} | Dept: {student[3]} | Year: {student[4]} | Total: {student[10]}"
            c.drawString(30, y, text)
            y -= 15
            if y < 50:
                c.showPage()
                c.setFont("Helvetica", 10)
                y = height - 50

        c.save()
        page.add(Text(f"✅ فایل PDF بەسەرکەوتوویی دروستکرا: {filename}", color="green"))
        page.update()

    # ====== Widgets ======
    tname = TextField(label="ناوی قوتابی", icon=Icons.PERSON, rtl=True, height=30)
    tcode = TextField(label="کۆدی قوتابی", icon=Icons.CODE, rtl=True, height=30)
    tdepartment = TextField(label="بەش", icon=Icons.LOCAL_FIRE_DEPARTMENT, rtl=True, height=30)
    tsale = TextField(label=" ساڵی دەرچوون", icon=Icons.PIN, rtl=True, height=30)
    tcomment = TextField(label="تێبینی", icon=Icons.COMMENT, rtl=True, height=30)

    stage1 = TextField(label="قۆناغی یەکەم", width=140, height=30, on_change=lambda e: calculate_total())
    stage2 = TextField(label="قۆناغی دووەم", width=140, height=30, on_change=lambda e: calculate_total())
    stage3 = TextField(label="قۆناغی سێیەم", width=140, height=30, on_change=lambda e: calculate_total())
    stage4 = TextField(label="قۆناغی چوارەم", width=140, height=30, on_change=lambda e: calculate_total())
    total = TextField(label="کۆنمرە", width=130, height=30, read_only=True)

    addbutton = ElevatedButton("تۆمار", icon=Icons.ADD, icon_color='green', width=100, height=40, bgcolor='green', color='white', on_click=add)
    removebutton = ElevatedButton("سڕینەوە", icon=Icons.DELETE, icon_color='red', width=100, height=40, bgcolor='red', color='white', on_click=remove)
    clearbutton = ElevatedButton("پاککردن", icon=Icons.CLEAR, icon_color='orange', width=100, height=40, bgcolor='orange', color='white', on_click=clyear)
    serchbutton = ElevatedButton("گەڕان", icon=Icons.SEARCH, icon_color='purple', width=100, height=40, bgcolor='purple', color='white', on_click=serch_student)
    showbutton = ElevatedButton("پیشاندان", icon=Icons.SHOW_CHART, icon_color='blue', width=100, height=40, bgcolor='blue', color='white', on_click=show)
    updatebutton = ElevatedButton("✏️ گۆرانکاری", icon=Icons.EDIT, icon_color='blue', width=120, height=40, bgcolor='yellow', color='black', on_click=update_student)
    export_pdf_button = ElevatedButton("📄 PDF ناردن", icon=Icons.PICTURE_AS_PDF, icon_color='red', width=120, height=40, bgcolor='black', color='white', on_click=export_to_pdf)

    # ====== Layout ======
    page.add(
        Row([Image(src="sue.png", width=125)], alignment=MainAxisAlignment.CENTER),

        Row([
            Text(" كۆلێژی هونەرە جوانەکان ", size=16, color="black", text_align=TextAlign.CENTER),
            Text(str(row_count), size=14, color="red", text_align=TextAlign.CENTER)
        ], alignment=MainAxisAlignment.CENTER, rtl=True),

        tname, tcode, tdepartment, tsale, tcomment,

        Row([showbutton, clearbutton], alignment=MainAxisAlignment.CENTER),
        Row([updatebutton, export_pdf_button], alignment=MainAxisAlignment.CENTER),
        Row([stage1, stage2], alignment=MainAxisAlignment.CENTER, rtl=True),
        Row([stage3, stage4], alignment=MainAxisAlignment.CENTER, rtl=True),
        Row([total], alignment=MainAxisAlignment.CENTER),
        Row([addbutton, removebutton, serchbutton], alignment=MainAxisAlignment.CENTER, rtl=True)
    )

    page.update()

app(main)
