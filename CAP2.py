from flask import Flask, flash,render_template, render_template_string, request, redirect, url_for,session
import smtplib
import sqlite3
import secrets
import time
import os
import random
import pandas as pd

app = Flask(__name__)
app.secret_key = "super_secret_key_123"

import os

DB_PATH = 'Database.db'
DATA_PATH = 'Dataset.csv.xlsx'


# -------------------------------
# IMAGE HELPER (ADD HERE ✅)
# -------------------------------
IMG_FOLDER = os.path.join('static', 'IMAGE_DATASET COTY')

def get_student_image(enroll):
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        p = os.path.join(IMG_FOLDER, f"{enroll}{ext}")
        if os.path.exists(p):
            return '/' + p.replace('\\', '/')
    return '/static/default.jpg'

# -------------------------------
# Load Dataset Once
# -------------------------------
df = pd.read_excel(DATA_PATH)
df.columns = [c.strip().replace(" ", "_").upper() for c in df.columns]

# -------------------------------
# Home Page
# -------------------------------
@app.route('/')
def welcome():
    return render_template('home1.html')

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/features")
def features():
    return render_template("Feature.html")

@app.route('/outpass_m')
def outpass_m():
    return render_template('outpassmodule.html')

@app.route('/student_module')
def studentmodule():
    return redirect(url_for('otp_page'))

@app.route('/warden1')
def warden1():
    return redirect(url_for('hod_intermediate'))

@app.route("/request_dashboard")
def request_dashboard_dash():
    return render_template("request.html")

import sqlite3

import os



    
def create_new_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS OutpassRequests2(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        Enrollment_No TEXT,
        From_Date TEXT,
        To_Date TEXT,
        Out_Time TEXT,
        In_Time TEXT,
        Reason TEXT,
        Hostel_Name TEXT,
        Address TEXT,
        Status TEXT,
        Action_Time TEXT,
        return_date TEXT,
        return_time TEXT,
        return_status TEXT
    )
    """)
    conn.commit()
    conn.close()


    # # 2️⃣ Check existing columns
    # cursor.execute("PRAGMA table_info(OutpassRequests2);")
    # existing_columns = [col[1] for col in cursor.fetchall()]

    # # 3️⃣ Add Address column if not exists
    # if "Address" not in existing_columns:
    #     cursor.execute("ALTER TABLE OutpassRequests2 ADD COLUMN Address TEXT")
    #     print("Address column added successfully.")
    # else:
    #     print("Address column already exists, skipping.")

    # # 4️⃣ Add Status column if not exists
    # if "Status" not in existing_columns:
    #     cursor.execute("ALTER TABLE OutpassRequests2 ADD COLUMN Status TEXT")
    #     print("Status column added successfully.")
    # else:
    #     print("Status column already exists, skipping.")

    # conn.commit()
    # conn.close()
    print("Database and table initialized successfully.")

# Call the function at startup
create_new_db()


# ================= EMAIL CONFIG =================
SENDER_EMAIL = "arunasbamnekar@gmail.com"
SENDER_PASSWORD = "sslbtzwlvtjqygqj"

# ================= OTP SEND PAGE =================

OTP_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <title>OTP Verification</title>
    <meta charset="UTF-8">
    <style>
        body {
            margin: 0;
            height: 100vh;
            font-family: 'Segoe UI', sans-serif;
            background: url('{{ url_for("static", filename="college_bg.jpg") }}') no-repeat center center/cover;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        body::before {
            content: "";
            position: absolute;
            inset: 0;
            background: rgba(0, 0, 0, 0.35);
            backdrop-filter: blur(8px);
        }

        .card {
            position: relative;
            width: 360px;
            padding: 35px 30px;
            background: rgba(255, 255, 255, 0.18);
            backdrop-filter: blur(15px);
            border-radius: 18px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
            text-align: center;
            color: #ffffff;
        }

        h2 {
            margin-bottom: 10px;
            font-size: 22px;
            color: #e6f4ff;
        }

        p {
            font-size: 14px;
            color: #ffdcdc;
        }

        input {
            width: 100%;
            padding: 12px;
            border: none;
            border-radius: 8px;
            margin-top: 15px;
            background: rgba(255, 255, 255, 0.25);
            color: #fff;
            text-align: center;
            font-size: 14px;
        }

        input::placeholder {
            color: #e0f1ff;
        }

        button {
            width: 100%;
            margin-top: 20px;
            padding: 12px;
            border: none;
            border-radius: 8px;
            background: linear-gradient(90deg, #007bff, #00c6ff);
            color: white;
            font-size: 15px;
            font-weight: bold;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="card">
        <h2>📧 Email OTP Verification</h2>

        {% with messages = get_flashed_messages() %}
            {% for msg in messages %}
                <p>{{ msg }}</p>
            {% endfor %}
        {% endwith %}

        <form method="POST">
            <input type="email" name="email" placeholder="Enter Email" required>
            <button type="submit">Send OTP</button>
        </form>
    </div>
</body>
</html>
"""

# ================= VERIFY OTP PAGE =================

VERIFY_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Verify OTP</title>
    <style>
        body {
            margin: 0;
            height: 100vh;
            background: url('/static/college_bg.jpg') no-repeat center center/cover;
            display: flex;
            justify-content: center;
            align-items: center;
            font-family: Arial, sans-serif;
        }

        .overlay {
            position: absolute;
            inset: 0;
            backdrop-filter: blur(8px);
            background: rgba(0, 0, 0, 0.4);
        }

        .card {
            position: relative;
            background: white;
            padding: 30px;
            width: 320px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 10px 25px rgba(0,0,0,0.3);
            z-index: 1;
        }

        input {
            width: 90%;
            padding: 10px;
            margin-bottom: 15px;
            font-size: 16px;
        }

        button {
            width: 100%;
            padding: 10px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
        }

        .resend-btn {
            margin-top: 10px;
            background: #28a745;
        }

        .timer {
            margin-top: 10px;
            font-weight: bold;
            color: #dc3545;
        }

        .msg {
            color: red;
        }
    </style>
</head>

<body>
<div class="overlay"></div>

<div class="card">
    <h2>🔐 Enter OTP</h2>

    {% with messages = get_flashed_messages() %}
        {% for msg in messages %}
            <p class="msg">{{ msg }}</p>
        {% endfor %}
    {% endwith %}

    <form method="POST">
        <input type="number" name="otp" placeholder="Enter OTP" required>
        <button type="submit">Verify OTP</button>
    </form>

    {% if show_resend %}
    <form method="POST">
        <button type="submit" name="resend" class="resend-btn">🔄 Resend OTP</button>
    </form>
    {% endif %}

    <div class="timer">
        Time left: <span id="time">01:00</span>
    </div>
</div>

<script>
let time = 60;
const timerEl = document.getElementById("time");

const countdown = setInterval(() => {
    time--;
    timerEl.textContent =
        `00:${time.toString().padStart(2,'0')}`;

    if (time <= 0) {
        clearInterval(countdown);
        timerEl.textContent = "Expired";
    }
}, 1000);
</script>
</body>
</html>
"""

# ================= SEND EMAIL FUNCTION =================

def send_email(receiver_email, otp):
    msg = f"Subject: OTP Verification\n\nYour OTP is: {otp}"
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, receiver_email, msg)

# ================= OTP ROUTE =================
@app.route('/otp', methods=['GET', 'POST'])
def otp_page():
    if request.method == 'POST':
        email = request.form['email']
        otp = random.randint(100000, 999999)

        session['email'] = email
        session['otp'] = str(otp)
        session['otp_time'] = time.time()

        send_email(email, otp)
        flash("✅ OTP sent successfully")
        return redirect(url_for('verify_otp'))  # ✅ redirect

    return render_template_string(OTP_HTML)

# ================= VERIFY OTP ROUTE =================

@app.route('/verify_otp', methods=['GET', 'POST'])
def verify_otp():
    if 'email' not in session:
        return redirect(url_for('otp_page'))

    if request.method == 'POST':

        if 'resend' in request.form:
            otp = random.randint(100000, 999999)
            session['otp'] = str(otp)
            session['otp_time'] = time.time()
            send_email(session['email'], otp)
            flash("✅ New OTP sent")
            return redirect(url_for('verify_otp'))

        entered = request.form['otp']
        saved = session.get('otp')
        sent_time = session.get('otp_time')

        if not saved or time.time() - sent_time > 60:
            flash("❌ OTP expired")
            return render_template_string(VERIFY_HTML, show_resend=True)

        if entered == saved:
            session.pop('otp', None)
            session.pop('otp_time', None)
            flash("✅ OTP Verified")
            return redirect(url_for('verify_enrollment'))

        flash("❌ Invalid OTP")
        return render_template_string(VERIFY_HTML, show_resend=True)

    return render_template_string(VERIFY_HTML, show_resend=False)

# -------------------------------
# Enrollment Verification Page
# --------------------------------
@app.route('/verify', methods=['GET', 'POST'])
def verify_enrollment():
    email = session.get('email')  # 👈 fetch email from previous page

    html = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Enrollment Verification | GRW Polytechnic Tasgaon</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
            body {
                margin: 0; padding: 0; height: 100vh;
                font-family: 'Poppins', sans-serif;
                background-image: url("{{ url_for('static', filename='college_bg.jpg') }}");
                background-size: cover; background-position: center;
                display: flex; justify-content: center; align-items: center;
                overflow: hidden;
            }
            body::before {
                content: ""; position: absolute;
                inset: 0;
                background: rgba(0,0,0,0.3);
                backdrop-filter: blur(8px);
            }
            .container {
                position: relative;
                background: rgba(255,255,255,0.15);
                backdrop-filter: blur(15px);
                border-radius: 20px;
                padding: 40px 50px;
                text-align: center;
                width: 380px;
                box-shadow: 0 4px 30px rgba(0,0,0,0.4);
            }
            h1 { font-size: 22px; color: #fff; }
            h2 { font-size: 18px; color: #d9f1ff; margin-bottom: 10px; }
            .email { font-size: 14px; color: #ffeaa7; margin-bottom: 20px; }
            input {
                width: 100%; padding: 12px;
                border: none; border-radius: 8px;
                margin-bottom: 20px;
                background: rgba(255,255,255,0.25);
                color: #fff; text-align: center;
            }
            button {
                width: 100%; padding: 12px;
                border: none; border-radius: 8px;
                background: linear-gradient(90deg,#007bff,#00c6ff);
                color: white; font-weight: bold;
                cursor: pointer;
            }
            .footer { margin-top: 15px; font-size: 13px; color: #e1e1e1; }
        </style>
        <script>function showAlert(msg){alert(msg);}</script>
    </head>
    <body>
        <div class="container">
            <h1>🏫 GRW Polytechnic, Tasgaon</h1>
            <h2>Enrollment Verification</h2>

            <div class="email">📧 {{ email }}</div>

            <form method="POST">
                <input type="text" name="enroll" placeholder="Enter Enrollment Number" required>
                <button type="submit">Verify Enrollment</button>
            </form>

            <div class="footer">© 2025 GRW Polytechnic Tasgaon</div>
        </div>

        {% if alert %}
        <script>showAlert("{{ alert }}");</script>
        {% endif %}
    </body>
    </html>
    '''

    if request.method == 'POST':
        enroll = request.form['enroll'].strip()
        email = session.get('email')

        student = df.loc[
            (df['ENROLLMENT_NO'].astype(str) == enroll) &
            (df['EMAIL'].str.lower() == email.lower())
        ]

        if not student.empty:
            return redirect(url_for('outpass_form', enroll=enroll))
        else:
            return render_template_string(
                html,
                email=email,
                alert="❌ Enrollment number does not match this email!"
            )

    return render_template_string(html, email=email)

# -------------------------------
# Outpass Form
# -------------------------------

@app.route('/form', methods=['GET', 'POST'])
def outpass_form():

    if request.method == 'POST':
        enroll_no = request.form.get('enroll')     # form value
    else:
        enroll_no = request.args.get('enroll')     # url value only first time

    if request.method == 'POST':
        # Basic fields
        enroll_no = request.form.get('enroll')
        from_date = request.form.get('from_date')
        to_date = request.form.get('to_date')
        reason = request.form.get('reason')

        # ✅ NEW FIELDS
        hostel_name = request.form.get('hostel_name')
        address = request.form.get('address')

        # Outgoing time parts
        out_hour = request.form.get('out_hour')
        out_minute = request.form.get('out_minute')
        out_ampm = request.form.get('out_ampm')

        # Incoming time parts
        in_hour = request.form.get('in_hour')
        in_minute = request.form.get('in_minute')
        in_ampm = request.form.get('in_ampm')
        
        # Combine time values
        out_time = f"{out_hour}:{out_minute} {out_ampm}"
        in_time = f"{in_hour}:{in_minute} {in_ampm}"

        # Validation
        if not all([
            enroll_no, from_date, to_date, reason,
            hostel_name, address,
            out_hour, out_minute, out_ampm,
            in_hour, in_minute, in_ampm
        ]):
            return "All fields are required", 400

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS OutpassRequests2 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                Enrollment_No TEXT,
                From_Date TEXT,
                To_Date TEXT,
                Out_Time TEXT,
                In_Time TEXT,
                Reason TEXT,
                Hostel_Name TEXT,
                Address TEXT,
                Status TEXT
            )
        ''')
        
        cursor.execute("""
            INSERT INTO OutpassRequests2
            (Enrollment_No, From_Date, To_Date, Out_Time, In_Time, Reason, Hostel_Name, Address, Status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            enroll_no,
            from_date,
            to_date,
            out_time,
            in_time,
            reason,
            hostel_name,
            address,
            "Pending"
        ))

        conn.commit()
        conn.close()

        return redirect(url_for(
            'success_page',
            enroll=enroll_no,
            from_date=from_date,
            to_date=to_date,
            out_time=out_time,
            in_time=in_time,
            reason=reason,
            hostel_name=hostel_name,
            address=address
        ))

    return render_template('index.html', enroll_no=enroll_no)
# -------------------------------
# Utility : CLEAN PHONE NUMBER
# -------------------------------
def clean_phone(value):
    if value is None:
        return "N/A"

    # Handle pandas / numpy NaN safely
    try:
        if pd.isna(value):
            return "N/A"
    except:
        pass

    value = str(value).strip()

    if value.lower() in ["nan", "none", ""]:
        return "N/A"

    # Remove Excel float issue (.0)
    if value.endswith(".0"):
        value = value[:-2]

    # Remove spaces
    value = value.replace(" ", "")

    return value


# -------------------------------
# Success Page (with student image + check status link)
# -------------------------------
@app.route('/success/<enroll>')
def success_page(enroll):
    import os
    from flask import request

    # ---------------- FIND STUDENT ----------------
    student = df.loc[df['ENROLLMENT_NO'].astype(str) == str(enroll)]
    if student.empty:
        return "<h3 style='color:red;'>Student not found!</h3>"

    record = student.iloc[0].to_dict()

    # ---------------- OUTPASS DATA ----------------
    from_date = request.args.get('from_date', 'N/A')
    to_date = request.args.get('to_date', 'N/A')
    reason = request.args.get('reason', 'N/A')
    hostel_name = request.args.get('hostel_name', 'N/A')
    address = request.args.get('address', 'N/A')
    out_time = request.args.get('out_time', 'N/A')
    in_time = request.args.get('in_time', 'N/A')

    # ---------------- PHONE FIX ----------------
    student_phone = clean_phone(record.get('STUDENT_PHONE_NO'))
    parent_phone = clean_phone(record.get('PARENTS_PHONE_NO'))

    # ---------------- IMAGE HANDLING ----------------
    img_folder = os.path.join('static', 'IMAGE_DATASET COTY')
    image_found = None

    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        p = os.path.join(img_folder, f"{enroll}{ext}")
        if os.path.exists(p):
            image_found = p
            break

    img_url = '/' + image_found.replace('\\', '/') if image_found else '/static/default.jpg'

    # ---------------- HTML ----------------
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Outpass Submitted | GRW Polytechnic</title>

<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
body {{
    margin:0;
    height:100vh;
    font-family:'Poppins',sans-serif;
    background:url('/static/college_bg.jpg') center/cover;
    display:flex;
    justify-content:center;
    align-items:center;
}}
body::before {{
    content:"";
    position:fixed;
    inset:0;
    background:rgba(0,0,0,0.45);
    backdrop-filter:blur(8px);
}}
.card {{
    position:relative;
    background:rgba(255,255,255,0.15);
    backdrop-filter:blur(18px);
    border-radius:22px;
    padding:30px 45px;
    width:560px;
    color:#fff;
    text-align:center;
    box-shadow:0 8px 35px rgba(0,0,0,0.6);
    z-index:1;
}}
.profile-pic {{
    width:150px;height:150px;
    border-radius:50%;
    border:3px solid #fff;
    margin:0 auto 15px;
    overflow:hidden;
    background:#fff;
}}
.profile-pic img {{
    width:100%;height:100%;
    object-fit:cover;
}}
table {{ width:100%; font-size:14px; }}
td {{ padding:6px 0; }}
hr {{ border:none;height:1px;background:rgba(255,255,255,0.3); }}
a.btn {{
    background:linear-gradient(90deg,#007bff,#00c6ff);
    color:white;
    padding:10px 20px;
    border-radius:8px;
    font-weight:bold;
    margin:8px 4px;
    display:inline-block;
    text-decoration:none;
}}
a.btn:hover {{ transform:scale(1.05); }}
</style>
</head>

<body>
<div class="card">

<div class="profile-pic">
<img src="{img_url}">
</div>

<h2>{record.get('NAME','N/A')}</h2>

<table>
<tr><td><b>Enrollment No:</b></td><td>{enroll}</td></tr>
<tr><td><b>Outpass ID:</b></td><td>{record.get('OUTPASS_ID','N/A')}</td></tr>
<tr><td><b>Department:</b></td><td>{record.get('DEPARTMENT','N/A')}</td></tr>
<tr><td><b>Year:</b></td><td>{record.get('YEAR','N/A')}</td></tr>
<tr><td><b>Academic Year:</b></td><td>{record.get('YEAR.1','N/A')}</td></tr>
<tr><td><b>Student Phone:</b></td><td>{student_phone}</td></tr>
<tr><td><b>Parent Phone:</b></td><td>{parent_phone}</td></tr>

<tr><td colspan="2"><hr></td></tr>

<tr><td><b>From Date:</b></td><td>{from_date}</td></tr>
<tr><td><b>To Date:</b></td><td>{to_date}</td></tr>
<tr><td><b>Reason:</b></td><td>{reason}</td></tr>
<tr><td><b>Hostel Name:</b></td><td>{hostel_name}</td></tr>
<tr><td><b>Address:</b></td><td>{address}</td></tr>
<tr><td><b>Out Time:</b></td><td>{out_time}</td></tr>
<tr><td><b>In Time:</b></td><td>{in_time}</td></tr>
</table>

<br>
<a href="/outpass_m" class="btn">📨 Submit</a>
<a href="/" class="btn">🏠 Home</a>
<a href="/check_status?enroll={enroll}" class="btn">📋 Status</a>

</div>
</body>
</html>
"""
    return html

# -------------------------------
# HOD Login Page
# -------------------------------
from flask import Flask, request, redirect
@app.route('/hod_intermediate', methods=['GET', 'POST'])
def hod_intermediate():
    HOD_USERNAME = 'Sujata Patil'
    HOD_PASSWORD = 'GRWPCO'
    error_msg = ""

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username == HOD_USERNAME and password == HOD_PASSWORD:
            return redirect('/warden')
        else:
            error_msg = "Incorrect username or password!"

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>HOD Login</title>
        <style>
            body {{
                font-family: Arial;
                display:flex;
                justify-content:center;
                align-items:center;
                height:100vh;
                background:#f0f7ff;
            }}
            .container {{
                background:white;
                padding:30px;
                border-radius:15px;
                box-shadow:0 0 20px rgba(0,0,0,0.3);
                text-align:center;
                width:350px;
            }}
            input {{
                padding:10px;
                margin:5px 0;
                width:100%;
                border-radius:5px;
                border:1px solid #ccc;
            }}
            button {{
                padding:10px 20px;
                margin-top:10px;
                border:none;
                border-radius:5px;
                background:#007bff;
                color:white;
                font-weight:bold;
                cursor:pointer;
                width:100%;
            }}
            .error {{ color:red; margin-bottom:10px; }}
        </style>
        <script>
            function showAlert(msg) {{
                if(msg) {{
                    alert(msg);
                }}
            }}
        </script>
    </head>
    <body onload="showAlert('{error_msg}')">
        <div class="container">
            <h2>✅ Request Submitted Successfully</h2>
            <p>Enter HOD credentials to continue to Warden Dashboard:</p>

            <form method="POST">
                <input type="text" name="username" placeholder="HOD Username" required>
                <input type="password" name="password" placeholder="HOD Password" required>
                <button type="submit">📨 Login</button>
            </form>
        </div>
    </body>
    </html>
    """

from flask import Flask, request, redirect, url_for, render_template_string
import sqlite3, os
import pandas as pd
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# -------------------------------
# LOAD EXCEL DATA
# -------------------------------
df = pd.read_excel("Dataset_New.xlsx")
df.columns = df.columns.str.strip().str.upper()
df['ENROLLMENT_NO'] = df['ENROLLMENT_NO'].astype(str).str.strip()

SENDER_EMAIL = "arunasbamnekar@gmail.com"
SENDER_PASSWORD = "sslbtzwlvtjqygqj"

from flask import request, render_template_string
import sqlite3, os
from datetime import datetime

# -------------------------------
# WARDEN DASHBOARD
# -------------------------------


def ensure_action_time_column():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(OutpassRequests2)")
    columns = [col[1] for col in cursor.fetchall()]
    if "Action_Time" not in columns:
        cursor.execute("ALTER TABLE OutpassRequests2 ADD COLUMN Action_Time TEXT")
        print("✅ Action_Time column added")
    else:
        print("✅ Action_Time column already exists")
    conn.commit()
    conn.close()

# Call it at startup
ensure_action_time_column()


from datetime import datetime

@app.route('/warden')
def warden_dashboard():
    search = request.args.get('search', '').strip()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if search:
        cursor.execute("""
            SELECT id, Enrollment_No, From_Date, To_Date, Out_Time, In_Time,
                   Reason, Hostel_Name, Address, Status, Action_Time
            FROM OutpassRequests2
            WHERE Enrollment_No LIKE ?
        """, (f'%{search}%',))
    else:
        cursor.execute("""
            SELECT id, Enrollment_No, From_Date, To_Date, Out_Time, In_Time,
                   Reason, Hostel_Name, Address, Status, Action_Time
            FROM OutpassRequests2
        """)

    rows = cursor.fetchall()
    conn.close()

    enriched = []
    img_folder = os.path.join('static', 'IMAGE_DATASET COTY')

    for r in rows:
        (req_id, enroll, from_date, to_date, out_time, in_time,
         reason, hostel_name, address, status, Action_Time) = r

        # Student info
        student = df.loc[df['ENROLLMENT_NO'].astype(str) == str(enroll)]
        if not student.empty:
            rec = student.iloc[0]
            student_phone = rec.get('STUDENT_PHONE_NO', 'N/A')
            parent_phone = rec.get('PARENTS_PHONE_NO', 'N/A')
            name = rec.get('NAME', 'N/A')
            Outpass_ID = rec.get('OUTPASS_ID', 'N/A')
            dept = rec.get('DEPARTMENT', 'N/A')
            year = rec.get('YEAR', 'N/A')
            acad_year = rec.get('YEAR.1', 'N/A')
        else:
            student_phone = parent_phone = name = Outpass_ID = dept = year = acad_year = 'N/A'

        # Student image
        image_found = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            possible_path = os.path.join(img_folder, f"{enroll}{ext}")
            if os.path.exists(possible_path):
                image_found = '/' + possible_path.replace('\\', '/')
                break
        if not image_found:
            image_found = '/static/default.jpg'

        # Info Card
        card_html = f"""
        <button onclick="toggleCard('card_{req_id}')"
                style="padding:6px 12px; border:none; border-radius:6px;
                       background:#007bff; color:white; cursor:pointer;
                       font-weight:bold;">
            View Info
        </button>

        <div id="card_{req_id}" style="display:none; margin-top:10px;
             width:300px; background:#f0f7ff; border-radius:15px;
             padding:15px; text-align:center; font-family:Arial;
             box-shadow:0 4px 8px rgba(0,0,0,0.2);">

            <img src="{image_found}" style="width:100px; height:100px;
                 border-radius:50%; object-fit:cover; margin-bottom:10px;"><br>

            <b>{name}</b><br>
            <small>Enrollment: {enroll}</small><br>
            <small>Outpass_ID: {Outpass_ID}</small><br>
            <small>Dept: {dept}</small><br>
            <small>Year: {year}</small><br>
            <small>Acad Year: {acad_year}</small><br>
            <hr>
            <small>From: {from_date}</small><br>
            <small>To: {to_date}</small><br>
            <small>Reason: {reason}</small><br>
            <small>Hostel: {hostel_name}</small><br>
            <small>Address: {address}</small><br>
            <small>Out Time: {out_time}</small><br>
            <small>In Time: {in_time}</small><br>
        </div>
        """

        enriched.append({
            'id': req_id,
            'enroll': enroll,
            'student_phone': student_phone,
            'parent_phone': parent_phone,
            'card_html': card_html,
            'status': status,
            'action_time': Action_Time
        })

    return render_template_string(TEMPLATE, enriched=enriched, request=request)

SENDER_EMAIL = "arunasbamnekar@gmail.com"
SENDER_PASSWORD = "sslbtzwlvtjqygqj"


@app.route('/update_status/<int:req_id>/<status>')
def update_status_email(req_id, status):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        Action_Time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Update status and time
        cursor.execute("UPDATE OutpassRequests2 SET Status=?, Action_Time=? WHERE id=?",
                       (status, Action_Time, req_id))
        conn.commit()

        cursor.execute("SELECT Enrollment_No FROM OutpassRequests2 WHERE id=?", (req_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            print("Request not found")
            return redirect(url_for('warden_dashboard'))

        enroll = str(row[0]).strip()
        df['ENROLLMENT_NO'] = df['ENROLLMENT_NO'].astype(str).str.strip()
        student = df[df['ENROLLMENT_NO'] == enroll]

        if student.empty:
            print(f"No student found for {enroll}")
            return redirect(url_for('warden_dashboard'))

        student_email = str(student.iloc[0].get('EMAIL')).strip()
        student_name = student.iloc[0].get('NAME', 'Student')

        if not student_email or "@" not in student_email:
            print("INVALID EMAIL:", student_email)
            return redirect(url_for('warden_dashboard'))

        # Email sending
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = student_email
        msg['Subject'] = f"Outpass Request {status}"
        msg.attach(MIMEText(f"Hello {student_name},\n\nYour outpass request has been {status.lower()}.\n\nRegards,\nWarden", 'plain'))

        # Use same method as working first code
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()

        print("EMAIL SENT SUCCESSFULLY to", student_email)

    except Exception as e:
        print("EMAIL ERROR:", e)

    return redirect(url_for('warden_dashboard'))

# -------------------------------
# UPDATED HTML TEMPLATE
# -------------------------------
TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<title>Warden Dashboard</title>
<style>
body { font-family: Arial; padding: 20px; background: #f0f7ff; }
table { width: 100%; border-collapse: collapse; margin-top: 20px; }
th, td { border: 1px solid #ccc; padding: 8px; text-align: center; vertical-align: top; }
th { background: #007bff; color: white; }
tr:nth-child(even) { background: #f9f9f9; }

.btn { padding: 6px 12px; color: white; border-radius: 6px; text-decoration: none; font-weight: bold; margin: 2px; display:inline-block; }
.approve { background: #28a745; }
.reject { background: #dc3545; }

.call-btn, .whatsapp-btn { margin-left: 6px; font-size: 16px; text-decoration: none; }
.call-btn:hover { color: #28a745; }
.whatsapp-btn:hover { color: #25d366; }

.top-left-btn {
    position: fixed; top: 20px; left: 20px;
    background: #343a40; color: white;
    padding: 10px 16px; border-radius: 8px;
    text-decoration: none; font-weight: bold;
}

/* NEW LOGS BUTTON TOP RIGHT */
.top-right-btn {
    position: fixed; top: 20px; right: 20px;
    background: #198754; color: white;
    padding: 10px 16px; border-radius: 8px;
    text-decoration: none; font-weight: bold;
}
</style>

<script>
function toggleCard(id) {
    const el = document.getElementById(id);
    el.style.display = (el.style.display === "none") ? "block" : "none";
}
</script>
</head>

<body>

<a href="{{ url_for('request_dashboard') }}" class="top-left-btn">
   Go to Request Dashboard
</a>

<a href="{{ url_for('logs') }}" class="top-right-btn">
    View Logs
</a>

<center><h2>🏫 Warden Dashboard - Outpass Requests</h2></center>

<form method="get" style="text-align:center; margin-bottom:20px;">
    <input type="text" name="search" placeholder="Search by Enrollment Number"
           value="{{ request.args.get('search','') }}"
           style="width:300px; padding:8px; border-radius:8px;">
    <button type="submit" style="padding:8px 15px; border:none; background:#007bff; color:white; border-radius:8px;">
        Search
    </button>
</form>

<table>
<thead>
<tr>
    <th>Enrollment No</th>
    <th>Student Phone</th>
    <th>Parent Phone</th>
    <th>Info Card</th>
    <th>Action</th>
    <th>Status</th>
    <th>Action Time</th>
</tr>
</thead>

<tbody>
{% for r in enriched %}
<tr>

<td>{{ r.enroll }}</td>

<td>
{{ r.student_phone }}
{% if r.student_phone != 'N/A' %}
<a href="tel:{{ r.student_phone }}" class="call-btn">📞</a>
<a href="https://wa.me/{{ r.student_phone }}" target="_blank" class="whatsapp-btn">💬</a>
{% endif %}
</td>

<td>
{{ r.parent_phone }}
{% if r.parent_phone != 'N/A' %}
<a href="tel:{{ r.parent_phone }}" class="call-btn">📞</a>
<a href="https://wa.me/{{ r.parent_phone }}" target="_blank" class="whatsapp-btn">💬</a>
{% endif %}
</td>

<td>{{ r.card_html|safe }}</td>

<td>
{% if r.status == 'Pending' %}
<a class="btn approve" href="{{ url_for('update_status_email', req_id=r.id, status='Approved') }}">Approve</a>
<a class="btn reject" href="{{ url_for('update_status_email', req_id=r.id, status='Rejected') }}">Reject</a>
{% else %}-{% endif %}
</td>

<td><b>{{ r.status }}</b></td>

<td>
{% if r.Action_Time %}
{{ r.Action_Time }}
{% else %}
-
{% endif %}
</td>

</tr>
{% endfor %}
</tbody>
</table>

</body>
</html>"""

# =====================================================
# DAILY LOG VIEW (ALL DAY-WISE WARDEN ACTIONS)
# =====================================================

from datetime import date
import sqlite3
import pandas as pd
from flask import request, render_template_string

DATASET_PATH = "Dataset_New.xlsx"

# Load excel once
df = pd.read_excel(DATASET_PATH)

excel_rows = []
for _, row in df.iterrows():
    excel_rows.append((
        str(row["OUTPASS_ID"]),
        row["NAME"]
    ))

@app.route("/logs")
def logs():
    today = request.args.get("date", date.today().strftime("%Y-%m-%d"))

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Fetch DB rows for that date
    c.execute("""
        SELECT Enrollment_No, Status, Action_Time
        FROM OutpassRequests2
        WHERE DATE(Action_Time)=?
        ORDER BY Action_Time
    """, (today,))
    db_rows = c.fetchall()
    conn.close()

    # Convert Excel to dict for easy lookup
    # Key = Enrollment_No, Value = Name
    excel_dict = {str(row["ENROLLMENT_NO"]).strip(): row["NAME"] for _, row in df.iterrows()}

    final_rows = []
    for enroll_no, status, action_time in db_rows:
        name = excel_dict.get(str(enroll_no).strip(), "Unknown")
        final_rows.append((enroll_no, name, status, action_time))

    return render_template_string(LOGS_PAGE, rows=final_rows, today=today)

LOGS_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
<title>Warden Logs</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
</head>

<body class="bg-light">
<div class="container py-4">

<h1 class="mb-4">Warden Logs - {{today}}</h1>

<form method="GET" action="/logs" class="mb-3">
    <input type="date" name="date" value="{{today}}" class="form-control mb-2">
    <button type="submit" class="btn btn-primary">Show</button>
</form>

<table class="table table-striped table-bordered">
<thead>
<tr>
    <th>Enrollment No</th>
    <th>Name</th>
    <th>Status</th>
    <th>Action Time</th>
</tr>
</thead>

<tbody>
{% for r in rows %}
<tr>
    <td>{{ r[0] }}</td>   <!-- Enrollment No -->
    <td>{{ r[1] }}</td>   <!-- Student Name -->
    <td>{{ r[2] }}</td>   <!-- Status -->
    <td>{{ r[3] }}</td>   <!-- Action Time -->
</tr>
{% endfor %}
</tbody>
</table>

<a href="/warden" class="btn btn-secondary">Back</a>

</div>
</body>
</html>"""



# -------------------------------
# WARDEN  Login Page
# -------------------------------
from flask import Flask, request, redirect

@app.route('/WARDEN_intermediate', methods=['GET', 'POST'])
def WARDEN_intermediate():
    HOD_USERNAME = 'SECURITY_GUARD'
    HOD_PASSWORD = 'GUARD123'
    error_msg = ""

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username == HOD_USERNAME and password == HOD_PASSWORD:
            return redirect('/request')
        else:
            error_msg = "Incorrect username or password!"

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>WARDEN Login</title>
        <style>
            body {{
                font-family: Arial;
                display:flex;
                justify-content:center;
                align-items:center;
                height:100vh;
                background:#f0f7ff;
            }}
            .container {{
                background:white;
                padding:30px;
                border-radius:15px;
                box-shadow:0 0 20px rgba(0,0,0,0.3);
                text-align:center;
                width:350px;
            }}
            input {{
                padding:10px;
                margin:5px 0;
                width:100%;
                border-radius:5px;
                border:1px solid #ccc;
            }}
            button {{
                padding:10px 20px;
                margin-top:10px;
                border:none;
                border-radius:5px;
                background:#007bff;
                color:white;
                font-weight:bold;
                cursor:pointer;
                width:100%;
            }}
            .error {{ color:red; margin-bottom:10px; }}
        </style>
        <script>
            function showAlert(msg) {{
                if(msg) {{
                    alert(msg);
                }}
            }}
        </script>
    </head>
    <body onload="showAlert('{error_msg}')">
        <div class="container">
            <h2>✅ Request Submitted Successfully</h2>
            <p>Enter WARDEN credentials to continue to REQUEST Dashboard:</p>

            <form method="POST">
                <input type="text" name="username" placeholder="GUARD Username" required>
                <input type="password" name="password" placeholder="GUARD Password" required>
                <button type="submit">📨 Login</button>
            </form>
        </div>
    </body>
    </html>
    """


# -------------------------------
# Route for Request Dashboard
# -------------------------------
@app.route('/request')
def request_dashboard():   # renamed to avoid conflict with Flask's 'request'
    return render_template('request.html')

# -------------------------------
# Route for Request Dashboard OUTGOING
# -------------------------------

from datetime import datetime
import sqlite3, os
from flask import request, render_template

DB_PATH = 'outpass.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS GuardActionsNew (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            enrollment_no TEXT,
            guard_action TEXT,
            action_time TEXT
        )
    """)
    
    conn.commit()
    conn.close()

init_db()


# ====================== UPDATE GUARD ACTION ======================
@app.route("/update_status", methods=["POST"])
def update_status():
    data = request.get_json()
    enrollment = data.get("enrollment")
    guard_action = data.get("status")   # Grant / Deny

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # ---- Get Warden Status ----
    cursor.execute("""
        SELECT Status FROM OutpassRequests2
        WHERE Enrollment_No = ?
        ORDER BY id DESC LIMIT 1
    """, (enrollment,))
    
    row = cursor.fetchone()
    warden_status = row[0] if row else "Unknown"

    # ---- Time ----
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ---- SAVE ONLY WHEN GUARD CLICKS BUTTON ----
    cursor.execute("""
        INSERT INTO GuardActionsNew (enrollment_no, guard_action, action_time)
        VALUES (?, ?, ?)
    """, (enrollment, guard_action, now))

    conn.commit()
    conn.close()

    return {
        "status": "success",
        "msg": "Guard action saved successfully",
        "warden_status": warden_status,
        "guard_action": guard_action,
        "time": now
    }

# ====================== WARDEN SUMMARY ======================
@app.route("/warden_summary")
def warden_summary():
    search_outpass = request.args.get('search_id', '').strip()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    enroll_list = []
    if search_outpass:
        matched = df[df['OUTPASS_ID'].astype(str) == str(search_outpass)]
        if not matched.empty:
            enroll_list = matched['ENROLLMENT_NO'].astype(str).tolist()

    if enroll_list:
        placeholders = ",".join("?" for _ in enroll_list)
        cursor.execute(f"""
            SELECT id, Enrollment_No, From_Date, To_Date, Out_Time, In_Time,
                   Reason, Hostel_Name, Address, Status
            FROM OutpassRequests2
            WHERE Enrollment_No IN ({placeholders})
        """, enroll_list)
        rows = cursor.fetchall()

    elif search_outpass:
        rows = []

    else:
        cursor.execute("""
            SELECT id, Enrollment_No, From_Date, To_Date, Out_Time, In_Time,
                   Reason, Hostel_Name, Address, Status
            FROM OutpassRequests2
        """)
        rows = cursor.fetchall()

    enriched = []
    img_folder = os.path.join("static", "IMAGE_DATASET COTY")

    for r in rows:
        req_id, enroll, from_date, to_date, out_time, in_time, reason, hostel_name, address, status = r

        # -------- NO GUARD ACTION FETCH (REMOVED DB DEPENDENCY) --------
        guard_status = "-"
        guard_time = "-"

        # ---------- Student Info ----------
        student = df[df['ENROLLMENT_NO'].astype(str) == str(enroll)]
        if not student.empty:
            rec = student.iloc[0]
            name = rec.get("NAME", "N/A")
            outpass_id = rec.get("OUTPASS_ID", "N/A")
            dept = rec.get("DEPARTMENT", "N/A")
            year = rec.get("YEAR", "N/A")
            acad_year = rec.get("YEAR.1", "N/A")
        else:
            name = outpass_id = dept = year = acad_year = "N/A"

        # ---------- Image ----------
        image_found = "/static/default.jpg"
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            p = os.path.join(img_folder, f"{enroll}{ext}")
            if os.path.exists(p):
                image_found = "/" + p.replace("\\", "/")
                break

        card_html = f"""
        <button onclick="toggleCard('card_{req_id}')" style="padding:6px 12px;border:none;border-radius:6px;background:#007bff;color:white;cursor:pointer;">
            View Info
        </button>

        <div id="card_{req_id}" style="display:none;margin-top:10px;width:280px;background:#f0f7ff;border-radius:15px;padding:15px;text-align:center;">
            <img src="{image_found}" style="width:90px;height:90px;border-radius:50%;object-fit:cover;"><br>
            <b>{name}</b><br>
            <small>Enrollment: {enroll}</small><br>
            <small>Outpass ID: {outpass_id}</small><br>
            <small>{dept} - {year}</small><br>
            <small>Academic Year: {acad_year}</small><hr>
            <small>From: {from_date}</small><br>
            <small>To: {to_date}</small><br>
            <small>Reason: {reason}</small><br>
            <small>Status: {status}</small>
        </div>
        """

        enriched.append({
            "enroll": enroll,
            "card_html": card_html,
            "status": status,
            "guard_status": guard_status,
            "guard_action_time": guard_time
        })

    conn.close()
    return render_template("warden_summary.html", enriched=enriched, request=request)

# -------------------------------
# Route for Request Dashboard INCOMING
# -------------------------------
# ================== IMPORTS ==================
from flask import Flask, request, redirect, render_template
from datetime import datetime
import sqlite3, os
import pandas as pd


DB_PATH = "outpass.db"

# Load student dataset
df = pd.read_excel("Dataset.csv.xlsx")

# ================== GUARD GRANTED PAGE ==================
@app.route("/guard_granted")
def guard_granted():
    import sqlite3, os

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 🔹 ALL outpass records (NO GROUP BY → duplicates allowed)
    cursor.execute("""
        SELECT id, Enrollment_No, From_Date, To_Date, Out_Time, In_Time,
               Reason, Hostel_Name, Address, Status
        FROM OutpassRequests2
        ORDER BY id DESC
    """)
    rows = cursor.fetchall()

    enriched_list = []
    img_folder = os.path.join("static", "IMAGE_DATASET COTY")

    for r in rows:
        req_id, enroll, from_date, to_date, out_time, in_time, reason, hostel, address, status = r

        # 🔹 Check if ANY Grant exists for this enrollment
        cursor.execute("""
            SELECT guard_action, action_time
            FROM GuardActionsNew
            WHERE enrollment_no = ?
              AND guard_action = 'Grant'
            ORDER BY id DESC
            LIMIT 1
        """, (str(enroll),))

        g = cursor.fetchone()
        if not g:
            continue   # ❌ no grant → don’t show

        guard_action, guard_time = g

        # 🔹 Student Info from dataframe
        student = df[df['ENROLLMENT_NO'].astype(str) == str(enroll)]
        if not student.empty:
            rec = student.iloc[0]
            name = rec.get("NAME", "N/A")
            outpass_id = rec.get("OUTPASS_ID", "N/A")
            dept = rec.get("DEPARTMENT", "N/A")
            year = rec.get("YEAR", "N/A")
            acad_year = rec.get("YEAR.1", "N/A")
        else:
            name = outpass_id = dept = year = acad_year = "N/A"

        # 🔹 Image detection
        image_found = "/static/default.jpg"
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            p = os.path.join(img_folder, f"{enroll}{ext}")
            if os.path.exists(p):
                image_found = "/" + p.replace("\\", "/")
                break

        enriched_list.append({
            "req_id": req_id,
            "enroll": enroll,
            "from_date": from_date,
            "to_date": to_date,
            "out_time": out_time,
            "in_time": in_time,
            "reason": reason,
            "hostel": hostel,
            "address": address,
            "status": status,
            "guard_action": guard_action,
            "guard_time": guard_time,
            "name": name,
            "outpass_id": outpass_id,
            "dept": dept,
            "year": year,
            "acad_year": acad_year,
            "image": image_found
        })

    conn.close()
    return render_template("guard_granted.html", records=enriched_list)


@app.route("/student_return", methods=["POST"])
def student_return():
    import sqlite3
    from datetime import datetime
    data = request.get_json()
    enroll = data.get("enrollment")

    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Fetch To_Date
    cursor.execute("""
        SELECT To_Date 
        FROM OutpassRequests2
        WHERE Enrollment_No = ?
          AND (return_status IS NULL OR return_status='Not Returned')
        ORDER BY id DESC LIMIT 1
    """, (enroll,))
    
    row = cursor.fetchone()

    return_status = "On Time"   # default

    if row:
        to_date_str = row[0]

        # Convert database value to date object
        try:
            to_date = datetime.strptime(to_date_str, "%Y-%m-%d")
        except:
            to_date = datetime.strptime(to_date_str, "%Y-%m-%d %H:%M:%S")

        # Compare actual return
        if now.date() > to_date.date():
            return_status = "Late Return"
        else:
            return_status = "On Time"

    # Update DB
    cursor.execute("""
        UPDATE OutpassRequests2
        SET return_date = ?, return_time = ?, return_status = ?
        WHERE Enrollment_No = ?
          AND (return_status IS NULL OR return_status='Not Returned')
    """, (current_date, current_time, return_status, enroll))

    conn.commit()
    conn.close()

    return {
        "success": True,
        "return_time": current_time,
        "return_status": return_status
    }
# -------------------------------
# Check Status (Student view)
# -------------------------------
@app.route('/check_status', methods=['GET', 'POST'])
def check_status():
    results = None
    enroll = request.args.get('enroll','')
    if request.method == 'POST':
        enroll = request.form['enroll'].strip()
    if enroll:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT From_Date, To_Date, Reason, Status FROM OutpassRequests2 WHERE Enrollment_No=?", (enroll,))
        rows = cursor.fetchall()
        conn.close()
        results = [{'from_date': r[0], 'to_date': r[1], 'reason': r[2], 'status': r[3]} for r in rows]

    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Check Outpass Status</title>
        <style>
            body { font-family: Arial; padding:20px; background:#f0f7ff; }
            h2 { text-align:center; }
            table { border-collapse: collapse; width:60%; margin:20px auto; }
            th, td { border:1px solid #ccc; padding:10px; text-align:center; }
            th { background:#007bff; color:white; }
            td.status-approved { color:green; font-weight:bold; }
            td.status-rejected { color:red; font-weight:bold; }
            td.status-pending { color:orange; font-weight:bold; }
            input, button { padding:10px; margin:5px; }
        </style>
    </head>
    <body>
        <h2>Check Your Outpass Status</h2>
        <form method="POST" style="text-align:center;">
            <input type="text" name="enroll" placeholder="Enter Enrollment No" value="{{ enroll }}" required>
            <button type="submit">Check Status</button>
        </form>
        {% if results %}
        <table>
            <tr><th>From</th><th>To</th><th>Reason</th><th>Status</th></tr>
            {% for r in results %}
            <tr>
                <td>{{ r.from_date }}</td>
                <td>{{ r.to_date }}</td>
                <td>{{ r.reason }}</td>
                <td class="status-{{ r.status|lower }}">{{ r.status }}</td>
            </tr>
            {% endfor %}
        </table>
        {% elif enroll %}<p style="text-align:center;">No outpass found for this enrollment number.</p>{% endif %}
    </body>
    </html>
    """
    return render_template_string(html, results=results, enroll=enroll)

#============================================================
# COLLEGE ENTRY EXIT SYSTEM (HIGH ACCURACY – FACE RECOGNITION)
# ============================================================

import threading
import time
import pandas as pd
import os
import base64
import cv2
import numpy as np
import sqlite3
import face_recognition
from flask import Flask, request, jsonify, render_template
from datetime import datetime

# ---------------- APP ----------------


# ---------------- PATHS ----------------
DATA_PATH = "Dataset.csv.xlsx"
IMAGE_DIR = "static/images"

STUDENT_IMAGE_DIR = "student_images"

# ---------------- DATABASE ----------------
def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS GateLogs(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        enrollment TEXT,
        name TEXT,
        department TEXT,
        year TEXT,
        phone TEXT,
        action TEXT,
        timestamp TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS Status(
        enrollment TEXT PRIMARY KEY,
        current_status TEXT,
        last_time TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS Students(
        enrollment TEXT PRIMARY KEY,
        name TEXT,
        department TEXT,
        year TEXT,
        phone TEXT
    )
    """)

    con.commit()
    con.close()

init_db()

# ---------------- DATASET ----------------
df = pd.read_excel(DATA_PATH)
df.columns = [c.strip().upper().replace(" ", "_") for c in df.columns]


# ---------------- FACE ENCODING ----------------
def get_face_encoding(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="hog")

    if len(boxes) != 1:
        return None

    return face_recognition.face_encodings(rgb, boxes)[0]


# ---------------- load new student faces----------------
def load_known_faces():
    global known_encodings, known_enrollments

    known_encodings = []
    known_enrollments = []

    print("🔄 Reloading student faces...")

    for _, row in df.iterrows():
        enroll = str(row["ENROLLMENT_NO"])

        for img_name in os.listdir(IMAGE_DIR):
            if img_name.startswith(enroll):
                img = cv2.imread(os.path.join(IMAGE_DIR, img_name))
                if img is None:
                    continue

                enc = get_face_encoding(img)
                if enc is not None:
                    known_encodings.append(enc)
                    known_enrollments.append(enroll)

    print(f"✅ Loaded {len(known_encodings)} face encodings")


# ---------------- LOAD STUDENT FACES ----------------
known_encodings = []
known_enrollments = []

load_known_faces()


# print("🔄 Loading student faces...")

# for _, row in df.iterrows():
#     enroll = str(row["ENROLLMENT_NO"])

#     for img_name in os.listdir(IMAGE_DIR):
#         if img_name.startswith(enroll):
#             img = cv2.imread(os.path.join(IMAGE_DIR, img_name))
#             if img is None:
#                 continue

#             enc = get_face_encoding(img)
#             if enc is not None:
#                 known_encodings.append(enc)
#                 known_enrollments.append(enroll)

# print(f"✅ Loaded {len(known_encodings)} face samples")


# ---------------- FACE MATCH ----------------
def match_face_live(live_encoding, tolerance=0.45):
    matches = face_recognition.compare_faces(
        known_encodings, live_encoding, tolerance
    )

    if True not in matches:
        return None

    distances = face_recognition.face_distance(
        known_encodings, live_encoding
    )

    return known_enrollments[np.argmin(distances)]

# ---------------- ROUTES ----------------
@app.route("/entryexit")
def entryexit():
    return render_template("verify.html")

@app.route("/verify1", methods=["POST"])
def verify1():

    if "frame" not in request.files:
        return render_template("error.html", msg="❌ No image received")

    file = request.files["frame"]
    if file.filename == "":
        return render_template("error.html", msg="❌ No image selected")

    img = cv2.imdecode(
        np.frombuffer(file.read(), np.uint8), 1
    )

    live_enc = get_face_encoding(img)
    if live_enc is None:
        return render_template(
            "error.html", msg="❌ Exactly ONE face required"
        )

    enroll = match_face_live(live_enc)
    if enroll is None:
        return render_template(
            "error.html", msg="❌ Student not registered"
        )

    s = df[df["ENROLLMENT_NO"].astype(str) == enroll].iloc[0]

    return render_template(
        "result.html",
        enroll=enroll,
        name=s.NAME,
        dept=s.DEPARTMENT,
        year=s.YEAR
    )

@app.route("/mark/<action>/<enroll>")
def mark(action, enroll):
    ts = datetime.now().strftime("%d-%m-%Y %I:%M %p")
    s = df[df["ENROLLMENT_NO"].astype(str) == enroll].iloc[0]

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute("""
    INSERT INTO GateLogs VALUES (NULL,?,?,?,?,?,?,?)
    """, (
        enroll,
        s.NAME,
        s.DEPARTMENT,
        s.YEAR,
        s.STUDENT_PHONE_NO,
        action,
        ts
    ))

    cur.execute("""
    REPLACE INTO Status VALUES (?,?,?)
    """, (
        enroll,
        "OUTSIDE" if action == "EXIT" else "INSIDE",
        ts
    ))

    con.commit()
    con.close()

    return render_template(
        "success2.html",
        action=action,
        name=s.NAME,
        time=ts
    )

@app.route("/add_student")
def add_student():
    return render_template("add_new_student.html")


@app.route("/dashboard")
def dashboard():

    # 🔹 1. calendar & search input घ्या
    selected_date = request.args.get("date")
    search = request.args.get("search", "").strip()

    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row

    # 🔹 2. base query
    query = """
    SELECT g.*, s.current_status
    FROM GateLogs g
    LEFT JOIN Status s ON g.enrollment = s.enrollment
    WHERE 1=1
    """
    params = []

    # 🔹 3. date filter (calendar)
    if selected_date:
        d = datetime.strptime(selected_date, "%Y-%m-%d").strftime("%d-%m-%Y")
        query += " AND g.timestamp LIKE ?"
        params.append(f"{d}%")

    # 🔹 4. student search (name / enrollment / phone)
    if search:
        query += """
        AND (
            g.enrollment LIKE ?
            OR g.name LIKE ?
            OR g.phone LIKE ?
        )
        """
        params.extend([
            f"%{search}%",
            f"%{search}%",
            f"%{search}%"
        ])

    query += " ORDER BY g.id DESC"

    logs = con.execute(query, params).fetchall()
    con.close()

    return render_template(
        "DASHBOARD1.html",
        logs=logs,
        selected_date=selected_date
    )

@app.route('/student/<enroll>')
def student_info(enroll):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT name, department, year FROM Students WHERE enrollment=?",
        (enroll,)
    )
    student = cursor.fetchone()

    conn.close()

    if not student:
        return "Student not found"

    name, dept, year = student

    # student photos path
    image_folder = f"static/dataset/{enroll}"
    images = []

    if os.path.exists(image_folder):
        images = os.listdir(image_folder)

    return render_template(
        "student_info.html",
        name=name,
        department=dept,
        year=year,
        images=images,
        enroll=enroll
    )
@app.route("/save_student_camera", methods=["POST"])
def save_student_camera():
    try:
        os.makedirs(STUDENT_IMAGE_DIR, exist_ok=True)

        data = request.json

        enroll = data.get("enroll")
        name = data.get("name")
        dept = data.get("dept")
        year = data.get("year")
        phone = data.get("phone")
        images = data.get("images")

        if not all([enroll, name, dept, year, phone]):
            return jsonify({"status":"error","msg":"All fields required"})

        if not images or len(images) < 4:
            return jsonify({"status":"error","msg":"4 images required"})

        student_folder = os.path.join(STUDENT_IMAGE_DIR, enroll)
        os.makedirs(student_folder, exist_ok=True)

        encodings_list = []

        for i, img_data in enumerate(images):
            header, encoded = img_data.split(",", 1)
            img_bytes = base64.b64decode(encoded)

            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is None:
                continue

            img_path = os.path.join(student_folder, f"{enroll}_{i+1}.jpg")
            cv2.imwrite(img_path, img)

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            enc = face_recognition.face_encodings(rgb)

            if enc:
                encodings_list.append(enc[0])
            else:
                print("❌ No face detected in one image")

        if len(encodings_list) < 2:
            return jsonify({
                "status": "error",
                "msg": "Face not clear. Please capture again"
            })

        avg_encoding = np.mean(encodings_list, axis=0)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
        INSERT OR REPLACE INTO Students
        (enrollment, name, department, year, phone)
        VALUES (?, ?, ?, ?, ?)
        """, (enroll, name, dept, year, phone))

        conn.commit()
        conn.close()

        np.save(
            os.path.join(student_folder, "encoding.npy"),
            avg_encoding
        )

        load_known_faces()

        return jsonify({
            "status":"success",
            "msg":"✅ Student saved successfully"
        })

    except Exception as e:
        print("🔥 SAVE STUDENT ERROR:", e)
        return jsonify({
            "status":"error",
            "msg":"Server error while saving student"
        })

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)