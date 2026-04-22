from flask import Flask, render_template, request, make_response, redirect, url_for
import joblib
import pandas as pd
import numpy as np
from fpdf import FPDF
import sqlite3
import base64
import os

app = Flask(__name__)

# --- 1. Model Loading ---
try:
    # Ensure train_model.py has been run to create this file with 11 features
    model = joblib.load('heart_model.pkl')
    print("Model loaded successfully with 11 parameters!")
except Exception as e:
    print(f"Error: heart_model.pkl nahi mila! Pehle train_model.py chalayein. {e}")


# --- 2. Database Initialization ---
def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    # Table schema with all 11 clinical parameters
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            age INTEGER, 
            sex TEXT, 
            cp INTEGER, 
            bp INTEGER, 
            chol INTEGER,
            fbs INTEGER, 
            restecg INTEGER, 
            thalach INTEGER, 
            exang INTEGER,
            oldpeak REAL, 
            slope INTEGER, 
            score REAL, 
            status TEXT, 
            date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


init_db()


# --- 3. Routes ---

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/predictor', methods=['GET', 'POST'])
def predictor():
    result_data = None
    user_chart_data = []
    # Normal Reference Values for the 5 parameters shown in chart
    ref_chart_data = [50, 120, 200, 150, 1.0]

    if request.method == 'POST':
        try:
            # Form se saare 11 inputs nikaalein
            age = int(request.form.get('age', 0))
            sex = int(request.form.get('sex', 0))
            cp = int(request.form.get('cp', 0))
            trestbps = int(request.form.get('trestbps', 0))
            chol = int(request.form.get('chol', 0))
            fbs = int(request.form.get('fbs', 0))
            restecg = int(request.form.get('restecg', 0))
            thalach = int(request.form.get('thalach', 0))
            exang = int(request.form.get('exang', 0))
            oldpeak = float(request.form.get('oldpeak', 0.0))
            slope = int(request.form.get('slope', 0))

            # AI Model Prediction (All 11 features)
            input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope]])
            prob = model.predict_proba(input_data)[0][1] * 100
            score = round(prob, 2)

            # Risk Categorization
            if score < 30:
                status, css_class = "Low Risk ✅", "low-risk"
                advice = "Your heart health appears stable. Maintain a balanced diet."
            elif 30 <= score < 70:
                status, css_class = "Moderate Risk ⚠️", "moderate-risk"
                advice = "Precautionary measures needed. Monitor your lifestyle."
            else:
                status, css_class = "High Risk 🚨", "high-risk"
                advice = "Immediate medical consultation recommended."

            result_data = {"score": score, "status": status, "css_class": css_class, "advice": advice}

            # Charts ke liye sirf top 5 parameters (Age, BP, Chol, HR, Oldpeak)
            user_chart_data = [age, trestbps, chol, thalach, oldpeak]

            # Database mein Save karein
            conn = sqlite3.connect('database.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO history (age, sex, cp, bp, chol, fbs, restecg, thalach, exang, oldpeak, slope, score, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (age, 'Male' if sex == 1 else 'Female', cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak,
                  slope, score, status))
            conn.commit()
            conn.close()

        except Exception as e:
            print(f"Prediction Error: {e}")

    return render_template('predictor.html', result=result_data, user_data=user_chart_data, ref_data=ref_chart_data)

@app.route('/predict_bulk', methods=['POST'])
def predict_bulk():
    if 'file' not in request.files:
        return "No file uploaded"
    file = request.files['file']
    if file.filename == '':
        return "No file selected"

    if file and file.filename.endswith('.csv'):
        try:
            df = pd.read_csv(file)

            df.insert(0, 'id', range(1, len(df) + 1))

            # --- 1. SMART CLEANING ---
            df.columns = df.columns.str.strip().str.lower()

            # Mapping Dictionaries (UCI & Clinical Standard)
            sex_map = {'male': 1, 'm': 1, '1': 1, 'female': 0, 'f': 0, '0': 0}
            cp_map = {'ta': 0, '0': 0, 'ata': 1, '1': 1, 'nap': 2, '2': 2, 'asy': 3, '3': 3}
            ecg_map = {'normal': 0, '0': 0, 'st': 1, '1': 1, 'lvh': 2, '2': 2}
            angina_map = {'y': 1, 'yes': 1, '1': 1, 'n': 0, 'no': 0, '0': 0}
            slope_map = {'up': 0, '0': 0, 'flat': 1, '1': 1, 'down': 2, '2': 2, '3': 2}

            # --- 2. TRANSFORMATION ---
            df['sex_n'] = df['sex'].astype(str).str.lower().map(sex_map)
            df['cp_n'] = df['chestpaintype'].astype(str).str.lower().map(cp_map)
            df['ecg_n'] = df['restingecg'].astype(str).str.lower().map(ecg_map)
            df['angina_n'] = df['exerciseangina'].astype(str).str.lower().map(angina_map)
            df['slope_n'] = df['st_slope'].astype(str).str.lower().map(slope_map)

            # --- 3. MODEL PREDICTION ---
            input_cols = {
                'age': 'age', 'sex_n': 'sex', 'cp_n': 'cp', 
                'restingbp': 'trestbps', 'cholesterol': 'chol', 'fastingbs': 'fbs',
                'ecg_n': 'restecg', 'maxhr': 'thalach', 'angina_n': 'exang',
                'oldpeak': 'oldpeak', 'slope_n': 'slope'
            }
            
            X = df[list(input_cols.keys())].rename(columns=input_cols)
            
            # Predict Risk Scores
            probabilities = model.predict_proba(X)[:, 1] * 100
            df['risk_score'] = probabilities.round(2)
            df['result'] = ['High Risk' if p >= 50 else 'Low Risk' for p in probabilities]

            # --- 4. PREPARE DISPLAY ---
            # display_df banate waqt sirf zaroori columns select karein
            display_cols = ['id','age', 'sex', 'chestpaintype', 'restingbp', 'cholesterol', 'risk_score', 'result']
            existing_cols = [c for c in display_cols if c in df.columns]
            display_df = df[existing_cols]
        
            result_path = "static/bulk_results.csv"
            df.to_csv(result_path, index=False)

            return render_template('bulk_result.html', 
                                 tables=[display_df.to_html(classes='data premium-table', index=False)], 
                                 file_path=result_path)

        except KeyError as e:
            return f"Mapping Error: Column {str(e)} nahi mila. CSV headers check karein."
        except Exception as e:
            return f"General Error: {str(e)}"

    return "Invalid format! Sirf .csv file allowed hai."

@app.route('/history')
def history():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM history ORDER BY date DESC')
    rows = cursor.fetchall()
    conn.close()
    return render_template('history.html', history=rows)


@app.route('/details/<int:id>')
def details(id):
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM history WHERE id = ?', (id,))
    row = cursor.fetchone()
    conn.close()
    return render_template('details.html', data=row) if row else "Record not found!"


@app.route('/delete/<int:id>')
def delete_history(id):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM history WHERE id = ?', (id,))
    conn.commit()
    conn.close()
    return redirect(url_for('history'))


import base64
from io import BytesIO


@app.route('/download_report', methods=['POST'])
def download_report():
    try:
        f = request.form
        pdf = FPDF()
        pdf.add_page()

        # Auto Page Break abhi bhi band rakhenge taki 1 page par rahe
        pdf.set_auto_page_break(auto=False, margin=0)

        # --- 1. Header (Compact: 22mm) ---
        pdf.set_fill_color(44, 62, 80)
        pdf.rect(0, 0, 210, 22, 'F')
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", 'B', 18)
        pdf.cell(190, 8, txt="MEDVISION HEALTH REPORT", ln=True, align='C')
        pdf.set_font("Arial", size=9)
        pdf.cell(190, 4, txt="AI-Powered Cardiovascular Assessment", ln=True, align='C')
        pdf.ln(10)

        # --- 2. Risk Summary (Height: 10mm) ---
        score = float(f.get('score', 0))
        status = f.get('status', 'Unknown').replace('✅', '').replace('🚨', '').replace('⚠️', '')

        if score < 30:
            pdf.set_fill_color(39, 174, 96)
        elif score < 70:
            pdf.set_fill_color(241, 196, 15)
        else:
            pdf.set_fill_color(231, 76, 60)

        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", 'B', 13)
        pdf.cell(190, 10, txt=f"RISK ASSESSMENT: {status} ({score}%)", ln=True, align='C', fill=True)
        pdf.ln(6)  # Spacing badhayi

        # --- 3. Clinical Data Table (Bada Font aur Height) ---
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", 'B', 12)  # Font 9 se 12 kiya
        pdf.set_fill_color(242, 242, 242)
        pdf.cell(190, 10, txt=" Patient Clinical Information", ln=True, fill=True)
        pdf.set_font("Arial", size=10)  # Data font 8 se 10 kiya

        # Sabhi 11 parameters ki list (6 rows x 2 columns)
        data = [
            ["Age", f.get('age'), "Sex", "Male" if f.get('sex') == '1' else "Female"],
            ["Chest Pain", f.get('cp'), "Resting BP", f"{f.get('trestbps')} mmHg"],
            ["Cholesterol", f"{f.get('chol')} mg/dl", "Fasting Sugar", "High" if f.get('fbs') == '1' else "Normal"],
            ["Resting ECG", f.get('restecg'), "Max Heart Rate", f.get('thalach')],
            ["Exercise Angina", "Yes" if f.get('exang') == '1' else "No", "Oldpeak", f.get('oldpeak')],
            ["ST Slope", "Up" if f.get('slope') == '0' else "Flat" if f.get('slope') == '1' else "Down", "", ""]
        ]

        # Table Draw Karein (Spacious Row Height: 10mm)
        for row in data:
            pdf.cell(45, 10, txt=f" {row[0]}:", border='LTB')
            pdf.cell(50, 10, txt=str(row[1]), border='RTB')
            if row[2]:  # Agar doosre column mein data hai
                pdf.cell(45, 10, txt=f" {row[2]}:", border='LTB')
                pdf.cell(50, 10, txt=str(row[3]), border='RTB', ln=True)
            else:  # Last row mein khali cell ke liye
                pdf.cell(95, 10, txt="", border='RTB', ln=True)

        # --- 4. Gap aur Diagrams (Table se door) ---
        pdf.ln(12)  # Gap 3mm se badha kar 12mm kiya taki diagram table se na chipe
        curr_y = pdf.get_y()

        bar_img_data = f.get('bar_chart_img')
        if bar_img_data:
            img_data = base64.b64decode(bar_img_data.split(',')[1])
            with open("temp_chart.png", "wb") as fh: fh.write(img_data)
            pdf.image("temp_chart.png", x=15, y=curr_y, w=80)

        radar_img_data = f.get('radar_chart_img')
        if radar_img_data:
            img_data_r = base64.b64decode(radar_img_data.split(',')[1])
            with open("temp_radar.png", "wb") as fh: fh.write(img_data_r)
            pdf.image("temp_radar.png", x=115, y=curr_y, w=80)

        # --- 5. Footer ---
        pdf.set_y(278)
        pdf.set_font("Arial", 'I', 7)
        pdf.set_text_color(160, 160, 160)
        pdf.cell(190, 5, txt="Note: This is an AI-generated report. Please consult a doctor for medical advice.",
                 ln=True, align='C')

        response = make_response(pdf.output(dest='S').encode('latin-1', 'ignore'))
        response.headers.set('Content-Disposition', 'attachment', filename='Medvision_Report.pdf')
        response.headers.set('Content-Type', 'application/pdf')
        return response
    except Exception as e:
        return f"Layout Error: {str(e)}"


if __name__ == '__main__':
    # Cloud platforms port ko environment variable se uthate hain
    port = int(os.environ.get("PORT", 8080)) 
    app.run(host='0.0.0.0', port=port)
