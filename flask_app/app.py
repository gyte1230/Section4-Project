from flask import Flask, render_template, request
import pandas as pd
import joblib
import sqlite3
import subprocess

db_path = 'Food.db'

conn = sqlite3.connect(db_path)
query = 'SELECT * FROM nutrition_facts'
food = pd.read_sql_query(query, conn)
conn.close()

app = Flask(__name__)

# 모델 불러오기
model = joblib.load('knn_model.pkl')

# 홈 페이지
@app.route('/')
def home():
    return render_template('index.html')

# 결과 예측 처리
@app.route('/recommend', methods=['POST'])
def predict():
    # 입력 값 받아오기
    data = request.form.to_dict()
    carb = float(data['carb'])
    protein = float(data['protein'])
    fat = float(data['fat'])

    # 입력 값으로 예측
    input_data = [[carb, protein, fat]]
    distances, indices = model.kneighbors(input_data)

    # 가장 근사한 값으로 예측된 음식 10가지 추출
    top_10_indices = indices.flatten()  # 인덱스를 1차원으로 변환
    top_10_foods = food.iloc[top_10_indices, :] 

    # 결과 반환
    return render_template('result.html', result=top_10_foods)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    # Flask 애플리케이션 실행 전에 ngrok을 실행하여 포트 포워딩
    ngrok_process = subprocess.Popen(['ngrok', 'http', '5000'])  # 포트 번호를 Flask 애플리케이션의 포트에 맞게 변경
    
    # Flask 애플리케이션 실행
    app.run(debug=True)

    # Flask 애플리케이션 실행 후에 ngrok 프로세스 종료
    ngrok_process.terminate()