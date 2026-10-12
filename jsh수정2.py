import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap  # XAI(설명 가능한 AI)를 위한 라이브러리
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 한글 폰트 깨짐 방지 (윈도우 맑은 고딕)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 화면 넓게 쓰기
st.set_page_config(layout="wide", page_title="반도체 식각 공정 AI 모니터링")

# --- [체크리스트 반영] 사이드바 면책 조항 (실제 계측 완전 대체 불가 경고) ---
st.sidebar.markdown("## 📌 시스템 안내")
st.sidebar.info("본 대시보드는 식각 공정 최적화를 위한 AI 보조 시뮬레이션 툴입니다.")
st.sidebar.warning("⚠️ **면책 조항**: 본 AI 예측 및 분석 결과는 참고용이며, **실제 팹(Fab) 계측 및 현직 엔지니어의 최종 판단을 완전히 대체하지 않습니다.**")

# 1. 학습된 AI 모델 불러오기
@st.cache_resource
def load_models():
    rf = joblib.load('etch_depth_predictor.pkl')
    iso = joblib.load('anomaly_detector.pkl')
    return rf, iso

rf_model, iso_model = load_models()

# 2. 웹 툴 제목 및 설명
st.title("반도체 식각 공정 AI 모니터링 툴 🔬 (최종 검증 버전)")
st.write("공정 센서 데이터(Excel)를 업로드하면 예상 Etch Depth와 이상 점수, 그리고 주요 영향 센서를 정밀 분석합니다.")

# 3. 데이터 업로드 UI
uploaded_file = st.file_uploader("새로운 공정 데이터 엑셀 파일을 업로드하세요.", type=['xlsx'])

if uploaded_file is not None:
    uploaded_file.seek(0)
    df = pd.read_excel(uploaded_file, sheet_name=0, header=3, engine='openpyxl')

    st.write("입력된 센서 데이터 미리보기:")
    st.dataframe(df.head())

    if st.button("🚀 AI 정밀 분석 시작"):
        with st.spinner('AI가 데이터를 다각도로 분석 중입니다...'):
            # 4. 전처리 및 필수 컬럼 검사
            cols_to_drop = ['날짜', 'Wafer', 'Lot', 'Y_Si_Etch_89pt_um', 'Y_Si_Uniformity_89pt_pct', 'REF_Si_Etch_9pt_um', 'REF_Si_Uniformity_9pt_pct']
            X_raw = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
            X = pd.get_dummies(X_raw, columns=['공정_Type'], drop_first=True)
            X = X.fillna(X.median())
            
            # 모델 학습 피처와 업로드 데이터 피처 강제 일치화 및 누락 컬럼 친절한 경고 안내
            expected_features = rf_model.feature_names_in_
            missing_cols = [col for col in expected_features if col not in X.columns]
            
            if missing_cols:
                st.warning(f"⚠️ **데이터 검사 알림**: 업로드된 파일에 모델이 요구하는 필수 센서 컬럼 중 일부({len(missing_cols)}개)가 누락되어 기본값(0)으로 안전하게 대체되었습니다.")
            
            X = X.reindex(columns=expected_features, fill_value=0)

            # 5. AI 추론 (Depth 예측 및 이상 점수 산출)
            depth_predictions = rf_model.predict(X)
            anomaly_scores = iso_model.decision_function(X)
            
            def get_status(score):
                if score < -0.05:
                    return '🔴 위험 (설비 점검 요망)'
                elif score < 0.05:
                    return '🟡 주의 (트렌드 이탈 의심)'
                else:
                    return '🟢 정상'

            # 6. 결과 정리
            result_df = df[['날짜', 'Wafer', 'Lot', '공정_Type']].copy()
            result_df['예측 Si Etch Depth (um)'] = np.round(depth_predictions, 4)
            result_df['이상 점수 (Score)'] = np.round(anomaly_scores, 3)
            result_df['상태'] = [get_status(s) for s in anomaly_scores]

            # --- [체크리스트 반영] MAE, RMSE, R² 정량적 성능 지표 표시 (실제값이 파일에 있는 경우) ---
            target_col = 'Y_Si_Etch_89pt_um'
            if target_col in df.columns:
                y_true = df[target_col]
                common_idx = y_true.dropna().index.intersection(X.index)
                if len(common_idx) > 0:
                    y_t = y_true.loc[common_idx]
                    y_p = depth_predictions[common_idx]
                    mae = mean_absolute_error(y_t, y_p)
                    rmse = np.sqrt(mean_squared_error(y_t, y_p))
                    r2 = r2_score(y_t, y_p)

                    st.divider()
                    st.subheader("🎯 모델 정량적 성능 평가 지표 (Actual vs Predicted)")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("MAE (평균 절대 오차)", f"{mae:.4f} um")
                    m2.metric("RMSE (평균 제곱근 오차)", f"{rmse:.4f} um")
                    m3.metric("R² Score (결정 계수)", f"{r2:.4f}")

            st.divider()
            st.subheader("📊 1단계: AI 기본 분석 결과 (스코어링 도입)")
            st.dataframe(result_df, use_container_width=True)

            # --- 고도화 기능 파트 ---
            col1, col2 = st.columns(2)

            # --- [체크리스트 반영] "원인" 단정 방지 및 SHAP 기여도 분석으로 타이틀 개선 ---
            with col1:
                st.subheader("🔍 2단계: 이상 징후 주요 영향 센서 분석 (SHAP 기여도)")
                danger_idx = result_df[result_df['상태'].str.contains('위험|주의')].index
                
                if len(danger_idx) > 0:
                    st.error(f"⚠️ {len(danger_idx)}개의 공정에서 이상 패턴이 감지되었습니다. 아래에서 상세 기여도를 확인하세요.")
                    target_row = st.selectbox("분석할 이상 공정(행 번호)을 선택하세요:", danger_idx)
                    
                    explainer = shap.Explainer(rf_model)
                    shap_values = explainer(X.iloc[[target_row]])
                    
                    fig, ax = plt.subplots(figsize=(6, 4))
                    shap.plots.waterfall(shap_values[0], show=False)
                    st.pyplot(fig)
                    st.caption("그래프 해석: 붉은색 막대는 예상 Depth를 높이는 데 기여한 센서값, 파란색은 낮추는 데 기여한 센서값입니다. (통계적 기여도)")
                else:
                    st.success("✅ 감지된 이상 징후가 없어 추가 기여도 분석이 필요하지 않습니다.")

            with col2:
                st.subheader("📈 3단계: 공정 변동성 모니터링")
                st.info("💡 향후 모델 재학습 시, 아래의 분산/표준편차 등 변동성 데이터를 피처에 추가하면 예측력이 극대화됩니다.")
                
                sensor_cols = [c for c in X.columns if 'mean' in c or 'Temp' in c or 'Flow' in c][:5]
                if sensor_cols:
                    st.line_chart(X[sensor_cols])
                    st.caption("위 차트는 현재 업로드된 배치의 주요 센서 흐름입니다. 값이 튀는(Spike) 구간이 이상 징후와 일치하는지 확인하세요.")