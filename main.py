# -*- coding: utf-8 -*-
"""
나이브 베이즈 기반 확률을 이용한 분류
제작 : 윤진석
Streamlit 단계별 진행 앱
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ----------------------------------------------------------------------
# 페이지 기본 설정
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="나이브 베이즈 분류 학습 앱",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 나이브 베이즈 기반 확률을 이용한 분류")
st.caption("제작 : 윤진석")

# ----------------------------------------------------------------------
# 세션 상태 초기화
# ----------------------------------------------------------------------
if "step" not in st.session_state:
    st.session_state.step = 1

def go_next():
    st.session_state.step += 1

def reset_all():
    keys = list(st.session_state.keys())
    for k in keys:
        del st.session_state[k]
    st.session_state.step = 1

# ----------------------------------------------------------------------
# 사이드바 : 진행 상황 + 리셋
# ----------------------------------------------------------------------
STEPS = [
    "1단계: 데이터 업로드",
    "2단계: 데이터 정보 확인 및 결측치 제거",
    "3단계: 문제(X)와 정답(y) 분리",
    "4단계: 원-핫 인코딩",
    "5단계: 학습/테스트 데이터 분할",
    "6단계: 나이브 베이즈 모델 학습",
    "7단계: 예측 및 평가",
]

with st.sidebar:
    st.header("📚 진행 상황")
    for i, name in enumerate(STEPS, start=1):
        if i < st.session_state.step:
            st.markdown(f"✅ {name}")
        elif i == st.session_state.step:
            st.markdown(f"▶ **{name}**")
        else:
            st.markdown(f"⬜ {name}")
    st.divider()
    if st.button("🔄 처음부터 다시 시작"):
        reset_all()
        st.rerun()

# ======================================================================
# 1단계 : 데이터 업로드
# ======================================================================
st.header("1단계 · 데이터 업로드")
st.write("CSV 파일을 업로드하세요. (예: Train.csv)")

uploaded = st.file_uploader("CSV 파일 선택", type=["csv"], key="uploader")

if uploaded is not None and "df" not in st.session_state:
    try:
        df = pd.read_csv(uploaded)
        st.session_state.df = df
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")

if "df" in st.session_state:
    df = st.session_state.df
    st.success(f"업로드 완료! 데이터 크기: {df.shape[0]} 행 × {df.shape[1]} 열")
    st.dataframe(df.head(20), use_container_width=True)

    if st.session_state.step == 1:
        if st.button("➡ 2단계로 진행", type="primary"):
            go_next()
            st.rerun()

# ======================================================================
# 2단계 : 데이터 구성 정보 확인 및 결측치 제거
# ======================================================================
if st.session_state.step >= 2:
    st.header("2단계 · 데이터 구성 정보 확인 및 결측치 제거")

    df = st.session_state.df

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("데이터 정보 (info)")
        info_df = pd.DataFrame({
            "컬럼명": df.columns,
            "비결측 개수": df.notna().sum().values,
            "결측치 개수": df.isna().sum().values,
            "데이터 타입": df.dtypes.astype(str).values,
        })
        st.dataframe(info_df, use_container_width=True)

    with col2:
        st.subheader("기본 통계량")
        st.dataframe(df.describe(include="all").T, use_container_width=True)

    st.markdown(f"- 원본 데이터: **{df.shape[0]} 행 × {df.shape[1]} 열**")
    st.markdown(f"- 전체 결측치 수: **{int(df.isna().sum().sum())} 개**")

    if "df_clean" not in st.session_state:
        if st.button("🧹 결측치 제거 (dropna) 실행"):
            st.session_state.df_clean = df.dropna().reset_index(drop=True)
            st.rerun()
    else:
        df_clean = st.session_state.df_clean
        st.success(f"결측치 제거 후: {df_clean.shape[0]} 행 × {df_clean.shape[1]} 열")
        st.dataframe(df_clean.head(20), use_container_width=True)

        if st.session_state.step == 2:
            if st.button("➡ 3단계로 진행", type="primary"):
                go_next()
                st.rerun()

# ======================================================================
# 3단계 : 문제(X) / 정답(y) 분리
# ======================================================================
if st.session_state.step >= 3:
    st.header("3단계 · 문제(X)와 정답(y) 분리")

    df_clean = st.session_state.df_clean
    columns = list(df_clean.columns)

    st.write("문제(특성) 열과 정답(레이블) 열을 선택하세요.")

    # 기본값 : 원본 코드와 동일하게 1~9번 열을 X, 10번 열을 y 로 설정
    default_X = columns[1:10] if len(columns) >= 11 else columns[:-1]
    default_y = columns[10] if len(columns) >= 11 else columns[-1]

    feature_cols = st.multiselect(
        "문제(X) 열 선택",
        options=columns,
        default=default_X,
        key="feature_cols",
    )
    label_col = st.selectbox(
        "정답(y) 열 선택",
        options=columns,
        index=columns.index(default_y) if default_y in columns else 0,
        key="label_col",
    )

    if label_col in feature_cols:
        st.warning("정답 열은 문제 열에 포함될 수 없습니다. 문제 열에서 제외해주세요.")
    elif len(feature_cols) == 0:
        st.warning("최소 한 개 이상의 문제 열을 선택해주세요.")
    else:
        X = df_clean[feature_cols]
        y = df_clean[label_col]

        st.session_state.X = X
        st.session_state.y = y

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**X 형태:** {X.shape}")
            st.dataframe(X.head(10), use_container_width=True)
        with c2:
            st.markdown(f"**y 형태:** {y.shape}")
            st.dataframe(y.head(10).to_frame(), use_container_width=True)
            st.markdown("**정답(y) 클래스 분포**")
            st.dataframe(
                y.value_counts().rename_axis(label_col).reset_index(name="개수"),
                use_container_width=True,
            )

        if st.session_state.step == 3:
            if st.button("➡ 4단계로 진행", type="primary"):
                go_next()
                st.rerun()

# ======================================================================
# 4단계 : 원-핫 인코딩
# ======================================================================
if st.session_state.step >= 4:
    st.header("4단계 · 원-핫 인코딩 (get_dummies)")

    X = st.session_state.X
    X_encoded = pd.get_dummies(X)
    st.session_state.X_encoded = X_encoded

    st.markdown(
        f"- 인코딩 전: **{X.shape[1]} 개 열**\n"
        f"- 인코딩 후: **{X_encoded.shape[1]} 개 열**"
    )
    st.dataframe(X_encoded.head(20), use_container_width=True)

    if st.session_state.step == 4:
        if st.button("➡ 5단계로 진행", type="primary"):
            go_next()
            st.rerun()

# ======================================================================
# 5단계 : 학습/테스트 데이터 분할
# ======================================================================
if st.session_state.step >= 5:
    st.header("5단계 · 학습 데이터와 테스트 데이터 분할")

    test_size = st.slider("테스트 데이터 비율 (test_size)", 0.1, 0.5, 0.3, 0.05)
    random_state = st.number_input("random_state", value=42, step=1)

    if st.button("✂ 데이터 분할 실행"):
        X_encoded = st.session_state.X_encoded
        y = st.session_state.y
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y,
            test_size=test_size,
            random_state=int(random_state),
        )
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test

    if "X_train" in st.session_state:
        st.success("데이터 분할 완료!")
        st.markdown(
            f"- X_train: **{st.session_state.X_train.shape}**, "
            f"X_test: **{st.session_state.X_test.shape}**\n"
            f"- y_train: **{st.session_state.y_train.shape}**, "
            f"y_test: **{st.session_state.y_test.shape}**"
        )

        if st.session_state.step == 5:
            if st.button("➡ 6단계로 진행", type="primary"):
                go_next()
                st.rerun()

# ======================================================================
# 6단계 : 모델 학습
# ======================================================================
if st.session_state.step >= 6:
    st.header("6단계 · 나이브 베이즈 모델 학습 (GaussianNB)")

    if st.button("🧠 모델 학습 시작"):
        model = GaussianNB()
        model.fit(st.session_state.X_train, st.session_state.y_train)
        st.session_state.model = model

    if "model" in st.session_state:
        st.success("모델 학습 완료!")
        model = st.session_state.model
        st.markdown(f"- 학습된 클래스: `{list(model.classes_)}`")
        st.markdown(f"- 클래스별 사전확률(prior): `{np.round(model.class_prior_, 4).tolist()}`")

        if st.session_state.step == 6:
            if st.button("➡ 7단계로 진행", type="primary"):
                go_next()
                st.rerun()

# ======================================================================
# 7단계 : 예측 및 평가
# ======================================================================
if st.session_state.step >= 7:
    st.header("7단계 · 예측 및 평가")

    model = st.session_state.model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    st.metric("정확도 (Accuracy)", f"{accuracy:.4f}")

    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("혼동 행렬 (Confusion Matrix)")
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=model.classes_,
            yticklabels=model.classes_,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with c2:
        st.subheader("분류 리포트 (Classification Report)")
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

    st.subheader("예측 결과 미리보기")
    result_df = pd.DataFrame({
        "실제값(y_test)": np.array(y_test),
        "예측값(y_pred)": y_pred,
        "정답 여부": np.where(np.array(y_test) == y_pred, "⭕", "❌"),
    })
    st.dataframe(result_df.head(50), use_container_width=True)

    csv = result_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 예측 결과 CSV 다운로드",
        data=csv,
        file_name="prediction_result.csv",
        mime="text/csv",
    )

    st.balloons()
    st.success("🎉 모든 단계가 완료되었습니다!")
