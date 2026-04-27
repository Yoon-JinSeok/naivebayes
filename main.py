# -*- coding: utf-8 -*-
"""
나이브 베이즈 기반 확률을 이용한 분류
제작 : 윤진석
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ─────────────────────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="나이브 베이즈 분류기",
    page_icon="🧮",
    layout="wide",
)

st.title("🧮 나이브 베이즈 기반 확률을 이용한 분류")
st.caption("제작 : 윤진석")

# ─────────────────────────────────────────────────────────────
# 세션 상태 초기화
# ─────────────────────────────────────────────────────────────
if "step" not in st.session_state:
    st.session_state.step = 1
if "df" not in st.session_state:
    st.session_state.df = None
if "df_clean" not in st.session_state:
    st.session_state.df_clean = None
if "X" not in st.session_state:
    st.session_state.X = None
if "y" not in st.session_state:
    st.session_state.y = None
if "X_encoded" not in st.session_state:
    st.session_state.X_encoded = None
if "split" not in st.session_state:
    st.session_state.split = None
if "model" not in st.session_state:
    st.session_state.model = None
if "y_pred" not in st.session_state:
    st.session_state.y_pred = None


def go_next():
    st.session_state.step += 1


def reset_all():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.session_state.step = 1


# ─────────────────────────────────────────────────────────────
# 사이드바: 진행 상태
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📍 진행 단계")
    steps_label = [
        "1단계 · 데이터 업로드",
        "2단계 · 정보 확인 / 결측치 제거",
        "3단계 · 문제(X)·정답(y) 분리",
        "4단계 · 원-핫 인코딩",
        "5단계 · 학습/테스트 분할",
        "6단계 · 모델 학습",
        "7단계 · 예측 및 평가",
    ]
    for i, label in enumerate(steps_label, start=1):
        if i < st.session_state.step:
            st.markdown(f"✅ {label}")
        elif i == st.session_state.step:
            st.markdown(f"**▶ {label}**")
        else:
            st.markdown(f"⬜ {label}")

    st.divider()
    if st.button("🔄 처음부터 다시 시작", use_container_width=True):
        reset_all()
        st.rerun()


# ─────────────────────────────────────────────────────────────
# 파일 읽기 유틸리티 (CSV / Excel + 다양한 인코딩 지원)
# ─────────────────────────────────────────────────────────────
def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    업로드된 CSV/Excel 파일을 안전하게 읽어 DataFrame으로 반환.
    - Excel(.xlsx, .xls): openpyxl/xlrd 엔진 사용
    - CSV(.csv, .txt): 여러 인코딩과 구분자를 순차 시도
    """
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()  # bytes

    # ---------- Excel ----------
    if name.endswith((".xlsx", ".xlsm")):
        return pd.read_excel(io.BytesIO(raw), engine="openpyxl")
    if name.endswith(".xls"):
        try:
            return pd.read_excel(io.BytesIO(raw), engine="xlrd")
        except Exception:
            # xlrd가 없으면 openpyxl 재시도
            return pd.read_excel(io.BytesIO(raw))

    # ---------- CSV / TXT ----------
    # 1) BOM 자동 처리를 위해 utf-8-sig를 가장 먼저 시도
    encodings = [
        "utf-8-sig", "utf-8",
        "cp949", "euc-kr",
        "cp1252", "latin1",
        "utf-16", "utf-16le", "utf-16be",
        "iso-8859-1",
    ]
    separators = [None, ",", ";", "\t", "|"]  # None = pandas 자동 추론(engine=python)

    last_err = None
    for enc in encodings:
        for sep in separators:
            try:
                kwargs = dict(encoding=enc)
                if sep is None:
                    kwargs.update(sep=None, engine="python")
                else:
                    kwargs.update(sep=sep)
                df = pd.read_csv(io.BytesIO(raw), **kwargs)
                # 열이 1개로만 잡히면 구분자 추정 실패일 가능성 → 다음 후보 시도
                if df.shape[1] == 1 and sep is not None:
                    continue
                return df
            except Exception as e:
                last_err = e
                continue

    # 모두 실패하면 마지막 에러를 발생
    raise RuntimeError(
        f"파일을 읽지 못했습니다. 마지막 오류: {last_err}\n"
        f"지원 형식: .csv, .txt, .xls, .xlsx, .xlsm"
    )


# ─────────────────────────────────────────────────────────────
# 1단계 : 데이터 업로드
# ─────────────────────────────────────────────────────────────
st.header("1단계 · 데이터 업로드")
st.write("CSV 또는 Excel 파일을 업로드하세요. 다양한 텍스트 인코딩(UTF-8, CP949/EUC-KR, UTF-16 등)을 자동으로 처리합니다.")

uploaded = st.file_uploader(
    "데이터 파일 선택",
    type=["csv", "txt", "xlsx", "xls", "xlsm"],
    help="CSV, TXT, XLSX, XLS, XLSM 형식을 지원합니다.",
)

if uploaded is not None:
    try:
        with st.spinner("파일을 읽는 중..."):
            df = read_uploaded_file(uploaded)
        st.session_state.df = df
        st.success(f"✅ 파일 로드 완료: **{uploaded.name}**  |  shape = {df.shape}")
    except Exception as e:
        st.error(f"❌ 파일을 읽는 중 오류가 발생했습니다.\n\n{e}")

if st.session_state.df is not None:
    df = st.session_state.df
    c1, c2 = st.columns(2)
    c1.metric("행(데이터) 수", f"{df.shape[0]:,}")
    c2.metric("열(특성) 수", f"{df.shape[1]:,}")

    st.subheader("📄 데이터 미리보기")
    st.dataframe(df, use_container_width=True)

    if st.session_state.step == 1:
        if st.button("➡ 2단계로 진행", type="primary"):
            go_next()
            st.rerun()

# ─────────────────────────────────────────────────────────────
# 2단계 : 데이터 구성 정보 확인 및 결측치 제거
# ─────────────────────────────────────────────────────────────
if st.session_state.step >= 2 and st.session_state.df is not None:
    st.header("2단계 · 데이터 구성 정보 확인 및 결측치 제거")
    df = st.session_state.df

    st.subheader("📋 데이터 구성 정보 ( df.info() )")
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Non-Null Count": [df[c].notna().sum() for c in df.columns],
        "Null Count": [df[c].isna().sum() for c in df.columns],
        "Dtype": [str(df[c].dtype) for c in df.columns],
    })
    st.dataframe(info_df, use_container_width=True)

    total_nulls = int(df.isna().sum().sum())
    st.info(f"전체 결측치 개수: **{total_nulls}** 개")

    if st.session_state.df_clean is None:
        if st.button("🧹 결측치 제거 ( df.dropna() )", type="primary"):
            st.session_state.df_clean = df.dropna().reset_index(drop=True)
            st.rerun()
    else:
        df_clean = st.session_state.df_clean
        st.success(f"✅ 결측치 제거 완료 — shape: {df.shape} → **{df_clean.shape}**")
        st.subheader("🧼 정제 후 데이터 미리보기")
        st.dataframe(df_clean.head(20), use_container_width=True)

        if st.session_state.step == 2:
            if st.button("➡ 3단계로 진행", type="primary"):
                go_next()
                st.rerun()

# ─────────────────────────────────────────────────────────────
# 3단계 : 문제(X)와 정답(y) 분리
# ─────────────────────────────────────────────────────────────
if st.session_state.step >= 3 and st.session_state.df_clean is not None:
    st.header("3단계 · 문제(X) 와 정답(y) 분리")
    df_clean = st.session_state.df_clean
    all_cols = list(df_clean.columns)

    # 원본 코드의 기본값: 1~9열이 X, 10열이 y
    default_X = all_cols[1:10] if len(all_cols) >= 11 else all_cols[:-1]
    default_y = all_cols[10] if len(all_cols) >= 11 else all_cols[-1]

    col1, col2 = st.columns(2)
    with col1:
        X_cols = st.multiselect(
            "🔢 문제(X)로 사용할 열 선택",
            options=all_cols,
            default=default_X,
        )
    with col2:
        y_col = st.selectbox(
            "🎯 정답(y)으로 사용할 열 선택",
            options=all_cols,
            index=all_cols.index(default_y) if default_y in all_cols else len(all_cols) - 1,
        )

    if X_cols and y_col and (y_col not in X_cols):
        X = df_clean[X_cols]
        y = df_clean[y_col]
        st.session_state.X = X
        st.session_state.y = y

        c1, c2 = st.columns(2)
        c1.write(f"**X.shape** = {X.shape}")
        c2.write(f"**y.shape** = {y.shape}")

        st.write("**X 미리보기**")
        st.dataframe(X.head(), use_container_width=True)
        st.write("**y 미리보기**")
        st.dataframe(y.head().to_frame(), use_container_width=True)

        if st.session_state.step == 3:
            if st.button("➡ 4단계로 진행", type="primary"):
                go_next()
                st.rerun()
    else:
        st.warning("X에 1개 이상의 열을 선택하고, y는 X에 포함되지 않은 열이어야 합니다.")

# ─────────────────────────────────────────────────────────────
# 4단계 : 원-핫 인코딩
# ─────────────────────────────────────────────────────────────
if st.session_state.step >= 4 and st.session_state.X is not None:
    st.header("4단계 · 원-핫 인코딩")
    X = st.session_state.X
    X_encoded = pd.get_dummies(X)
    st.session_state.X_encoded = X_encoded

    st.write(f"인코딩 전 shape: {X.shape}  →  인코딩 후 shape: **{X_encoded.shape}**")
    st.dataframe(X_encoded.head(20), use_container_width=True)

    if st.session_state.step == 4:
        if st.button("➡ 5단계로 진행", type="primary"):
            go_next()
            st.rerun()

# ─────────────────────────────────────────────────────────────
# 5단계 : 학습 / 테스트 데이터 분할
# ─────────────────────────────────────────────────────────────
if st.session_state.step >= 5 and st.session_state.X_encoded is not None:
    st.header("5단계 · 학습 데이터 / 테스트 데이터 분할")
    c1, c2 = st.columns(2)
    test_size = c1.slider("test_size (테스트 비율)", 0.1, 0.5, 0.3, 0.05)
    random_state = c2.number_input("random_state", value=42, step=1)

    X_train, X_test, y_train, y_test = train_test_split(
        st.session_state.X_encoded,
        st.session_state.y,
        test_size=test_size,
        random_state=int(random_state),
    )
    st.session_state.split = (X_train, X_test, y_train, y_test)

    st.write(
        f"X_train: **{X_train.shape}** | X_test: **{X_test.shape}** | "
        f"y_train: **{y_train.shape}** | y_test: **{y_test.shape}**"
    )

    if st.session_state.step == 5:
        if st.button("➡ 6단계로 진행", type="primary"):
            go_next()
            st.rerun()

# ─────────────────────────────────────────────────────────────
# 6단계 : 나이브 베이즈 모델 학습
# ─────────────────────────────────────────────────────────────
if st.session_state.step >= 6 and st.session_state.split is not None:
    st.header("6단계 · 나이브 베이즈 모델 학습 (GaussianNB)")
    X_train, X_test, y_train, y_test = st.session_state.split

    if st.session_state.model is None:
        if st.button("🚀 모델 학습 시작", type="primary"):
            model = GaussianNB()
            model.fit(X_train, y_train)
            st.session_state.model = model
            st.rerun()
    else:
        model = st.session_state.model
        st.success("✅ 모델 학습 완료")
        st.write("**클래스 종류 (classes_)**:", list(model.classes_))
        st.write("**클래스별 사전확률 (class_prior_)**:")
        prior_df = pd.DataFrame({
            "class": model.classes_,
            "prior_probability": model.class_prior_,
        })
        st.dataframe(prior_df, use_container_width=True)

        if st.session_state.step == 6:
            if st.button("➡ 7단계로 진행", type="primary"):
                go_next()
                st.rerun()

# ─────────────────────────────────────────────────────────────
# 7단계 : 예측 및 평가
# ─────────────────────────────────────────────────────────────
if st.session_state.step >= 7 and st.session_state.model is not None:
    st.header("7단계 · 예측 및 평가")
    X_train, X_test, y_train, y_test = st.session_state.split
    model = st.session_state.model

    y_pred = model.predict(X_test)
    st.session_state.y_pred = y_pred

    accuracy = accuracy_score(y_test, y_pred)
    st.metric("🎯 정확도 (Accuracy)", f"{accuracy:.4f}")

    # 혼동행렬
    st.subheader("📊 혼동행렬 (Confusion Matrix)")
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=model.classes_, yticklabels=model.classes_, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    # 분류 리포트
    st.subheader("📑 분류 리포트 (Classification Report)")
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

    # 예측 결과 다운로드
    st.subheader("⬇ 예측 결과 다운로드")
    result_df = pd.DataFrame({
        "y_test (실제)": np.array(y_test),
        "y_pred (예측)": np.array(y_pred),
    })
    st.dataframe(result_df.head(20), use_container_width=True)
    st.download_button(
        "예측 결과 CSV 다운로드",
        data=result_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="predictions.csv",
        mime="text/csv",
    )

    st.balloons()
    st.success("🎉 모든 단계가 완료되었습니다!")
