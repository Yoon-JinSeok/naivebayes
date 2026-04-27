import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from scipy.spatial.distance import hamming

# ---------------------------------------------------------------
# 페이지 설정
# ---------------------------------------------------------------
st.set_page_config(
    page_title="나이브 베이즈 기반 분류",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 나이브 베이즈 기반 확률을 이용한 분류")
st.caption("제작 : 윤진석")
st.markdown("---")

# ---------------------------------------------------------------
# 세션 상태 초기화 (단계별 진행 관리)
# ---------------------------------------------------------------
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
if "split_data" not in st.session_state:
    st.session_state.split_data = None
if "model" not in st.session_state:
    st.session_state.model = None
if "y_pred" not in st.session_state:
    st.session_state.y_pred = None


def goto(step: int):
    st.session_state.step = step


# ---------------------------------------------------------------
# 사이드바 - 진행 상황
# ---------------------------------------------------------------
with st.sidebar:
    st.header("📍 진행 단계")
    steps = [
        "1️⃣ 데이터 업로드",
        "2️⃣ 정보 확인 / 결측치 제거",
        "3️⃣ 문제와 정답 분리",
        "4️⃣ 원-핫 인코딩",
        "5️⃣ 학습/테스트 분할",
        "6️⃣ 모델 학습",
        "7️⃣ 예측 및 평가",
    ]
    for i, name in enumerate(steps, start=1):
        if i == st.session_state.step:
            st.markdown(f"**▶ {name}**")
        elif i < st.session_state.step:
            st.markdown(f"✅ {name}")
        else:
            st.markdown(f"⬜ {name}")

    st.markdown("---")
    if st.button("🔄 처음부터 다시 시작"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ===============================================================
# 1단계 : 데이터 업로드
# ===============================================================
if st.session_state.step >= 1:
    st.header("1단계 : 데이터 업로드")
    uploaded = st.file_uploader("CSV 파일을 업로드하세요 (예: Train.csv)", type=["csv"])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.session_state.df = df
            st.success(f"✅ 데이터 로드 완료! 행: {df.shape[0]}, 열: {df.shape[1]}")
            st.write("**데이터 미리보기 (상위 10행)**")
            st.dataframe(df.head(10))
            st.write(f"**Shape:** {df.shape}")

            if st.session_state.step == 1:
                if st.button("➡️ 다음 단계로", key="to2"):
                    goto(2)
                    st.rerun()
        except Exception as e:
            st.error(f"파일 읽기 오류: {e}")


# ===============================================================
# 2단계 : 데이터 정보 확인 및 결측치 제거
# ===============================================================
if st.session_state.step >= 2 and st.session_state.df is not None:
    st.markdown("---")
    st.header("2단계 : 데이터 구성 정보 확인 및 결측치 제거")

    df = st.session_state.df

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 데이터 정보 (info)")
        info_df = pd.DataFrame({
            "컬럼명": df.columns,
            "Non-Null 개수": [df[c].notna().sum() for c in df.columns],
            "결측치 개수": [df[c].isna().sum() for c in df.columns],
            "Dtype": [str(df[c].dtype) for c in df.columns],
        })
        st.dataframe(info_df, use_container_width=True)
        st.write(f"전체 행: **{df.shape[0]}**, 전체 열: **{df.shape[1]}**")
        st.write(f"총 결측치: **{df.isna().sum().sum()}**")

    with col2:
        st.subheader("🧹 결측치 제거 후")
        df_clean = df.dropna()
        st.session_state.df_clean = df_clean
        info_df2 = pd.DataFrame({
            "컬럼명": df_clean.columns,
            "Non-Null 개수": [df_clean[c].notna().sum() for c in df_clean.columns],
            "Dtype": [str(df_clean[c].dtype) for c in df_clean.columns],
        })
        st.dataframe(info_df2, use_container_width=True)
        st.write(f"전체 행: **{df_clean.shape[0]}**, 전체 열: **{df_clean.shape[1]}**")
        st.success(f"제거된 행 수: {df.shape[0] - df_clean.shape[0]}")

    if st.session_state.step == 2:
        if st.button("➡️ 다음 단계로", key="to3"):
            goto(3)
            st.rerun()


# ===============================================================
# 3단계 : 문제(X)와 정답(y) 분리
# ===============================================================
if st.session_state.step >= 3 and st.session_state.df_clean is not None:
    st.markdown("---")
    st.header("3단계 : 문제(학습용 속성)와 정답(레이블) 분리")

    df_clean = st.session_state.df_clean
    cols = list(df_clean.columns)

    st.write("학습용으로 사용할 **문제(X) 컬럼들**과 **정답(y) 컬럼**을 선택하세요.")

    # 기본값: 1~9열을 X, 10열을 y (원본 코드와 동일)
    default_X = cols[1:10] if len(cols) >= 11 else cols[:-1]
    default_y = cols[10] if len(cols) >= 11 else cols[-1]

    x_cols = st.multiselect("📥 문제(X) 컬럼 선택", cols, default=default_X)
    y_col = st.selectbox(
        "🎯 정답(y) 컬럼 선택",
        cols,
        index=cols.index(default_y) if default_y in cols else len(cols) - 1,
    )

    if x_cols and y_col and y_col not in x_cols:
        X = df_clean[x_cols]
        y = df_clean[y_col]
        st.session_state.X = X
        st.session_state.y = y

        col1, col2 = st.columns(2)
        with col1:
            st.write("**X (문제) 미리보기**")
            st.dataframe(X.head())
            st.write(f"X.shape = {X.shape}")
        with col2:
            st.write("**y (정답) 미리보기**")
            st.dataframe(y.head())
            st.write(f"y.shape = {y.shape}")
            st.write("**클래스 분포**")
            st.bar_chart(y.value_counts())

        if st.session_state.step == 3:
            if st.button("➡️ 다음 단계로", key="to4"):
                goto(4)
                st.rerun()
    else:
        st.warning("문제(X)와 정답(y)을 올바르게 선택하세요. 정답은 X에 포함되지 않아야 합니다.")


# ===============================================================
# 4단계 : 원-핫 인코딩
# ===============================================================
if st.session_state.step >= 4 and st.session_state.X is not None:
    st.markdown("---")
    st.header("4단계 : 원-핫 인코딩")

    X = st.session_state.X
    X_encoded = pd.get_dummies(X)
    st.session_state.X_encoded = X_encoded

    st.write("**원-핫 인코딩 결과**")
    st.dataframe(X_encoded.head(10))
    st.info(f"인코딩 전: {X.shape[1]}개 컬럼 → 인코딩 후: {X_encoded.shape[1]}개 컬럼")

    if st.session_state.step == 4:
        if st.button("➡️ 다음 단계로", key="to5"):
            goto(5)
            st.rerun()


# ===============================================================
# 5단계 : 학습/테스트 데이터 분할
# ===============================================================
if st.session_state.step >= 5 and st.session_state.X_encoded is not None:
    st.markdown("---")
    st.header("5단계 : 학습 데이터와 테스트 데이터 분할")

    test_size = st.slider("테스트 데이터 비율", 0.1, 0.5, 0.3, 0.05)
    random_state = st.number_input("random_state", value=42, step=1)

    X_encoded = st.session_state.X_encoded
    y = st.session_state.y

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=int(random_state)
    )
    st.session_state.split_data = (X_train, X_test, y_train, y_test)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("X_train", f"{X_train.shape}")
    c2.metric("X_test", f"{X_test.shape}")
    c3.metric("y_train", f"{y_train.shape}")
    c4.metric("y_test", f"{y_test.shape}")

    if st.session_state.step == 5:
        if st.button("➡️ 다음 단계로", key="to6"):
            goto(6)
            st.rerun()


# ===============================================================
# 6단계 : 모델 학습
# ===============================================================
if st.session_state.step >= 6 and st.session_state.split_data is not None:
    st.markdown("---")
    st.header("6단계 : 모델 학습 (Gaussian Naive Bayes)")

    if st.button("🚀 모델 학습 시작", key="train"):
        X_train, X_test, y_train, y_test = st.session_state.split_data
        with st.spinner("모델을 학습 중입니다..."):
            model = GaussianNB()
            model.fit(X_train, y_train)
            st.session_state.model = model
        st.success("✅ 모델 학습 완료!")
        st.write(model)

    if st.session_state.model is not None and st.session_state.step == 6:
        if st.button("➡️ 다음 단계로", key="to7"):
            goto(7)
            st.rerun()


# ===============================================================
# 7단계 : 예측 및 평가
# ===============================================================
if st.session_state.step >= 7 and st.session_state.model is not None:
    st.markdown("---")
    st.header("7단계 : 예측 및 평가")

    if st.button("🔮 예측 실행", key="predict"):
        X_train, X_test, y_train, y_test = st.session_state.split_data
        model = st.session_state.model
        y_pred = model.predict(X_test)
        st.session_state.y_pred = y_pred

    if st.session_state.y_pred is not None:
        X_train, X_test, y_train, y_test = st.session_state.split_data
        y_pred = st.session_state.y_pred

        accuracy = accuracy_score(y_test, y_pred)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("🎯 정확도 (Accuracy)", f"{accuracy:.4f}")

            # ----- 보너스: 해밍거리 -----
            st.markdown("---")
            st.subheader("📐 해밍거리(Hamming Distance)")
            st.caption("y_test와 y_pred 사이의 해밍거리 (불일치 비율 및 개수)")

            y_test_arr = np.array(y_test)
            y_pred_arr = np.array(y_pred)
            mismatch_count = int(np.sum(y_test_arr != y_pred_arr))
            hamming_ratio = mismatch_count / len(y_test_arr)

            st.write(f"- 불일치 개수: **{mismatch_count} / {len(y_test_arr)}**")
            st.write(f"- 해밍거리(비율): **{hamming_ratio:.4f}**")
            st.write(f"- 1 − 해밍거리(비율) = 정확도 = **{1 - hamming_ratio:.4f}**")
            st.info("✔ 분류 결과의 해밍거리(비율)는 (1 − accuracy)와 같습니다.")

        with col2:
            st.subheader("혼동 행렬 (Confusion Matrix)")
            cm = confusion_matrix(y_test, y_pred, labels=sorted(np.unique(y_test_arr)))
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=sorted(np.unique(y_test_arr)),
                yticklabels=sorted(np.unique(y_test_arr)),
                ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

        st.markdown("### 예측 결과 샘플")
        result_df = pd.DataFrame({
            "실제값(y_test)": np.array(y_test)[:30],
            "예측값(y_pred)": y_pred[:30],
            "일치여부": (np.array(y_test)[:30] == y_pred[:30]),
        })
        st.dataframe(result_df, use_container_width=True)

        st.balloons()
