
import random
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# 1. CẤU HÌNH
# ============================================================

SEED = 42

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

st.set_page_config(
    page_title="Dự báo giá cổ phiếu LSTM",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# 2. GIAO DIỆN
# ============================================================

st.title("📈 DỰ BÁO GIÁ CỔ PHIẾU BẰNG LSTM")

st.markdown(
    """
    ### Mô hình dự báo giá cổ phiếu sử dụng mạng LSTM

    Ứng dụng sử dụng dữ liệu lịch sử từ **Yahoo Finance** và
    mô hình **Long Short-Term Memory (LSTM)** để dự báo giá đóng cửa.
    
    ⚠️ **Lưu ý:** Kết quả chỉ mang tính chất tham khảo và nghiên cứu,
    không phải khuyến nghị đầu tư.
    """
)


# ============================================================
# 3. THANH CÀI ĐẶT
# ============================================================

st.sidebar.header("⚙️ THIẾT LẬP MÔ HÌNH")

ticker = st.sidebar.text_input(
    "Mã cổ phiếu",
    value="VNM.VN"
).strip().upper()

start_date = st.sidebar.date_input(
    "Ngày bắt đầu",
    value=date(2018, 1, 1)
)

today = date.today()

end_date = st.sidebar.date_input(
    "Ngày kết thúc",
    value=today,
    max_value=today
)

window_size = st.sidebar.selectbox(
    "Số ngày dữ liệu đầu vào",
    [30, 45, 60, 90],
    index=2
)

forecast_days = st.sidebar.slider(
    "Số phiên muốn dự báo",
    min_value=1,
    max_value=30,
    value=10
)

epochs = st.sidebar.slider(
    "Số Epoch",
    min_value=3,
    max_value=50,
    value=10
)

run_model = st.sidebar.button(
    "🚀 CHẠY MÔ HÌNH",
    use_container_width=True
)


# ============================================================
# 4. HÀM TẢI DỮ LIỆU
# ============================================================

@st.cache_data(ttl=1800)
def download_data(ticker, start_date, end_date):

    df = yf.download(
        ticker,
        start=start_date.isoformat(),
        end=(end_date + timedelta(days=1)).isoformat(),
        auto_adjust=False,
        progress=False,
        threads=False
    )

    if df.empty:
        return pd.DataFrame()

    # Xử lý trường hợp Yahoo Finance trả về MultiIndex
    if isinstance(df.columns, pd.MultiIndex):

        close = df["Close"]

        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

    else:
        close = df["Close"]

    result = pd.DataFrame({
        "Close": pd.to_numeric(
            close,
            errors="coerce"
        )
    })

    result.dropna(inplace=True)

    return result


# ============================================================
# 5. HÀM TẠO SEQUENCE
# ============================================================

def create_sequences(data, window_size):

    X = []
    y = []

    for i in range(window_size, len(data)):

        X.append(
            data[i - window_size:i]
        )

        y.append(
            data[i]
        )

    return np.array(X), np.array(y)


# ============================================================
# 6. CHỜ NGƯỜI DÙNG CHẠY
# ============================================================

if not run_model:

    st.info(
        "👈 Hãy nhập mã cổ phiếu và nhấn "
        "**CHẠY MÔ HÌNH** để bắt đầu."
    )

    st.stop()


# ============================================================
# 7. KIỂM TRA INPUT
# ============================================================

if not ticker:

    st.error("Vui lòng nhập mã cổ phiếu.")

    st.stop()


if start_date >= end_date:

    st.error(
        "Ngày bắt đầu phải nhỏ hơn ngày kết thúc."
    )

    st.stop()


# ============================================================
# 8. TẢI DỮ LIỆU
# ============================================================

with st.spinner(
    f"⏳ Đang tải dữ liệu {ticker}..."
):

    try:

        df = download_data(
            ticker,
            start_date,
            end_date
        )

    except Exception as e:

        st.error(
            f"Không thể tải dữ liệu: {e}"
        )

        st.stop()


if df.empty:

    st.error(
        "Không tìm thấy dữ liệu. "
        "Hãy kiểm tra lại mã cổ phiếu."
    )

    st.stop()


if len(df) < window_size + 30:

    st.error(
        f"Dữ liệu hiện có {len(df)} phiên. "
        f"Cần ít nhất khoảng {window_size + 30} phiên."
    )

    st.stop()


# ============================================================
# 9. THÔNG TIN CỔ PHIẾU
# ============================================================

st.subheader(
    f"📊 Dữ liệu cổ phiếu {ticker}"
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Giá đóng cửa gần nhất",
    f"{df['Close'].iloc[-1]:,.2f}"
)

col2.metric(
    "Số phiên dữ liệu",
    f"{len(df):,}"
)

col3.metric(
    "Ngày gần nhất",
    df.index[-1].strftime("%d/%m/%Y")
)


# ============================================================
# 10. BIỂU ĐỒ GIÁ LỊCH SỬ
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 5)
)

ax.plot(
    df.index,
    df["Close"],
    label="Giá đóng cửa"
)

ax.set_title(
    f"Giá cổ phiếu {ticker}"
)

ax.set_xlabel("Ngày")

ax.set_ylabel("Giá")

ax.grid(alpha=0.25)

ax.legend()

st.pyplot(fig)

plt.close(fig)


# ============================================================
# 11. CHUẨN BỊ DỮ LIỆU
# ============================================================

data = df[["Close"]].values.astype(float)

train_size = int(
    0.8 * len(data)
)

train_data = data[:train_size]

test_data = data[train_size:]


# ============================================================
# 12. CHUẨN HÓA DỮ LIỆU
# ============================================================

scaler = MinMaxScaler()

train_scaled = scaler.fit_transform(
    train_data
)

test_scaled = scaler.transform(
    test_data
)


# ============================================================
# 13. TẠO TRAIN SEQUENCE
# ============================================================

X_train, y_train = create_sequences(
    train_scaled,
    window_size
)


# ============================================================
# 14. TẠO TEST SEQUENCE
# ============================================================

test_combined = np.vstack(
    (
        train_scaled[-window_size:],
        test_scaled
    )
)

X_test, y_test = create_sequences(
    test_combined,
    window_size
)


# ============================================================
# 15. RESHAPE
# ============================================================

X_train = X_train.reshape(
    X_train.shape[0],
    X_train.shape[1],
    1
)

X_test = X_test.reshape(
    X_test.shape[0],
    X_test.shape[1],
    1
)


# ============================================================
# 16. CHIA VALIDATION
# ============================================================

val_size = max(
    1,
    int(0.1 * len(X_train))
)

X_train_final = X_train[:-val_size]

y_train_final = y_train[:-val_size]

X_val = X_train[-val_size:]

y_val = y_train[-val_size:]


# ============================================================
# 17. XÂY DỰNG MÔ HÌNH LSTM
# ============================================================

model = Sequential()

model.add(
    Input(
        shape=(window_size, 1)
    )
)

model.add(
    LSTM(
        50,
        return_sequences=True
    )
)

model.add(
    LSTM(50)
)

model.add(
    Dense(1)
)


# ============================================================
# 18. COMPILE
# ============================================================

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)


# ============================================================
# 19. EARLY STOPPING
# ============================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# ============================================================
# 20. HUẤN LUYỆN
# ============================================================

st.subheader(
    "🧠 Huấn luyện mô hình LSTM"
)

progress_text = st.empty()

with st.spinner(
    "Đang huấn luyện mô hình..."
):

    history = model.fit(

        X_train_final,
        y_train_final,

        validation_data=(
            X_val,
            y_val
        ),

        epochs=epochs,

        batch_size=32,

        shuffle=False,

        callbacks=[
            early_stop
        ],

        verbose=0
    )


st.success(
    "✅ Huấn luyện mô hình hoàn tất!"
)


# ============================================================
# 21. DỰ BÁO TẬP TEST
# ============================================================

test_pred = model.predict(
    X_test,
    verbose=0
)


test_pred = scaler.inverse_transform(
    test_pred
)


y_test_real = scaler.inverse_transform(
    y_test.reshape(-1, 1)
)


# ============================================================
# 22. ĐÁNH GIÁ
# ============================================================

rmse = np.sqrt(
    mean_squared_error(
        y_test_real,
        test_pred
    )
)

mae = mean_absolute_error(
    y_test_real,
    test_pred
)

r2 = r2_score(
    y_test_real,
    test_pred
)


# ============================================================
# 23. HIỂN THỊ METRICS
# ============================================================

st.subheader(
    "📌 Đánh giá mô hình"
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "RMSE",
    f"{rmse:.4f}"
)

col2.metric(
    "MAE",
    f"{mae:.4f}"
)

col3.metric(
    "R²",
    f"{r2:.4f}"
)


# ============================================================
# 24. BIỂU ĐỒ LOSS
# ============================================================

st.subheader(
    "📉 Loss Train và Validation"
)

fig, ax = plt.subplots(
    figsize=(10, 5)
)

ax.plot(
    history.history["loss"],
    label="Train Loss"
)

ax.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

ax.set_title(
    "Loss Train vs Validation"
)

ax.set_xlabel("Epoch")

ax.set_ylabel("Loss")

ax.legend()

ax.grid(alpha=0.25)

st.pyplot(fig)

plt.close(fig)


# ============================================================
# 25. BIỂU ĐỒ GIÁ THỰC TẾ VS DỰ BÁO
# ============================================================

st.subheader(
    "📈 Giá thực tế và giá dự báo"
)

test_dates = df.index[train_size:]


fig, ax = plt.subplots(
    figsize=(12, 6)
)

ax.plot(
    test_dates,
    y_test_real,
    label="Giá thật"
)

ax.plot(
    test_dates,
    test_pred,
    label="Dự báo"
)

ax.set_title(
    f"Giá thật vs dự báo - {ticker}"
)

ax.set_xlabel("Ngày")

ax.set_ylabel("Giá")

ax.legend()

ax.grid(alpha=0.25)

st.pyplot(fig)

plt.close(fig)


# ============================================================
# 26. DỰ BÁO TƯƠNG LAI
# ============================================================

st.subheader(
    f"🔮 Dự báo {forecast_days} phiên tiếp theo"
)


all_scaled = np.vstack(
    (
        train_scaled,
        test_scaled
    )
)


last_60_days = all_scaled[
    -window_size:
]


last_60_days = last_60_days.reshape(
    1,
    window_size,
    1
)


future_predictions = []


for i in range(
    forecast_days
):

    next_pred = model.predict(
        last_60_days,
        verbose=0
    )

    future_predictions.append(
        next_pred[0, 0]
    )

    last_60_days = np.concatenate(
        [
            last_60_days[:, 1:, :],
            next_pred.reshape(
                1,
                1,
                1
            )
        ],
        axis=1
    )


# ============================================================
# 27. CHUYỂN VỀ GIÁ THỰC
# ============================================================

future_predictions = np.array(
    future_predictions
).reshape(-1, 1)


future_predictions = scaler.inverse_transform(
    future_predictions
).ravel()


# ============================================================
# 28. TẠO NGÀY DỰ BÁO
# ============================================================

future_dates = pd.bdate_range(
    df.index[-1] + pd.Timedelta(days=1),
    periods=forecast_days
)


forecast_df = pd.DataFrame({

    "Phiên":
        range(
            1,
            forecast_days + 1
        ),

    "Ngày":
        future_dates.strftime(
            "%d/%m/%Y"
        ),

    "Giá dự báo":
        future_predictions
})


# ============================================================
# 29. HIỂN THỊ BẢNG
# ============================================================

st.dataframe(
    forecast_df.style.format(
        {
            "Giá dự báo": "{:,.2f}"
        }
    ),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 30. BIỂU ĐỒ DỰ BÁO TƯƠNG LAI
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 6)
)


# Lấy 90 phiên gần nhất
history_tail = df.tail(
    min(90, len(df))
)


ax.plot(
    history_tail.index,
    history_tail["Close"],
    label="Giá lịch sử"
)


ax.plot(
    future_dates,
    future_predictions,
    marker="o",
    linestyle="--",
    label="Dự báo"
)


ax.set_title(
    f"Dự báo giá cổ phiếu {ticker}"
)

ax.set_xlabel("Ngày")

ax.set_ylabel("Giá")

ax.legend()

ax.grid(alpha=0.25)


st.pyplot(fig)

plt.close(fig)


# ============================================================
# 31. DOWNLOAD CSV
# ============================================================

csv = forecast_df.to_csv(
    index=False
).encode(
    "utf-8-sig"
)


st.download_button(

    label="📥 Tải kết quả dự báo CSV",

    data=csv,

    file_name=f"du_bao_{ticker}.csv",

    mime="text/csv",

    use_container_width=True
)


# ============================================================
# 32. CẢNH BÁO
# ============================================================

st.warning(
    """
    ⚠️ **Lưu ý:**
    
    Mô hình LSTM chỉ sử dụng dữ liệu giá lịch sử để dự báo.
    Giá cổ phiếu thực tế có thể bị ảnh hưởng bởi tin tức,
    tình hình doanh nghiệp, kinh tế vĩ mô, thị trường,
    thanh khoản và nhiều yếu tố khác.
    
    Kết quả trên không phải là lời khuyên mua hoặc bán cổ phiếu.
    """
)
