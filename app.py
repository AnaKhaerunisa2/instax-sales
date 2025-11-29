import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib as mpl
import matplotlib.lines as mlines
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
from PIL import Image

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Forecasting Sales Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS CUSTOM (UNTUK BACKGROUND & TAMPILAN) ---
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://images.unsplash.com/photo-1680039211140-17b72b017381?q=80&w=755&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
             background-attachment: fixed;
             background-size: cover;
         }}
         /* Membuat konten lebih terbaca dengan background semi-transparan */
         .main .block-container {{
             background-color: rgba(255, 255, 255, 0.85);
             padding: 2rem;
             border-radius: 10px;
         }}
         /* Mode Gelap Support (Override jika user pake dark mode) */
         @media (prefers-color-scheme: dark) {{
             .main .block-container {{
                 background-color: rgba(14, 17, 23, 0.85);
             }}
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# Panggil fungsi background hanya di halaman Home
# (Kita atur logika navigasi dulu)

# --- FUNGSI LOAD DATA INSTAX & MODEL (CACHE) ---
@st.cache_data
def load_instax_data():
    try:
        df = pd.read_csv('instax_sales_transaction_data.csv')
        df['Tanggal'] = pd.to_datetime(df['Tanggal'])
        # Agregasi Bulanan
        df_monthly = df.set_index('Tanggal').resample('MS')['Qty'].sum().reset_index()
        df_monthly.columns = ['Bulan', 'Total_Qty']
        # Hapus bulan terakhir (sesuai analisis notebook)
        df_monthly = df_monthly.iloc[:-1]
        return df_monthly
    except FileNotFoundError:
        return None

@st.cache_resource
def load_models():
    models = {}
    try:
        with open('model_holtwinters.pkl', 'rb') as f:
            models['HW'] = pickle.load(f)
        with open('model_sarima.pkl', 'rb') as f:
            models['SARIMA'] = pickle.load(f)
    except:
        pass
    return models

# --- NAVIGASI SIDEBAR ---
st.sidebar.title("Navigasi")
page = st.sidebar.radio("Pergi ke:", ["🏠 Home", "📊 Analisis Model (Instax)", "🔮 Uji Coba Data Baru", "👤 Author"])

# =================================================================================
# HALAMAN 1: HOME
# =================================================================================
if page == "🏠 Home":
    add_bg_from_url() # Pakai background image
    
    st.title("Sistem Peramalan Penjualan (Forecasting)")
    st.subheader("Studi Kasus: Fujifilm Instax Sales Transaction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        Selamat datang di **Sales Forecasting Dashboard**. Web App ini dirancang untuk menganalisis data penjualan historis dan memprediksi tren penjualan di masa depan menggunakan metode *Machine Learning* Time Series.
        
        **Fitur Utama:**
        * 📊 **Analisis Data Historis:** Visualisasi tren penjualan Instax (2022-2025).
        * 🤖 **Perbandingan Model:** Evaluasi kinerja SARIMA vs Holt-Winters.
        * 🔮 **Uji Coba Fleksibel:** Upload dataset penjualan produk apa saja (CSV) dan dapatkan prediksinya secara otomatis.
        
        **Tentang Dataset:**
        Data yang digunakan dalam pelatihan model utama adalah data transaksi sintetis produk Fujifilm Instax.
        Dataset dapat diakses publik melalui Kaggle.
        """)
        
        st.link_button("📂 Lihat Dataset di Kaggle", "https://www.kaggle.com/datasets/bertnardomariouskono/fujifilm-instax-sales-transaction-data-synthetic")

    with col2:
        st.info("💡 **Metodologi:** CRISP-DM (Cross Industry Standard Process for Data Mining)")
        st.success("🏆 **Model Terbaik:** SARIMA (MAPE ~7.24%)")

# =================================================================================
# HALAMAN 2: ANALISIS MODEL (LATIHAN)
# =================================================================================
elif page == "📊 Analisis Model (Instax)":
    st.title("📊 Analisis & Evaluasi Model (Instax)")
    st.markdown("Halaman ini menampilkan hasil pelatihan model yang telah dilakukan di Google Colab menggunakan dataset Instax.")
    
    st.info("Notebook Google Colab sumber: [Klik disini untuk membuka Notebook](https://colab.research.google.com/drive/1hifESAr3HTskkhDxL9z13wu4f1jKDwym?usp=sharing)")

    df_instax = load_instax_data()
    models = load_models()
    
    if df_instax is not None and models:
        # --- Metrics ---
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Data", f"{len(df_instax)} Bulan")
        with col2:
            st.metric("Model Terbaik", "SARIMA")
        with col3:
            st.metric("Akurasi (MAPE)", "7.24%", delta="Sangat Akurat")
            
        # --- Visualisasi ---
        st.subheader("Grafik Peramalan: SARIMA vs Holt-Winters")
        
        # Slider Prediksi
        months_pred = st.slider("Prediksi Bulan ke Depan:", 1, 12, 6)
        
        # Generate Forecast
        last_date = df_instax['Bulan'].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=months_pred, freq='MS')
        
        # Forecast calculation
        pred_hw = models['HW'].forecast(months_pred)
        pred_sarima = models['SARIMA'].predict(n_periods=months_pred)
        
        # Matplotlib Graph with gradient historical line (black -> white)
        df_show = df_instax.iloc[-24:]
        x = pd.to_datetime(df_show['Bulan'])
        y = df_show['Total_Qty'].values

        # Create line segments for gradient
        # Convert pandas Series of datetimes to python datetimes array
        try:
            xnum = mpl.dates.date2num(x.dt.to_pydatetime())
        except Exception:
            xnum = mpl.dates.date2num(pd.to_datetime(x).values)
        points = np.array([xnum, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        fig, ax = plt.subplots(figsize=(12, 6))
        lc = LineCollection(segments, cmap='gray', norm=plt.Normalize(0, 1), linewidth=2)
        lc.set_array(np.linspace(0, 1, len(segments)))
        ax.add_collection(lc)
        ax.autoscale_view()

        # Plot predictions on top
        try:
            ax.plot(future_dates, pred_hw, marker='o', color='blue', linestyle='--', label='Prediksi Holt-Winters')
        except Exception:
            pass
        try:
            ax.plot(future_dates, pred_sarima, marker='o', color='red', linestyle='--', label='Prediksi SARIMA')
        except Exception:
            pass

        # Axis formatting
        ax.set_title("Proyeksi Penjualan Masa Depan")
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Qty Penjualan")
        ax.grid(True, linestyle=':', alpha=0.6)

        # Legend: create a proxy for the gradient historical line
        proxy_hist = mlines.Line2D([], [], color='gray', linewidth=2, label='Data Historis (Aktual)')
        handles, labels = ax.get_legend_handles_labels()
        handles.insert(0, proxy_hist)
        labels.insert(0, proxy_hist.get_label())
        ax.legend(handles=handles, labels=labels)

        st.pyplot(fig, use_container_width=True)
        
        # Tabel Data
        with st.expander("Lihat Detail Angka Prediksi"):
            df_res = pd.DataFrame({'Bulan': future_dates, 'SARIMA': pred_sarima.astype(int), 'Holt-Winters': pred_hw.astype(int)})
            st.dataframe(df_res)
            
    else:
        st.error("File dataset (csv) atau model (.pkl) tidak ditemukan di direktori folder.")

# =================================================================================
# HALAMAN 3: UJI COBA DATA BARU (TESTING)
# =================================================================================
elif page == "🔮 Uji Coba Data Baru":
    st.title("🔮 Uji Coba Peramalan Data Baru")
    st.markdown("""
    Di halaman ini, Anda bisa mengunggah dataset penjualan **apapun** (tidak harus Instax). 
    Sistem akan melatih model baru secara otomatis dan memberikan prediksi.
    """)
    
    with st.expander("ℹ️ Petunjuk Format Data Upload"):
        st.markdown("""
        Agar sistem berjalan lancar, pastikan file CSV Anda memiliki setidaknya 2 kolom:
        1. **Kolom Tanggal:** Format tanggal (YYYY-MM-DD atau DD/MM/YYYY).
        2. **Kolom Nilai:** Angka penjualan/quantity (Harus numerik).
        
        *Contoh:*
        | Tanggal | Penjualan |
        | :--- | :--- |
        | 2023-01-01 | 150 |
        | 2023-01-02 | 200 |
        """)

    uploaded_file = st.file_uploader("Upload File CSV Penjualan", type=['csv'])

    if uploaded_file is not None:
        try:
            df_new = pd.read_csv(uploaded_file)
            st.success("File berhasil diupload!")
            st.dataframe(df_new.head())
            
            # --- INPUT USER MAPPING KOLOM ---
            col1, col2 = st.columns(2)
            with col1:
                date_col = st.selectbox("Pilih Kolom Tanggal:", df_new.columns)
            with col2:
                val_col = st.selectbox("Pilih Kolom Nilai (Qty/Sales):", df_new.columns)
            
            # --- PREPROCESSING OTOMATIS ---
            # Konversi tanggal dan Resampling ke Bulanan (Agar stabil)
            df_new[date_col] = pd.to_datetime(df_new[date_col])
            df_agg = df_new.set_index(date_col).resample('MS')[val_col].sum().reset_index()
            df_agg.columns = ['Bulan', 'Value']
            
            st.divider()
            st.subheader("Visualisasi Data Anda (Per Bulan)")
            st.line_chart(df_agg.set_index('Bulan'))
            
            # --- MENU PREDIKSI ---
            st.sidebar.markdown("---")
            st.sidebar.header("⚙️ Pengaturan Prediksi")
            model_option = st.sidebar.selectbox("Pilih Model:", ["Holt-Winters", "SARIMA (Auto-ARIMA)"])
            period_option = st.sidebar.slider("Durasi Prediksi (Bulan):", 1, 24, 6)
            
            if st.button("🚀 Mulai Prediksi"):
                with st.spinner(f"Sedang melatih model {model_option} dengan data Anda..."):
                    
                    forecast_values = []
                    # Latih Model On-The-Fly
                    try:
                        if model_option == "Holt-Winters":
                            # Gunakan seasonal periods 12 jika data cukup, jika tidak none
                            seasonal_per = 12 if len(df_agg) >= 24 else None
                            trend_type = 'add'
                            seasonal_type = 'add' if seasonal_per else None
                            
                            model = ExponentialSmoothing(
                                df_agg['Value'], trend=trend_type, seasonal=seasonal_type, seasonal_periods=seasonal_per
                            ).fit()
                            forecast_values = model.forecast(period_option)
                            
                        elif model_option == "SARIMA (Auto-ARIMA)":
                            # Auto Arima yang ringan
                            model = auto_arima(df_agg['Value'], seasonal=True, m=12, suppress_warnings=True, stepwise=True)
                            forecast_values = model.predict(n_periods=period_option)
                        
                        # --- TAMPILKAN HASIL ---
                        future_dates = pd.date_range(start=df_agg['Bulan'].iloc[-1] + pd.DateOffset(months=1), periods=period_option, freq='MS')
                        
                        # Grafik Hasil
                        fig_test = go.Figure()
                        fig_test.add_trace(go.Scatter(x=df_agg['Bulan'], y=df_agg['Value'], mode='lines', name='Data Historis'))
                        fig_test.add_trace(go.Scatter(x=future_dates, y=forecast_values, mode='lines+markers', name='Prediksi', line=dict(color='green', width=3)))
                        fig_test.update_layout(title=f"Hasil Prediksi Menggunakan {model_option}", xaxis_title="Waktu", yaxis_title="Nilai")
                        
                        st.plotly_chart(fig_test, use_container_width=True)
                        
                        # Metric Ringkasan
                        total_pred = forecast_values.sum()
                        st.success(f"Prediksi selesai! Total nilai untuk {period_option} bulan ke depan diperkirakan: {total_pred:,.0f}")
                        
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat pemodelan: {e}. Pastikan data cukup panjang untuk analisis time series.")

        except Exception as e:
            st.error(f"Gagal memproses file: {e}")

# =================================================================================
# HALAMAN 4: AUTHOR
# =================================================================================
elif page == "👤 Author":
    st.title("Tentang Penulis")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        # Placeholder Image (Bisa diganti foto asli jika ada file-nya)
        # Jika Anda punya foto profil, simpan dengan nama profilepic.png di folder yang sama
        # atau gunakan URL gambar
        st.image("profilepic.png", width=200)
    
    with col2:
        st.markdown("""
        ### Ana Khaerunisa
        **NIM: 202210715117**
        
        **Data Scientist | Informatics Student**
        
        Website ini merupakan bagian dari proyek tugas akhir/portofolio di bidang *Data Science* & *Deep Learning*.
        Fokus penelitian adalah penerapan algoritma *Time Series Forecasting* untuk membantu perencanaan bisnis.
        
        ---
        **Hubungi Saya:**
        * 📧 Email: [202210715117@mhs.ubharajaya.ac.id](mailto:202210715117@mhs.ubharajaya.ac.id)
        """)
    
    st.markdown("---")
    st.markdown("© 2025 Ana Khaerunisa. Built with Streamlit & Python.")