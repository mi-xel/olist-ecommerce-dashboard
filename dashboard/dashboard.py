import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.set_page_config(
    page_title="Olist E-Commerce Executive Dashboard",
    page_icon="🔵",
    layout="wide"
)

# NAMA LENGKAP NEGARA BAGIAN BRASIL
BRAZIL_STATES = {
    "AC": "Acre",
    "AL": "Alagoas",
    "AP": "Amapá",
    "AM": "Amazonas",
    "BA": "Bahia",
    "CE": "Ceará",
    "DF": "Distrito Federal",
    "ES": "Espírito Santo",
    "GO": "Goiás",
    "MA": "Maranhão",
    "MT": "Mato Grosso",
    "MS": "Mato Grosso do Sul",
    "MG": "Minas Gerais",
    "PA": "Pará",
    "PB": "Paraíba",
    "PR": "Paraná",
    "PE": "Pernambuco",
    "PI": "Piauí",
    "RJ": "Rio de Janeiro",
    "RN": "Rio Grande do Norte",
    "RS": "Rio Grande do Sul",
    "RO": "Rondônia",
    "RR": "Roraima",
    "SC": "Santa Catarina",
    "SP": "São Paulo",
    "SE": "Sergipe",
    "TO": "Tocantins"
}

# TEMA BIRU-PUTIH OLIST, STICKY HEADER & KPI CARDS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: #F8FAFC;
    color: #0F172A;
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1380px;
}

[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E2E8F0;
}

.sticky-header {
    position: sticky;
    top: 2.85rem;
    z-index: 999;
    background: rgba(248, 250, 252, 0.96);
    backdrop-filter: blur(8px);
    padding: 0.75rem 0 0.85rem 0;
    border-bottom: 1px solid #E2E8F0;
    margin-bottom: 1.25rem;
}

.dashboard-title {
    font-size: 1.9rem;
    font-weight: 800;
    color: #0F172A;
    margin: 0;
    line-height: 1.2;
    letter-spacing: -0.02em;
}

.dashboard-title span {
    color: #0C29D0;
}

.reporting-period {
    font-size: 0.88rem;
    font-style: italic;
    color: #64748B;
    margin: 0.25rem 0 0 0;
}

.kpi-card {
    background: #FFFFFF;
    border-radius: 12px;
    border: 1px solid #E2E8F0;
    border-top: 4px solid #0C29D0;
    padding: 1.15rem 1rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    transition: transform 0.15s ease;
}

.kpi-card:hover {
    transform: translateY(-2px);
}

.kpi-value {
    font-size: 1.85rem;
    font-weight: 800;
    color: #0C29D0;
    line-height: 1.15;
    margin-bottom: 0.3rem;
    letter-spacing: -0.02em;
}

.kpi-label {
    font-size: 0.85rem;
    font-weight: 500;
    color: #475569;
}

.chart-title {
    font-size: 1.02rem;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 0.2rem;
}

.chart-subtitle {
    font-size: 0.8rem;
    color: #64748B;
    margin-bottom: 0.6rem;
}
</style>
""", unsafe_allow_html=True)

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#CBD5E1"
plt.rcParams["axes.linewidth"] = 0.8

OLIST_BLUE = "#0C29D0"
OLIST_MID_BLUE = "#3B82F6"
OLIST_LIGHT_BLUE = "#93C5FD"
ALERT_RED = "#DC2626"


# HELPER FUNCTIONS
def format_compact_currency(val):
    if val >= 1e6:
        return f"R${val / 1e6:.2f}M"
    elif val >= 1e3:
        return f"R${val / 1e3:.2f}K"
    return f"R${val:,.0f}"


def format_compact_number(val):
    if val >= 1e6:
        return f"{val / 1e6:.2f}M"
    elif val >= 1e3:
        return f"{val / 1e3:.2f}K"
    return f"{val:,}"


def create_monthly_orders_df(df):
    temp = df.copy()
    temp["year_month"] = temp["order_purchase_timestamp"].dt.to_period("M").astype(str)
    monthly_orders_df = temp.groupby("year_month", as_index=False).agg(
        order_count=("order_id", "nunique"),
        revenue=("price", "sum")
    ).sort_values("year_month")
    return monthly_orders_df


def create_category_perf_df(df):
    total_gmv = df["price"].sum()
    category_df = df.groupby("product_category_name_english", as_index=False).agg(
        total_orders=("order_id", "nunique"),
        total_items=("order_item_id", "count"),
        total_gmv=("price", "sum")
    ).sort_values("total_gmv", ascending=False)
    category_df["gmv_pct"] = (category_df["total_gmv"] / total_gmv * 100).round(2) if total_gmv > 0 else 0
    return category_df


def create_delivery_review_df(df):
    orders_unique = df.drop_duplicates(subset=["order_id"]).copy()
    delivery_review_df = orders_unique.groupby("delivery_status", as_index=False).agg(
        order_count=("order_id", "nunique"),
        mean_review_score=("review_score", "mean")
    )
    total_ord = len(orders_unique)
    delivery_review_df["order_pct"] = (delivery_review_df["order_count"] / total_ord * 100).round(2) if total_ord > 0 else 0
    return delivery_review_df


def create_state_delivery_df(df):
    orders_unique = df.drop_duplicates(subset=["order_id"]).copy()
    state_df = orders_unique.groupby("state_full", as_index=False).agg(
        total_orders=("order_id", "nunique"),
        late_orders=("delivery_status", lambda x: (x == "Late").sum()),
        mean_review_score=("review_score", "mean")
    )
    state_df["late_rate_pct"] = (state_df["late_orders"] / state_df["total_orders"] * 100).round(2)
    return state_df


def create_rfm_df(df):
    rfm_df = df.groupby("customer_unique_id", as_index=False).agg(
        max_order_timestamp=("order_purchase_timestamp", "max"),
        frequency=("order_id", "nunique"),
        monetary=("price", "sum")
    )

    recent_date = df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    rfm_df["recency"] = (recent_date - rfm_df["max_order_timestamp"]).dt.days
    rfm_df.drop("max_order_timestamp", axis=1, inplace=True)

    rfm_df["r_score"] = pd.qcut(rfm_df["recency"].rank(method="first"), q=4, labels=[4, 3, 2, 1]).astype(int)
    rfm_df["f_score"] = np.select(
        [rfm_df["frequency"] == 1, rfm_df["frequency"] == 2, rfm_df["frequency"] == 3, rfm_df["frequency"] >= 4],
        [1, 2, 3, 4],
        default=1
    )
    rfm_df["m_score"] = pd.qcut(rfm_df["monetary"].rank(method="first"), q=4, labels=[1, 2, 3, 4]).astype(int)

    def assign_segment(row):
        r, f, m = row["r_score"], row["f_score"], row["m_score"]
        if r >= 3 and f >= 2 and m >= 3:
            return "Champions"
        elif f >= 2:
            return "Loyal Customers"
        elif r >= 3 and m >= 3:
            return "Potential Loyalists"
        elif r >= 3 and m <= 2:
            return "Promising Customers"
        else:
            return "At Risk / Hibernating"

    rfm_df["rfm_segment"] = rfm_df.apply(assign_segment, axis=1)

    rfm_summary_df = rfm_df.groupby("rfm_segment", as_index=False).agg(
        customer_count=("customer_unique_id", "count"),
        avg_monetary=("monetary", "mean"),
        total_gmv=("monetary", "sum")
    ).sort_values("total_gmv", ascending=False)

    rfm_summary_df["customer_pct"] = (rfm_summary_df["customer_count"] / len(rfm_df) * 100).round(2)
    rfm_summary_df["gmv_pct"] = (rfm_summary_df["total_gmv"] / rfm_summary_df["total_gmv"].sum() * 100).round(2)
    return rfm_df, rfm_summary_df


# LOAD DATA
@st.cache_data
def load_cleaned_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "main_data.csv")
    df = pd.read_csv(csv_path)

    datetime_cols = ["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    if "delivery_status" not in df.columns and {"order_delivered_customer_date", "order_estimated_delivery_date"}.issubset(df.columns):
        df["delivery_status"] = np.where(
            df["order_delivered_customer_date"] <= df["order_estimated_delivery_date"],
            "On Time",
            "Late"
        )

    # Konversi kode singkatan negara bagian menjadi nama lengkap (contoh: SP -> São Paulo (SP))
    df["state_full"] = df["customer_state"].map(
        lambda code: f"{BRAZIL_STATES.get(code, code)} ({code})" if pd.notna(code) else "Unknown"
    )
    return df


all_df = load_cleaned_data()

# SIDEBAR DENGAN LOGO OLIST & FILTER
min_date = all_df["order_purchase_timestamp"].min().date()
max_date = all_df["order_purchase_timestamp"].max().date()

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 0.6rem 0 1.2rem 0;">
        <svg width="150" height="52" viewBox="0 0 180 60" xmlns="http://www.w3.org/2000/svg">
            <rect width="180" height="60" rx="12" fill="#EFF6FF"/>
            <text x="50%" y="66%" dominant-baseline="middle" text-anchor="middle"
                  font-family="Inter, Arial, sans-serif" font-size="38" font-weight="800"
                  letter-spacing="-1.5" fill="#0C29D0">olist</text>
            <circle cx="139" cy="18" r="4.5" fill="#3B82F6"/>
        </svg>
        <div style="font-size: 0.78rem; color: #64748B; margin-top: 0.35rem; font-weight: 500;">
            Brazilian E-Commerce Analytics
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Filter Data Transaksi")

    date_input = st.date_input(
        label="Rentang Waktu Pembelian",
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

    state_options = sorted(all_df["state_full"].dropna().unique().tolist())
    selected_states = st.multiselect(
        label="Negara Bagian Pelanggan",
        options=state_options,
        default=[],
        placeholder="Semua Negara Bagian"
    )

if len(date_input) == 2:
    start_date, end_date = date_input
    main_df = all_df[
        (all_df["order_purchase_timestamp"].dt.date >= start_date) &
        (all_df["order_purchase_timestamp"].dt.date <= end_date)
    ]
else:
    start_date, end_date = min_date, max_date
    main_df = all_df.copy()

if selected_states:
    main_df = main_df[main_df["state_full"].isin(selected_states)]

if main_df.empty:
    st.warning("Tidak ada data transaksi pada rentang tanggal atau negara bagian yang dipilih.")
    st.stop()

# Menyiapkan dataframe agregasi
monthly_orders_df = create_monthly_orders_df(main_df)
category_perf_df = create_category_perf_df(main_df)
delivery_review_df = create_delivery_review_df(main_df)
state_delivery_df = create_state_delivery_df(main_df)
rfm_df, rfm_summary_df = create_rfm_df(main_df)

# STICKY HEADER 
period_str = f"{start_date.strftime('%b %Y')} – {end_date.strftime('%b %Y')}"
state_label = f" | Region: {', '.join(selected_states)}" if selected_states else " | All Regions (Brazil)"

st.markdown(f"""
<div class="sticky-header">
    <h1 class="dashboard-title">Olist E-Commerce <span>Executive Dashboard</span></h1>
    <p class="reporting-period">Reporting Period: {period_str}{state_label}</p>
</div>
""", unsafe_allow_html=True)

# KPI CARDS 
total_orders = monthly_orders_df["order_count"].sum()
total_gmv = monthly_orders_df["revenue"].sum()
late_pct = delivery_review_df.loc[delivery_review_df["delivery_status"] == "Late", "order_pct"].sum()
avg_review = main_df.drop_duplicates("order_id")["review_score"].mean()

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{format_compact_currency(total_gmv)}</div>
        <div class="kpi-label">Total Revenue (GMV)</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{format_compact_number(total_orders)}</div>
        <div class="kpi-label">Total Completed Orders</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{avg_review:.2f} / 5.0</div>
        <div class="kpi-label">Average Review Score</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{late_pct:.2f}%</div>
        <div class="kpi-label">Late Delivery Rate</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 1.1rem;'></div>", unsafe_allow_html=True)

# REVENUE & ORDER TREND + TOP 5 CATEGORIES
r2_col1, r2_col2 = st.columns([1.35, 1])

with r2_col1:
    with st.container(border=True):
        st.markdown('<div class="chart-title">Revenue (GMV) & Order Volume Trend</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-subtitle">Pertumbuhan bulanan pendapatan (R$) dan jumlah pesanan unik</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(9.5, 4.3), facecolor="white")
        ax.set_facecolor("white")
        ax.plot(
            monthly_orders_df["year_month"],
            monthly_orders_df["revenue"] / 1e6,
            marker="o",
            markersize=5,
            linewidth=2.4,
            color=OLIST_BLUE,
            label="GMV (Juta R$)"
        )
        ax.fill_between(monthly_orders_df["year_month"], monthly_orders_df["revenue"] / 1e6, color=OLIST_BLUE, alpha=0.08)
        ax.set_ylabel("GMV (Juta R$)", color=OLIST_BLUE, fontsize=10, fontweight="bold")
        ax.tick_params(axis="y", labelcolor=OLIST_BLUE, labelsize=9)
        ax.tick_params(axis="x", rotation=45, labelsize=8.5)
        ax.spines["top"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4, color="#CBD5E1")

        ax_twin = ax.twinx()
        ax_twin.plot(
            monthly_orders_df["year_month"],
            monthly_orders_df["order_count"],
            marker="s",
            markersize=4,
            linestyle="--",
            linewidth=1.8,
            color=OLIST_MID_BLUE,
            label="Jumlah Pesanan"
        )
        ax_twin.set_ylabel("Jumlah Pesanan", color=OLIST_MID_BLUE, fontsize=10, fontweight="bold")
        ax_twin.tick_params(axis="y", labelcolor=OLIST_MID_BLUE, labelsize=9)
        ax_twin.spines["top"].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig)

with r2_col2:
    with st.container(border=True):
        st.markdown('<div class="chart-title">Top 5 Revenue-Generating Categories</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-subtitle">Kategori produk dengan kontribusi GMV terbesar</div>', unsafe_allow_html=True)

        top5_cat = category_perf_df.head(5).sort_values("total_gmv", ascending=True)
        fig, ax = plt.subplots(figsize=(7.2, 4.3), facecolor="white")
        ax.set_facecolor("white")
        colors_cat = [OLIST_LIGHT_BLUE] * (len(top5_cat) - 1) + [OLIST_BLUE]

        bars = ax.barh(top5_cat["product_category_name_english"], top5_cat["total_gmv"] / 1e6, color=colors_cat, height=0.58)
        ax.set_xlabel("Total GMV (Juta R$)", fontsize=10, fontweight="bold", color="#334155")
        ax.set_xlim(0, top5_cat["total_gmv"].max() / 1e6 * 1.32)
        ax.tick_params(axis="both", labelsize=9.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", linestyle="--", alpha=0.4, color="#CBD5E1")

        for bar, (_, row) in zip(bars, top5_cat.iterrows()):
            ax.text(
                bar.get_width() + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"R${row['total_gmv']/1e6:.2f}M ({row['gmv_pct']}%)",
                va="center", fontsize=9, fontweight="bold", color="#0F172A"
            )
        plt.tight_layout()
        st.pyplot(fig)

# ROW 3: DELIVERY LATENCY IMPACT & STATE LATE RATE
r3_col1, r3_col2 = st.columns([1, 1.35])

with r3_col1:
    with st.container(border=True):
        st.markdown('<div class="chart-title">Delivery Status vs Customer Review</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-subtitle">Perbandingan skor ulasan antara pengiriman tepat waktu dan terlambat</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(7, 4.2), facecolor="white")
        ax.set_facecolor("white")
        status_order = [s for s in ["On Time", "Late"] if s in delivery_review_df["delivery_status"].values]
        plot_rev = delivery_review_df.set_index("delivery_status").loc[status_order].reset_index()
        bar_colors = [OLIST_BLUE if s == "On Time" else ALERT_RED for s in plot_rev["delivery_status"]]

        bars = ax.bar(plot_rev["delivery_status"], plot_rev["mean_review_score"], color=bar_colors, width=0.45)
        ax.set_ylim(0, 5.3)
        ax.set_ylabel("Rata-Rata Skor Ulasan (1–5)", fontsize=10, fontweight="bold", color="#334155")
        ax.tick_params(axis="both", labelsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4, color="#CBD5E1")

        for bar, (_, row) in zip(bars, plot_rev.iterrows()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.12,
                f"{row['mean_review_score']:.2f} / 5.0\n({row['order_pct']}%)",
                ha="center", fontsize=9.5, fontweight="bold", color="#0F172A"
            )
        plt.tight_layout()
        st.pyplot(fig)

with r3_col2:
    with st.container(border=True):
        st.markdown('<div class="chart-title">Top 10 States by Late Delivery Rate</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-subtitle">Wilayah dengan tingkat keterlambatan pengiriman tertinggi (%)</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(9.5, 4.2), facecolor="white")
        ax.set_facecolor("white")
        top10_states = state_delivery_df.sort_values("late_rate_pct", ascending=False).head(10).sort_values("late_rate_pct", ascending=True)
        state_colors = [OLIST_MID_BLUE if pct < late_pct * 2 else OLIST_BLUE for pct in top10_states["late_rate_pct"]]

        bars = ax.barh(top10_states["state_full"], top10_states["late_rate_pct"], color=state_colors, height=0.6)
        ax.axvline(late_pct, color=ALERT_RED, linestyle="--", linewidth=1.3, label=f"Avg Late Rate ({late_pct:.2f}%)")
        ax.set_xlabel("Persentase Pesanan Terlambat (%)", fontsize=10, fontweight="bold", color="#334155")
        ax.set_xlim(0, max(top10_states["late_rate_pct"].max() * 1.22, 10))
        ax.tick_params(axis="both", labelsize=9.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", linestyle="--", alpha=0.4, color="#CBD5E1")
        ax.legend(loc="lower right", frameon=True, facecolor="white")

        for bar, (_, row) in zip(bars, top10_states.iterrows()):
            ax.text(
                bar.get_width() + 0.35,
                bar.get_y() + bar.get_height() / 2,
                f"{row['late_rate_pct']:.2f}%",
                va="center", fontsize=9, fontweight="bold", color="#0F172A"
            )
        plt.tight_layout()
        st.pyplot(fig)

# RFM CUSTOMER SEGMENTATION
r4_col1, r4_col2 = st.columns(2)
rfm_plot = rfm_summary_df.sort_values("total_gmv", ascending=True)
rfm_colors = [OLIST_BLUE if s in ["Champions", "Loyal Customers"] else OLIST_LIGHT_BLUE for s in rfm_plot["rfm_segment"]]

with r4_col1:
    with st.container(border=True):
        st.markdown('<div class="chart-title">Customer Distribution by RFM Segment</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="chart-subtitle">Avg Recency: {rfm_df["recency"].mean():.1f} hari | Avg Frequency: {rfm_df["frequency"].mean():.2f}x</div>',
            unsafe_allow_html=True
        )

        fig, ax = plt.subplots(figsize=(8, 4.1), facecolor="white")
        ax.set_facecolor("white")
        bars = ax.barh(rfm_plot["rfm_segment"], rfm_plot["customer_pct"], color=rfm_colors, height=0.56)
        ax.set_xlabel("Proporsi Pelanggan (%)", fontsize=10, fontweight="bold", color="#334155")
        ax.set_xlim(0, rfm_plot["customer_pct"].max() * 1.32)
        ax.tick_params(axis="both", labelsize=9.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", linestyle="--", alpha=0.4, color="#CBD5E1")

        for bar, (_, row) in zip(bars, rfm_plot.iterrows()):
            ax.text(
                bar.get_width() + 0.6,
                bar.get_y() + bar.get_height() / 2,
                f"{row['customer_pct']:.2f}% ({format_compact_number(row['customer_count'])})",
                va="center", fontsize=9, fontweight="bold", color="#0F172A"
            )
        plt.tight_layout()
        st.pyplot(fig)

with r4_col2:
    with st.container(border=True):
        st.markdown('<div class="chart-title">Revenue (GMV) Contribution by RFM Segment</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="chart-subtitle">Avg Monetary per Pelanggan: R${rfm_df["monetary"].mean():.2f}</div>',
            unsafe_allow_html=True
        )

        fig, ax = plt.subplots(figsize=(8, 4.1), facecolor="white")
        ax.set_facecolor("white")
        bars = ax.barh(rfm_plot["rfm_segment"], rfm_plot["gmv_pct"], color=rfm_colors, height=0.56)
        ax.set_xlabel("Kontribusi GMV (%)", fontsize=10, fontweight="bold", color="#334155")
        ax.set_xlim(0, rfm_plot["gmv_pct"].max() * 1.34)
        ax.tick_params(axis="both", labelsize=9.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", linestyle="--", alpha=0.4, color="#CBD5E1")

        for bar, (_, row) in zip(bars, rfm_plot.iterrows()):
            ax.text(
                bar.get_width() + 0.6,
                bar.get_y() + bar.get_height() / 2,
                f"{row['gmv_pct']:.2f}% (Avg R${row['avg_monetary']:.0f})",
                va="center", fontsize=9, fontweight="bold", color="#0F172A"
            )
        plt.tight_layout()
        st.pyplot(fig)

st.caption("Copyright © Setiti Budi Utami | Brazilian E-Commerce Public Dataset by Olist")