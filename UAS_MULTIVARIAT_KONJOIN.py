import streamlit as st
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import numpy as np

st.markdown(
    """
    <style>
        .stApp {
            background-color: white !important;
        }
        body, .stMarkdown, .stTextInput, .stButton, .stSelectbox, .stSlider, .stDataFrame, .stTable {
            color: black !important;
        }
        h1, h2, h3, h4, h5, h6, .stHeader, .stSubheader {
            color: black !important;
        }
    </style>
    """,
    unsafe_allow_html=True)

st.title("Analisis Conjoint of Data Wine")

st.subheader("Analisis Conjoint")
st.markdown("""
            <div style="text-align: justify;">
Analisis Konjoin merupakan sebuah teknik analisis dalam statistika yang digunakan guna mengkombinasikan atribut suatu produk atau jasa yang disukai konsumen sehingga
            mendapat persepsi konsumen terhadap produk atau jasa (HMPS Statistika FMIPA UNM, 2023).
analisis konjoin memiliki kosnep melakukan penentuan nilai kepentingan yang akan dikaitkan pada sebuah atribut dan nilai kegunaan atribut yang digunakan.
            metode analisis ini cukup menarik karena akan menghasilkan output yang erat berkaitan dengan sukses atau tidaknya suatu produk ketika dilakukan launching atau peluncuran
            ke market (MobileStatistik, n.d.)
             </div>
            """,
            unsafe_allow_html=True)

st.subheader("Cari satu dataset yang menarik")
st.markdown("""
            <div style="text-align: justify;">
            dataset yang digunakan pada penerapan analisis konjoin ini adalah dataset preferensi konsumen terhadap wine, dimana tiap responden 
            memilih di antara beberapa pilihan wine yang tersedia pada opsi.
            </div>
            """,  unsafe_allow_html=True)

st.header("Data: Preferensi Konsumen Terhadap Wine")
data_url = "https://github.com/KSwaviman/Conjoint-Analysis/tree/main"
url = "https://raw.githubusercontent.com/naaufald/UAS_MULTIVARIAT_CONJOINT/main/conjoint_survey_resp_v1.csv"
df = pd.read_csv(url)
st.dataframe(df)
st.markdown("Sumber dataset: [Dataset Wine](https://github.com/KSwaviman/Conjoint-Analysis/tree/main)")

st.subheader("Mengapa dataset tersebut cocok untuk dianalisis dengan metode conjoint?")
st.markdown("""
        <div style="text-align: justify;">
        Dataset ini cocok digunakan karena telah berisi preferensi konsumen terhadap berbagai pilihan produk, 
        hal ini sejalan dengan studi kasus yang akan dipecahkan, yaitu menyajikan berbagai kombinasi produk (profil) 
        dan meminta konsumen untuk membuat pilihan sehingga dari pola pilihan tersebut, 
        dapat melakukan dekomposisi nilai total sebuah produk menjadi nilai utilitas (part-worth) dari masing-masing atribut yang menyusunnya.
            </div>
            """, unsafe_allow_html=True)

st.header("Pembersihan dan Persiapan data")
st.subheader("Missing Values")
st.write(df.isnull().sum())

st.subheader("Pembersihan Missing Values")

# Hapus missing di kolom physical_activities
df['physical_activities'] = df['Physical Activities'].replace(['', ' ', 'NA'], pd.NA)
df = df.dropna(subset=['Physical Activities'])
st.dataframe(df)

st.subheader("Cek Atribut")
st.markdown(""" atribut yang digunakan adalah sebagai berikut: """)
atribut = [
    'Price',
    'Brand',
    'Type of Wine',
    'Percentage of Alcohol',
    'Aging time of Wine',
]
attributes = {col: df[col].dropna().unique().tolist() for col in atribut}
st.write("Level tiap atribut untuk conjoint:")
st.dataframe(attributes)
# Generate semua kombinasi profil produk
combinations = list(itertools.product(*attributes.values()))
df_profiles = pd.DataFrame(combinations, columns=attributes.keys())
st.write(f"Total Profiles: {len(df_profiles)}")
st.dataframe(df_profiles)

st.subheader("encoding")
df2 = df.copy()
for col in atribut:
    df2[col] = df2[col].astype(str)
df_encoded = pd.get_dummies(
    df2,
    columns=atribut,
    drop_first=True,
    dtype=bool)
df_display = df_encoded.copy()
bool_cols = df_display.select_dtypes(include='bool').columns

for col in bool_cols:
    df_display[col] = df_display[col].astype(str)

st.write("Encoded:")
st.dataframe(df_display)

st.markdown("""Pada tahap ini, setiap kombinasi atribut produk yang telah dibentuk masih berupa
variabel kategorikal. Agar dapat digunakan dalam analisis conjoint berbasis regresi,
variabel-variabel tersebut dikonversi ke dalam bentuk numerik melalui proses dummy encoding""")


st.header("Model Training: Binary Logit untuk Conjoint")
st.markdown("""model yang akan digunakan merupakan model binary logit guna menganalisis preferensi konsumen dalam choices-based conjoint.""")

atribut = [
    'Price',
    'Brand',
    'Type of Wine',
    'Percentage of Alcohol',
    'Aging time of Wine'
]
df2 = df.copy()
for col in atribut:
    df2[col] = df2[col].astype(str)
X = pd.get_dummies(
    df2[atribut],
    drop_first=True
)
X = X.astype(float)
X = sm.add_constant(X)
y = df2['Choice'].astype(int)
model = sm.Logit(y, X)
result = model.fit()

code = r"""
atribut = [
    'Price',
    'Brand',
    'Type of Wine',
    'Percentage of Alcohol',
    'Aging time of Wine'
]
df2 = df.copy()
for col in atribut:
    df2[col] = df2[col].astype(str)
X = pd.get_dummies(
    df2[atribut],
    drop_first=True
)
X = X.astype(float)
X = sm.add_constant(X)
y = df2['Choice'].astype(int)
model = sm.Logit(y, X)
result = model.fit()"""

st.code(code, language="python")

params = result.params.drop("const")

partworth = {}

for attr in atribut:
    levels = df2[attr].unique()
    levels = sorted(levels)

    pw_attr = {}

    for lvl in levels:
        dummy_col = f"{attr}_{lvl}"
        if dummy_col in params.index:
            pw_attr[lvl] = params[dummy_col]
        else:
            pw_attr[lvl] = 0.0  # baseline level

    partworth[attr] = pw_attr

code = r"""
params = result.params.drop("const")

partworth = {}

for attr in atribut:
    levels = df2[attr].unique()
    levels = sorted(levels)

    pw_attr = {}

    for lvl in levels:
        dummy_col = f"{attr}_{lvl}"
        if dummy_col in params.index:
            pw_attr[lvl] = params[dummy_col]
        else:
            pw_attr[lvl] = 0.0  # baseline level

    partworth[attr] = pw_attr"""

st.code(code, language="python")

# DataFrame part-worth
pw_rows = []
for attr, levels in partworth.items():
    for lvl, util in levels.items():
        pw_rows.append({
            "Attribute": attr,
            "Level": lvl,
            "Utility": util
        })
pw_df = pd.DataFrame(pw_rows)

# =============================
# 5. Relative Importance
# =============================
importance = {}
for attr, levels in partworth.items():
    vals = np.array(list(levels.values()))
    importance[attr] = vals.max() - vals.min()

total_range = sum(importance.values())
importance_pct = {k: (v/total_range * 100) for k, v in importance.items()}

imp_df = pd.DataFrame({
    "Attribute": list(importance_pct.keys()),
    "Relative Importance (%)": list(importance_pct.values())
})

# =============================
# 6. OUTPUT (untuk Streamlit)
# =============================
st.subheader("Hasil Model Binary Logit")
st.markdown("```text\n" + result.summary().as_text() + "\n```")

st.subheader("Part-Worth Utilities")
st.dataframe(pw_df)

code = r"""
pw_rows = []
for attr, levels in partworth.items():
    for lvl, util in levels.items():
        pw_rows.append({
            "Attribute": attr,
            "Level": lvl,
            "Utility": util
        })
pw_df = pd.DataFrame(pw_rows)
"""
st.code(code, language='python')

st.subheader("Relative Importance (%)")
st.dataframe(imp_df)

code = r"""
importance = {}
for attr, levels in partworth.items():
    vals = np.array(list(levels.values()))
    importance[attr] = vals.max() - vals.min()

total_range = sum(importance.values())
importance_pct = {k: (v/total_range * 100) for k, v in importance.items()}

imp_df = pd.DataFrame({
    "Attribute": list(importance_pct.keys()),
    "Relative Importance (%)": list(importance_pct.values())
})"""
st.code(code, language='python')

st.subheader("Visualisasi Part-Worth Utilities")

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(
    pw_df["Level"],
    pw_df["Utility"]
)
ax.set_title("Part-Worth Utilities (PWU)")
ax.set_xlabel("Utility")
ax.set_ylabel("Attribute Level")
plt.tight_layout()
st.pyplot(fig)

st.markdown("""
            <div style="text-align: justify;">
Berdasarkan hasil visualisasi nilai utilitas parsial (Part-Worth Utilities), dapat disimpulkan bahwa konsumen menunjukkan preferensi tertentu dalam memilih wine. Dari aspek harga, konsumen lebih cenderung memilih wine dengan kategori harga menengah. 
Dari sisi merek, konsumen lebih menyukai produk yang sudah dikenal atau sering dibeli sebelumnya.
Pada atribut jenis wine, white wine dan sparkling wine menjadi pilihan yang paling diminati dibandingkan jenis lainnya.
Untuk kadar alkohol, preferensi tertinggi ditunjukkan pada kadar 5.5%, diikuti oleh 7% dan 18%. Sementara itu, pada atribut lama penyimpanan (aging), konsumen paling menyukai wine yang telah disimpan selama 2 tahun, dibandingkan dengan durasi penyimpanan yang lebih lama. </div>
            """, unsafe_allow_html=True)

st.subheader("Visualisasi Relative Importance (%)")
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(
    imp_df["Attribute"],
    imp_df["Relative Importance (%)"]
)
ax.set_title("Relative Importance of Attributes")
ax.set_ylabel("Relative Importance (%)")
ax.set_xlabel("Attribute")
plt.xticks(rotation=45)
st.pyplot(fig)

st.markdown("""
<div style="text-align: justify;">
Berdasarkan hasil relative importance, diketahui bahwa keputusan konsumen dalam membeli wine paling dominan dipengaruhi oleh harga, dengan kontribusi sebesar 50.51%. 
            Temuan ini menunjukkan bahwa konsumen sangat sensitif terhadap harga dan menjadikannya faktor utama dalam proses pertimbangan pembelian.
Atribut brand berada pada urutan kedua dengan kontribusi 18.97%, yang mengindikasikan bahwa reputasi serta tingkat kepercayaan terhadap merek turut memengaruhi preferensi konsumen. 
            Selanjutnya, lama penyimpanan wine memberikan pengaruh sebesar 11.16%. dan Atribut tipe wine menyumbang pengaruh sebesar 10.09% dalam keputusan pembelian, menunjukkan bahwa preferensi terhadap jenis wine (misalnya white, red, rosé, atau sparkling) 
            tetap diperhatikan meskipun tidak menjadi faktor dominan. Terakhir, persentase alkohol memiliki pengaruh paling kecil, yaitu 9%, menunjukkan bahwa kadar alkohol bukan menjadi pertimbangan utama bagi konsumen ketika memilih wine.
            </div>""", unsafe_allow_html=True)

st.header("Produk Ideal")
st.markdown("""
<div style="text-align: justify;">
            berdasarkan Part-Worth Utilities yang dihasilkan, dapat ditarik kesimpulan bahwa wine yang diinginkan oleh kebutuhan pasar merupakan wine dengan harga menengah, 
            berasal dari brand yang telah memiliki kepopuleran seperti Mezzacorona atau Ferrari, 
            dengan tipe **White Wine** atau **Sparkling Wine**, kadar alkohol yang berada di angka **5.5%** 
            dan melalui proses penyimpanan sekitar **3 tahun**. </div>""", unsafe_allow_html=True)
st.header("Rekomendasi Strategis")
st.markdown("""
- strategi penerapan harga yang menjadi pokok utama
- brand identity, dan brand positioning
- kembangkan tipe White Wine atau Sparkling Wine dengan kadar alkohol 5.5% dan proses penyimpanan 2 tahun""")

st.header("Referensi")
st.write("HMPS Statistika FMIPA UNM. (2023). Analisis konjoin. Diperoleh dari https://hmpsstatistikafmipaunm.com/2023/06/17/analisis-konjoin/. Date Accessed 8 Desember 2025" \
"MobileStatistik (n.d.). Mengenal Konsep Analisis Konjoin (Conjoint Analysis). Diperoleh dari https://www.mobilestatistik.com/analisis-konjoin/. Date Accessed 8 Desember 2025.")
