# Latar Belakang
Home Credit saat ini sedang menggunakan berbagai macam metode statistik dan Machine Learning untuk membuat prediksi skor kredit.Dengan menggunakan data-data yang telah disedikan diharapkan dapat memastikan pelanggan yang mampu melakukan pelunasan tidak ditolak ketika melakukan pengajuan pinjaman, dan pinjaman dapat diberikan dengan principal, maturity, dan repayment calendar yang akan memotivasi pelanggan untuk sukses.
# Data Pre-Processing
- Membaca data yang akan digunakan
`````Python
app_test = pd.read_csv("application_test.csv")
app_train = pd.read_csv("application_train.csv")
`````
- Menentukan fitur-fitur yang akan digunakan
`````Python
app_train_used=app_train[['TARGET', 'NAME_CONTRACT_TYPE', 'CODE_GENDER',
       'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',
       'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
       'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
       'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
       'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'CNT_FAM_MEMBERS']]
`````
`````Python
app_test_used=app_test[['NAME_CONTRACT_TYPE', 'CODE_GENDER',
       'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN',
       'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
       'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
       'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
       'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'CNT_FAM_MEMBERS']]
`````
- Melakukan pemeriksaan missing value
`````Python
app_train_used.isna().sum()
app_test_used.isna().sum()
`````
- Menghapus missing value
`````Python
app_train_used= app_train_used.dropna()
app_test_used = app_test_used.dropna()
`````
- Memeriksa adanya data duplikat
`````Python
print("Jumlah duplikasi: ",app_train_used.duplicated().sum())
print("Jumlah duplikasi: ",app_test_used.duplicated().sum())
`````
- Menghapus data duplikat
`````Python
app_train_used.drop_duplicates(inplace=True)
app_test_used.drop_duplicates(inplace=True)
`````
- Mengecek ejaan kata dan menemukan kesalahan pada kolom CODE_GENDER dan NAME_INCOME_TYPE, yang kemudian dihapus
`````Python
print(app_train_used['TARGET'].unique(),'\n',
    app_train_used['NAME_CONTRACT_TYPE'].unique(),'\n',
      app_train_used['CODE_GENDER'].unique(),'\n',
      app_train_used['FLAG_OWN_CAR'].unique(),'\n',
      app_train_used['FLAG_OWN_REALTY'].unique(),'\n',
      app_train_used['NAME_TYPE_SUITE'].unique(),'\n',
      app_train_used['NAME_INCOME_TYPE'].unique(),'\n',
      app_train_used['NAME_EDUCATION_TYPE'].unique(),'\n',
      app_train_used['NAME_FAMILY_STATUS'].unique(),'\n',
      app_train_used['NAME_HOUSING_TYPE'].unique())
`````
`````Python
app_train_used.drop(app_train_used.index[app_train_used['CODE_GENDER']=='XNA'],inplace=True)
app_train_used.drop(app_train_used.index[app_train_used['NAME_INCOME_TYPE']=='Maternity leave'],inplace=True)
`````
- Menambahkan kolom AGE
`````Python
train_age=(app_train_used['DAYS_BIRTH']/-365).astype(int)
app_train_used=app_train_used.assign(AGE=train_age).drop('DAYS_BIRTH',axis=1)
app_train_used.head()
`````
`````Python
Test_age =(app_test_used['DAYS_BIRTH']/-365).astype(int)
app_test_used=app_test_used.assign(AGE=Test_age).drop('DAYS_BIRTH',axis=1)
app_test_used.head()
`````
- Mengubah data kategorikal menjadi biner
`````Python
l = LabelEncoder()
for q in app_test_used.describe(include='object').columns:
    app_test_used[q]=l.fit_transform(app_test_used[q])
app_test_used.head(3)
`````
`````Python
l = LabelEncoder()
for p in app_train_used.describe(include='object').columns:
    app_train_used[p]=l.fit_transform(app_train_used[p])
app_train_used.head(3)
`````
- Membagi dataset menjadi 80% data training dan 20% data testing
`````Python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)
`````
# Data Visualization and Business Insight
