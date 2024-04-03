#!/usr/bin/env python
# coding: utf-8

# # İş Problemi

# Bir oyun şirketi müşterilerinin bazı özelliklerini kullanarak seviye tabanlı(level based) yeni müşteri tanımları(persona) oluşturumak ve bu yeni müşteri tanımlarına göre segmentler oluşturup bu segmentlere göre yeni gelebilecek müşterilerin şirkete ortalama ne kadar kazzndırabileceğini tahmin etmek istemektedir.

# * Price: Müşterinin harcama tutarı
# * Soruce:Müşterinin bağlandığı cihaz türü
# * Sex: Müşterinin cinsiyeti
# * Country: Müşterinin ülkesi
# * Age: Müşterinin yaşı

# In[3]:


#Soru 1: persona.csv dosyasını okutunuz ve veriseti iler ilgili genel bilgileri görüntüleyiniz.
import pandas as pd
pd.set_option("display.max_rows",None)
df=pd.read_csv("persona.csv")


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


# Soru 2:Kaç unique SOURCE vardır? Frekansları nedir?
df["SOURCE"].nunique()
df["SOURCE"].value_counts()


# In[8]:


# Soru 3: Kaç unique PRICE vardır?
df["PRICE"].nunique()


# In[9]:


#Soru 4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?
df["PRICE"].value_counts()
#df.groupby("PRICE").agg({"PRICE":"count"})


# In[10]:


# Soru 5: Hangi ülkeden kaçar tane satış olmuş?
df["COUNTRY"].value_counts()
#df.groupby("COUNTRY")["PRICE"].count()
#df.groupby("COUNTRY")[["PRICE"]].count()
#df.groupby("COUNTRY")["COUNTRY"].count()

#df.pivot_table(values="PRICE",index="COUNTRY",aggfunc="count")


# In[11]:


# Soru 6: Ülkelerie göre satışlardan toplam ne kada kazanılmış?
#df.groupby("COUNTRY")["PRICE"].sum()
#df.groupby("COUNTRY").agg({"PRICE":"sum"})
df.pivot_table(values="PRICE",index=["COUNTRY"],aggfunc="sum")


# In[12]:


# Soru 7: SOURCE türlerine göre satış sayıları nedir?
df["SOURCE"].value_counts()


# In[13]:


# Soru 8: Ülkelere göre PRICE ortalamaları nedir?
df.groupby(["COUNTRY"]).agg({"PRICE":"mean"})


# In[14]:


# Soru 9: SOURCE'lara göre PRICE ortalamları nedşir?
df.groupby(["SOURCE"]).agg({"PRICE":"mean"})


# In[15]:


# Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?
df.groupby(["COUNTRY","SOURCE"]).agg({"PRICE":"mean"})


# In[16]:


# GÖREV 2: COUNTRY, SOURCE , SEX AGE kırılımında ortalama kazançlar nedir?
df.groupby(["COUNTRY","SOURCE","SEX","AGE"]).agg({"PRICE":"mean"}).head()


# In[17]:


# GÖREV 3: Çıktıyı PRICE'a göre sıralayanız.
# Önceki sorudaki çıktıyı daha iyi görebilmek için sort_values metodunu azalan olacak şekilde PRICE'a uygulayınız.
# Çıktıyı agg_df olarak kaydediniz.
agg_df = df.groupby(by=["COUNTRY", 'SOURCE', "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False)
agg_df.head()


# In[18]:


# GÖREV 4: Indekste yer alan isimleri değişken ismine çeviriniz.
# Üçüncü sorunun çıktısında eyr alan PRICE dışındaki tüm değişkenler index isimleridir.
# Bu isimleri değişken isimlerine çeviriniz.
# İpucu: reset_index()
agg_df.reset_index(inplace=True)
# agg_df = agg_df.reset_index()
agg_df.head()


# In[19]:


# GÖREV 5: AGE değişkeninin kategorik değişkenine çeviriniz ve agg_df'e ekleyiniz.
# AGE sayısal değişkenini kategorik değişkenine çeviriniz.
# Örneğin: "0_18"  "19_23" "24_30" "31_40" "41_70"

# AGE değişkeninin nerlerden bölüneceğini karar verinzi.
bins=[0,18,23,30,40,agg_df["AGE"].max()]


# In[20]:


# Bölünen norktalara karşılık gleen isimlendirmelerinin ne olacağını ifade ediniz
mylabels=["0_18" , "19_23" ,"24_30" ,"31_40" ,"41_70" +str(agg_df["AGE"].max())]


# In[21]:


# age'i bölelim
agg_df["age_cat"]=pd.cut(agg_df["AGE"],bins,labels=mylabels)
agg_df.head()


# In[22]:


# Kesişimlerinin gözlerim sayısı 
pd.crosstab(agg_df["AGE"],agg_df["age_cat"])


# In[23]:


# Gorev 6: Yeni level based müşterileri tanımlayınız ve veri setine değişken olaarak ekleyiniz.
# customers_level_based adında bir değişken tanımlayınız ve veri setine bu değişkeni ekleyiniz.
# list comp ile custommers_level_based değerleri oluşturulduktan sonra bu değerlerin tekilleştirilmesi gerekmektedir.
# Örneğin birden fazla şu ifadeden olabilir: USA_ANDROID_MALE_0_18
# Bunlaarı groupby'a alıp price ortalamalarını almak gerekmektedir.


# In[24]:


# YÖNTEM 1
# değişken isimleri:
agg_df.columns


# In[25]:


for row in agg_df.values:
    print(row)


# In[26]:


# COUNTRY,SOURCE,SEX VE age_cat değişkenlerinin değerlerinin yan yana koymak ve alt tireyle birleştirmek istiyoruz.
# Bunu lsit comprehension ile yapabilirz.
# Yukarıdaki döngüde gözlem değerlerinin bize  lazım olanlarını seçeçcek şekilde gerçekleştirelim:

#yontem 1
[row[0].upper() + "_" + row[1].upper() + "_" + row[2].upper() + "_" + row[5].upper() for row in agg_df.values]


# In[27]:


# yontem 2
[row["COUNTRY"].upper() + '_' + row["SOURCE"].upper() + '_' + row["SEX"].upper() + '_' + row["age_cat"].upper() for index, row in agg_df.iterrows()]


# In[29]:


# yontem 4
agg_df[['COUNTRY', 'SOURCE', 'SEX', 'age_cat']].apply(lambda x: '_'.join(x).upper(),axis=1)


# In[30]:


# Veri setine ekleyelim:
agg_df["customers_level_based"] = agg_df[['COUNTRY', 'SOURCE', 'SEX', 'age_cat']].apply(lambda x: '_'.join(x).upper(),axis=1)
agg_df.head()


# In[31]:


# Gereksiz değişkenleri çıkaralım:
agg_df1 = agg_df[["customers_level_based", "PRICE"]]
agg_df1.head()


# In[32]:


# Amacımıza bir adım daha yaklaştık.
# Burada ufak bir problem var. Bir çok aynı segment olacak.
# örneğin USA_ANDROID_MALE_0_18 segmentinden birçok sayıda olabilir.
# kontrol edelim:
agg_df1["customers_level_based"].value_counts()


# In[33]:


# Bu sebeple segmentlere göre groupby yaptıktan sonra price ortalamalarını almalı ve segmentleri tekilleştirmeliyiz.
agg_df1 = agg_df1.groupby("customers_level_based").agg({"PRICE": "mean"})


# In[34]:


# customers_level_based index'te yer almaktadır. Bunu değişkene çevirelim.
agg_df1.reset_index(inplace=True)
agg_df1.head()


# In[35]:


# kontrol edelim. her bir persona'nın 1 tane olmasını bekleriz:
agg_df1["customers_level_based"].value_counts()
agg_df1.head()


# In[36]:


# GÖREV 7: Yeni müşterileri (USA_ANDROID_MALE_0_18) segmentlere ayırınız.
#############################################
# PRICE'a göre segmentlere ayırınız,
# segmentleri "SEGMENT" isimlendirmesi ile agg_df'e ekleyiniz,
# segmentleri betimleyiniz,

agg_df1["SEGMENT"]= pd.qcut(agg_df1["PRICE"], 4, labels=["D", "C", "B", "A"]) #küçükten büyüğe !!!
agg_df1.head(30)


agg_df1.groupby("SEGMENT").agg({"PRICE": ["mean"]})


# In[37]:


# GÖREV 8: Yeni gelen müşterileri sınıflandırınız ne kadar gelir getirebileceğini tahmin ediniz.
#############################################
# 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?
new_user = "TUR_ANDROID_FEMALE_31_40"
agg_df1[agg_df1["customers_level_based"] == new_user]


# 35 yaşında IOS kullanan bir Fransız kadını hangi segmente ve ortalama ne kadar gelir kazandırması beklenir?
new_user2 = "FRA_IOS_FEMALE_31_40"
agg_df1[agg_df1["customers_level_based"] == new_user2]


# In[ ]:




