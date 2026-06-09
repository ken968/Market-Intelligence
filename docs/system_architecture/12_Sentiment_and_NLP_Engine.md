# 12. Sentiment and NLP Engine Architecture
Status: FINAL TECHNICAL SPECIFICATION
Date: 2026-06-08

## 1. Overview
Modul *Sentiment and NLP Engine* bertugas mengonversi unstructured text data (berita makro, RSS feeds, X/Twitter) menjadi *structured quantitative tensors* yang siap diinjeksikan ke dalam LSTM dan XGBoost Stacker. Modul ini dimigrasikan dari *rule-based lexicon* (VADER/TextBlob) menjadi arsitektur Deep Learning (Transformer).

## 2. FinBERT Core Architecture
Sistem tidak menggunakan model NLP generik. Sistem mengimplementasikan `ProsusAI/finbert`, sebuah turunan dari arsitektur BERT (Bidirectional Encoder Representations from Transformers) yang di-fine-tune secara spesifik pada dataset TRC2 (Thomson Reuters Text Research Collection) dan Financial PhraseBank.

### 2.1. Tokenization & Forward Pass
- **Input**: Kumpulan headline `H = {h_1, h_2, ..., h_n}`
- **Tokenization**: BertTokenizer(vocab_size=30522) memecah teks menjadi token id, attention mask, dan token type id.
- **Model Output**: Logits layer terakhir (Dense) menghasilkan vektor $[l_p, l_{neg}, l_{neu}]$ untuk setiap headline.
- **Softmax Activation**: Probabilitas dipetakan menjadi:
  $$ P(C=c_i | h) = \frac{\exp(l_i)}{\sum_{j} \exp(l_j)} $$
  Di mana $C \in \{Positive, Negative, Neutral\}$.

### 2.2. Composite Sentiment Formulation
Probabilitas agregat dihitung menjadi nilai skalar tunggal $\mu_S \in [-1, 1]$:
$$ \mu_{S, t} = \frac{1}{N} \sum_{i=1}^{N} \left[ P(Positive | h_i) - P(Negative | h_i) \right] $$
Kelebihan utama: Headline yang diklasifikasikan sebagai *Neutral* dengan probabilitas tinggi tidak akan mempengaruhi nilai ekspektasi secara drastis.

## 3. Z-Score Normalization (Regime Contextualization)
Nilai $\mu_{S, t}$ mentah memiliki masalah autokorelasi (biasa terjadi pada rezim *bear market* dimana berita selalu negatif secara persisten). Model Machine Learning tidak butuh nilai absolut, melainkan divergensi dari tren.

### 3.1. Rolling Z-Score Computation
$$ Z_{S, t} = \frac{\mu_{S, t} - E_{30}[\mu_S]}{\sqrt{Var_{30}(\mu_S) + \epsilon}} $$
- $E_{30}$ adalah *moving average* 30 hari.
- $\epsilon = 1e-6$ untuk mencegah *division by zero* pada hari libur pasar.
- **Output**: Vektor $Z_S \sim N(0, 1)$. Nilai $|Z_S| > 1.96$ menandakan *statistically significant sentiment shock* (95% CI).

## 4. Sentiment Volatility (Uncertainty Proxy)
Alih-alih membuang varians dari agregasi harian, sistem mengekstrak standar deviasi sentimen intra-day sebagai sinyal volatilitas independen:
$$ \sigma_{S, t} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (S_i - \mu_{S, t})^2} $$
- **Tujuan**: $\sigma_{S, t}$ tinggi mengindikasikan bahwa hari tersebut memuat berita yang sangat polarisasi (misal: "Fed menaikkan suku bunga" vs "Data pengangguran membaik").
- **Injection**: $\sigma_{S, t}$ digunakan sebagai fitur independen dalam `manager_anchor.py` untuk melebarkan *confidence interval* dari probabilitas klasifikasi.

## 5. Tensor Injection Workflow
- Di `scripts/sentiment_fetcher_v2.py`: Teks di-*fetch*, di-*inference* via CPU/GPU, diagregasi, dan disimpan sebagai `Sentiment` dan `Sentiment_Std` di DuckDB `macro_indicators`.
- Di `utils/layers/worker_lstm.py`: Vektor ini masuk sebagai matriks input $X_t$, di mana layer `MultiHeadAttention` akan mengalkulasi Attention Score antar $Z_{S, t}$ dan *Price Momentum*.
