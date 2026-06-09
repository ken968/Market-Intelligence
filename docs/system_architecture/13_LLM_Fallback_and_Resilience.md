# 13. Multi-LLM Fallback and System Resilience
Status: FINAL TECHNICAL SPECIFICATION
Date: 2026-06-08

## 1. Overview
Sistem tidak dapat berjalan secara optimal tanpa Layer CEO (Contextual Narrative Injection) yang bergantung pada Large Language Models (LLM). Namun, API pihak ketiga memiliki reliabilitas probabilistik (Rate Limits, HTTP 502 Bad Gateway). Dokumen ini menspesifikasikan *State Machine* untuk sistem *Failover* berlapis yang menjamin SLA waktu tayang 99.9% pada algoritma inferensi.

## 2. API Cascade State Machine
Jika fungsi `analyze_news_context()` dalam `utils/llm_manager.py` menemui eksepsi jaringan atau format JSON invalid, ia akan melakukan kaskade eksponensial:

### 2.1. Node 1: Gemini 2.0 Flash / Pro
- **Endpoint**: `google.generativeai.GenerativeModel('gemini-2.0-flash')`
- **Tujuan**: *Context window* masif dan kemampuan penalaran finansial terbaik (Primary Router).
- **Retry Mechanism**: $WaitTime = \min(2^k, 8)$ detik, maks 3 retries (HTTP 429). Jika gagal, sistem berpindah ke Node 2.

### 2.2. Node 2: Deepseek R1 (via Groq & OpenRouter)
- **Endpoint**: `groq.client.chat.completions.create(model='deepseek-r1-distill-llama-70b')`
- **Tujuan**: *Inference latency* sangat rendah (~300 tok/sec) via TPU/LPU Groq. Jika Groq *down*, sistem melakukan rotasi ke URL *base_path* OpenRouter dengan header kredensial `Bearer $OPENROUTER_KEY`.
- **Karakteristik Output**: Membutuhkan *Regex stripping* khusus untuk menghapus `<think>` tags dari Chain-of-Thought sebelum ekstraksi JSON.

### 2.3. Node 3: Meta Llama 3.3 70B
- **Endpoint**: Sama seperti di atas (Groq/OpenRouter).
- **Tujuan**: Standar *open-weights* terbaik. Model ini lebih patuh pada format JSON murni (`response_format={"type": "json_object"}`) meskipun kapabilitas penalaran makronya sedikit di bawah Gemini/Deepseek.

## 3. Quorum Normalization Layer (Cross-LLM Calibration)
Karena perbedaan topologi jaringan neural, LLM A mungkin selalu memberikan nilai rata-rata volatilitas 0.6 sementara LLM B memberikan rata-rata 0.4.

### Algoritma Z-Score Calibration per Provider
Sistem memelihara *State Dictionary* di DuckDB untuk setiap provider LLM:
$$ \mu_{LLM\_A}, \sigma_{LLM\_A} $$

Ketika LLM $i$ merespons dengan nilai raw $V_i$, nilai tersebut dikalibrasi (Z-Score) sebelum dikonversi menjadi multiplier.
$$ Z_{val} = \frac{V_i - \mu_{LLM\_i}}{\sigma_{LLM\_i}} $$
Kemudian distandarisasi kembali ke distribusi jangkar (contohnya Gemini). Ini mencegah diskontinuitas (*artificial jumps*) pada garis prediksi hari esok semata-mata karena model yang merespons berubah dari Gemini ke Llama.

## 4. Zero-Vector Injection (Offline Quant Failsafe)
Kondisi ekstrem: Tidak ada satupun Node (1-3) yang mengembalikan respons (Total internet blackout atau tidak ada key yang valid).
- **Trigger**: Catcher terakhir pada blok `try/except Exception as e`.
- **Injection Routine**:
  1. Variabel `is_fallback = True` di-flag ke UI.
  2. Fungsi mengembalikan *dictionary* *mock*:
     ```python
     {
         'narrative': 'SYSTEM OFFLINE - RUNNING IN PURE QUANTITATIVE MODE',
         'drift_multiplier': 1.0,  # Skalar identitas, tidak merubah nilai target
         'bias_vector': np.zeros(5), # Vektor Nol, mengeliminasi bias kontekstual
         'confidence': 0.5
     }
     ```
- **Matematika Inferensi**: Output dari *Manager Layer* (Ridge Stacker) yang seharusnya adalah $\hat{Y}_{final} = \hat{Y}_{stacker} \times Drift$, kini menjadi $\hat{Y}_{final} = \hat{Y}_{stacker} \times 1.0$.
- **Dampak**: Sistem tetap men-generate forecast 100% tepat waktu berdasarkan data harga historis murni (LSTM) dan sentimen kuantitatif lokal (FinBERT), tanpa membutuhkan kecerdasan naratif eksternal.
