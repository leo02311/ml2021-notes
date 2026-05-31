---
title: "Auto-encoder (Part 2 of 2)"
tags:
  - MachineLearning
  - ML2021
  - SelfSupervisedLearning
  - BERT
  - GPT
---

# 第30堂課：Auto-encoder (Part 2 of 2)

本堂課由李宏毅教授介紹「自我監督學習 (Self-supervised Learning)」。這是一種機器學習方法，系統會學習根據輸入的一部分來預測輸入的另一部分，從而無需人工標註即可獲取監督信號。

## 1. 什麼是自我監督學習 (Self-supervised Learning)

傳統的監督學習需要大量的標註數據 $y$。在自我監督學習中，我們利用輸入 $x$ 本身的一部分作為監督信號，從另一部分預測它。

- **核心概念**：
  - 輸入：$x$
  - 任務：利用 $x'$ 預測 $x''$，其中 $x = \{x', x''\}$。
- **Yann LeCun 的定義**：將「無監督學習 (Unsupervised)」改稱為「自我監督學習」，因為原先的術語容易引起混淆。系統透過預測輸入的某一部分來學習數據的表示。

## 2. BERT 系列模型

BERT (Bidirectional Encoder Representations from Transformers) 是自我監督學習的經典代表，其訓練過程包含以下關鍵任務：

### 核心任務
1. **Masked Token Prediction (掩碼語言模型)**：
   - 隨機將輸入序列中的部分 tokens 替換為 `[MASK]` 或隨機 token。
   - 模型透過 Transformer Encoder 輸出並接一個 Linear Layer，透過最小化 Cross Entropy 來預測被遮蔽的原始 token。
2. **Next Sentence Prediction (NSP)**：
   - 判斷兩句話之間的順序關係（已證明在某些情況下對下游任務幫助有限）。
   - **SOP (Sentence Order Prediction)**：在 ALBERT 等模型中作為改進的訓練策略。

### 下游任務 (Downstream Tasks)
預訓練 (Pre-train) 階段後，模型在具備少量標註數據的下游任務上進行微調 (Fine-tune)：
- **Case 1 (Classification)**：例如情感分析，輸入一個序列，輸出一個類別。
- **Case 2 (POS Tagging)**：輸入一個序列，每個位置輸出一個對應的標籤。
- **Case 3 (NLI)**：輸入兩個序列，判斷它們之間的邏輯關係（Entailment, Contradiction, Neutral）。
- **Case 4 (Extraction-based QA)**：給定文件 $D$ 與問題 $Q$，輸出答案的起始位置 $s$ 與結束位置 $e$。

## 3. GPT 系列模型

GPT 系列模型主要基於 Transformer Decoder，專注於生成任務。

- **核心任務**：**Predict Next Token** (預測下一個詞)。
- **In-context Learning (Few-shot)**：GPT 透過給予少量的範例 (examples) 和任務描述，即可在無需 Gradient Descent 的情況下執行特定任務。

## 4. 跨領域與多語言應用

- **Multi-lingual BERT**：在多語言語料庫上訓練，實現了「零樣本閱讀理解 (Zero-shot Reading Comprehension)」。
- **跨語言對齊 (Cross-lingual Alignment)**：即便不強制對齊，模型在大量資料訓練後，不同語言的相同語義詞彙在向量空間中也會趨近。
- **Beyond Text**：自我監督學習已擴展至語音處理 (Speech) 與影像處理 (CV)。
  - **SUPERB**：語音處理領域的通用性能基準測試。
  - **SimCLR / BYOL**：影像自我監督學習框架。

## 5. 知識圖譜

```mermaid
graph TD
    "Self-Supervised Learning" --> "Language Modeling"
    "Self-Supervised Learning" --> "Downstream Tasks"
    "Language Modeling" --> "BERT"
    "Language Modeling" --> "GPT"
    "BERT" --> "Masked Token Prediction"
    "BERT" --> "NSP/SOP"
    "BERT" --> "Fine-tuning"
    "GPT" --> "Predict Next Token"
    "GPT" --> "In-context Learning"
    "Beyond Text" --> "Speech"
    "Beyond Text" --> "CV"
    "Speech" --> "SUPERB"
    "CV" --> "SimCLR"
    "CV" --> "BYOL"
```

---

## 隨堂測驗

### Q1：什麼是自我監督學習 (Self-supervised Learning) 的主要目的？
<details>
<summary>點擊展開解答</summary>
自我監督學習旨在利用輸入數據本身的一部分作為監督信號（Label），來預測數據的另一部分，從而減少對大量人工標註數據的依賴。
</details>

### Q2：在 BERT 的預訓練中，Masked Token Prediction 的目標是什麼？
<details>
<summary>點擊展開解答</summary>
隨機將序列中的 tokens 遮蔽或替換，模型透過 Transformer Encoder 學習上下文表示，並嘗試透過 Linear Layer 輸出機率分布，藉由最小化 Cross Entropy 來正確預測原始被遮蔽的 token。
</details>

### Q3：GPT 模型提到的「In-context Learning」具有什麼特性？
<details>
<summary>點擊展開解答</summary>
該學習方法無需更新模型的參數（即 no gradient descent），只需透過在輸入的 Prompt 中提供任務描述與少量範例，模型即可根據上下文進行推論。
</details>

## 來自課程原聲的重點摘要

## 來自課程原聲的重點摘要

* **特徵解纏繞（Feature Disentanglement）的概念與意義：**
    * 教授將「解纏繞」形象地比喻為將混在一起、糾纏不清的事物分開的過程。
    * Autoencoder 可以將影像轉為編碼（Code），並從編碼還原影像。這意味著編碼包含了該影像的所有資訊（如色彩、質感等）。
    * **應用場景：** 在語音處理中，若將一段語音輸入 Autoencoder，解碼器能協助我們從編碼中「提取」出重要的資訊——包含語音的**內容（Content）**與**說話者（Speaker）資訊**。
    * **難點：** 在將編碼壓縮到一個向量（Vector）時，我們無法得知該向量中的哪個維度對應的是內容資訊，哪個對應的是說話者資訊。
    * **解決方案：** 特徵解纏繞（Feature Disentanglement）的核心在於訓練 Autoencoder 時，強制要求編碼器產出特定格式的向量，例如：前 50 個維度專門負責「內容」，後 50 個維度專門負責「說話者」。

* **語音轉換（Voice Conversion）的實現邏輯：**
    * 教授以過去的「柯南變聲領結」為例，說明語音轉換的技術演變。
    * 過去需要兩人的語音數據（A 說「你好」，B 也說「你好」）進行成對訓練。現在，利用特徵解纏繞技術，無需兩人說出完全相同的語句，也能實現語音轉換，讓機器學習將說話者的特徵與語音內容分離。

* **離散潛在表示（Discrete Latent Representation）與 VQVAE：**
    * 透過二值化（Binary）或 One-hot 編碼，讓編碼的維度具有「存在與否」的特徵，有利於模型的分類任務（如辨識手寫數字）。
    * **VQVAE (Vector Quantized Variational Autoencoder)：** 這是離散表示的一個重要應用，類似於 Self-attention 的機制，通過計算 Query 與 Key 的相似度來選擇最佳的 Codebook，能處理更複雜的生成任務。

* **異常檢測（Anomaly Detection）：**
    * Autoencoder 可作為異常檢測工具。當輸入異常數據時，因模型無法有效重構該數據，重構後的影像與原始影像差異會很大。
    * 教授提到，這在偵測信用卡詐騙、網路入侵與醫學影像檢測中有廣泛應用。
    * **注意事項：** 訓練這類模型時，數據的標記至關重要。通常我們會訓練模型學習「正常」的樣貌，若模型無法正確還原輸入資料，則該資料即被歸類為異常。
