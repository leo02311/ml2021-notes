---
title: "Self-supervised Learning (aka Foundation Model) (Part 2 of 3)"
tags:
  - MachineLearning
  - ML2021
  - NLP
  - Self-SupervisedLearning
  - BERT
  - GPT
---

# 第27堂課：Self-supervised Learning (aka Foundation Model) (Part 2 of 3)

本堂課由李宏毅教授介紹自監督學習（Self-Supervised Learning, SSL）。SSL 的核心概念是讓系統學習從輸入的一部份去預測另一部份，利用輸入資料本身的資訊作為監督訊號，而非依賴人工標註的資料。

## 1. 自監督學習的核心概念

*   **定義**：Yann LeCun 指出，「無監督學習（Unsupervised）」是一個容易混淆且帶有偏見的術語。在「自監督學習」中，系統會將輸入 $x$ 切分成 $x'$ 與 $x''$，並利用 $x'$ 作為輸入，試圖預測 $x''$。
*   **與監督學習的對比**：
    *   **監督學習**：輸入 $x \rightarrow$ 輸出 $y$（人工標註）。
    *   **自監督學習**：輸入 $x' \rightarrow$ 輸出 $x''$（從資料本身取得）。

## 2. BERT 系列模型

BERT (Bidirectional Encoder Representations from Transformers) 是基於 Transformer Encoder 的預訓練模型，常用於理解語言。

### 核心機制
*   **遮罩輸入 (Masking Input)**：隨機遮蔽輸入序列中的部分 token（替換為 `[MASK]` 或隨機文字），讓模型預測這些被遮住的內容。
*   **訓練目標**：最小化預測與真實文字之間的交叉熵（Cross Entropy）。
*   **任務類型**：
    *   **Masked Token Prediction**：填空任務。
    *   **Next Sentence Prediction (NSP)**：判斷兩個句子是否銜接，但在後來如 RoBERTa 與 ALBERT 的研究中，SOP (Sentence Order Prediction) 被認為更有效。

### BERT 的應用方式
1.  **分類問題**（如情感分析）：將 `[CLS]` 的輸出經過一層 Linear 層進行分類。
2.  **序列標註**（如詞性標註 POS Tagging）：針對每個 token 的輸出分別通過 Linear 層。
3.  **自然語言推理 (NLI)**：輸入兩個序列，判斷其關係（蘊含、矛盾、中立）。
4.  **抽取式問答 (Extraction-based QA)**：給定文件與問題，模型輸出兩個整數 $(s, e)$，代表答案在文件中的起始與結束位置。

## 3. GPT 系列模型

GPT (Generative Pre-trained Transformer) 基於 Transformer Decoder，主要擅長「生成 (Generation)」。

*   **預測下一個 Token**：訓練目標是根據當前序列預測下一個 token。
*   **In-context Learning**：GPT 具備強大的 Few-shot 能力，透過在提示詞（Prompt）中放入少量範例，模型即可在不需要梯度下降（Gradient Descent）的情況下執行新任務。

## 4. 自監督學習的廣泛應用

自監督學習不僅限於文字，還擴展至語音與電腦視覺（CV）。

*   **跨領域應用**：將 BERT 應用於蛋白質序列、DNA 分類與音樂生成。
*   **語音處理 (SUPERB)**：Speech processing Universal PERformance Benchmark 是一個統一的框架，用來測試各種自監督語音模型在 10 多種下游任務（如 ASR、語者識別、情感分析等）的表現。
*   **CV 領域**：如 SimCLR（最大化增強影像之間的一致性）與 BYOL（Bootstrap Your Own Latent）。

---

## 知識圖譜 (Knowledge Graph)

```mermaid
graph TD
    "Self-Supervised Learning" --> "BERT Series"
    "Self-Supervised Learning" --> "GPT Series"
    "Self-Supervised Learning" --> "Speech Processing"
    "Self-Supervised Learning" --> "Computer Vision"
    "BERT Series" --> "Masked Token Prediction"
    "BERT Series" --> "Downstream Tasks"
    "GPT Series" --> "Predict Next Token"
    "GPT Series" --> "Few-shot Learning"
    "Speech Processing" --> "SUPERB Benchmark"
    "Computer Vision" --> "SimCLR"
    "Computer Vision" --> "BYOL"
```

---

## 隨堂測驗

1.  **問題：為什麼 Yann LeCun 建議改用「自監督學習」而非「無監督學習」這個術語？**
    <details>
    <summary>點擊展開解答</summary>
    因為「無監督學習」是一個容易混淆且帶有負面意涵的術語。在自監督學習中，系統實際上是利用輸入資料的一部份來預測另一部份，這部份資料起到了「監督訊號」的作用。
    </details>

2.  **問題：在 BERT 的抽取式問答 (Extraction-based QA) 中，模型的輸出是什麼？**
    <details>
    <summary>點擊展開解答</summary>
    模型輸出兩個整數 $(s, e)$，分別代表答案在文件中出現的「起始位置 (Start)」與「結束位置 (End)」。
    </details>

3.  **問題：什麼是 GPT 的「In-context Learning」？**
    <details>
    <summary>點擊展開解答</summary>
    指模型在執行新任務時，僅需透過在輸入提示詞 (Prompt) 中提供少量範例（Few-shot），即可進行預測或生成，且此過程不需要進行傳統的梯度下降參數更新。
    </details>

## 來自課程原聲的重點摘要

## 來自課程原聲的重點摘要

*   **自我監督學習 (Self-supervised Learning) 的核心概念**
    *   **比喻**：自我監督學習就像是「沒有標準答案」的監督學習。我們沒有外部提供的標記資料，而是透過將原始資料切割，讓模型將其中一部分當作輸入，另一部分當作目標（Label），試圖學習資料內部的關聯性。
    *   **以 BERT 為例**：教授說明 BERT 是典型的自我監督學習架構。其運作邏輯為：輸入一個句子並隨機「遮蔽 (Mask)」部分 token，模型目標是根據前後文正確預測出被遮蔽的詞彙。這裡，被遮蔽的詞彙在訓練當下就扮演了目標（Label）的角色，因此稱為「自我監督」。

*   **訓練與難點邏輯**
    *   **遮蔽策略 (Masking)**：教授提到有兩種主要的遮蔽方式。一是將 token 換成一個特殊的 `[MASK]` 符號；二是直接將 token 替換成其他隨機詞彙。模型需要在這種「缺漏」的情況下，依據前後文上下文的表徵（Contextual Representation）來還原正確詞彙。
    *   **推導邏輯**：BERT 的訓練不僅僅是簡單的語言填充，其背後是透過強大的 Transformer 架構，提取序列中詞與詞之間的注意力權重（Self-attention），讓模型理解語意結構。

*   **學生常犯錯誤與重點觀念**
    *   **誤以為 BERT 僅限於 NLP**：教授強調，雖然 BERT 最早應用於文字處理，但其機制同樣適用於語音或影像，只要資料能被序列化（變成一連串向量），BERT 就能進行自我監督學習。
    *   **下游任務 (Downstream Tasks) 是關鍵**：教授特別提醒，自我監督訓練出來的 BERT 其實只是「預訓練 (Pre-training)」的基底。要應用於特定任務（如情感分析、分類），必須進行「微調 (Fine-tuning)」，也就是在模型後面接上適當的線性層（Linear Layer）和輸出層，共同優化以符合下游任務的需求。
    *   **任務選擇的重要性**：教授提到，儘管 BERT 很強，但並非所有下游任務的訓練策略都有效。例如他提到的「Next Sentence Prediction (NSP)」在後續的研究中被指出對某些任務的幫助有限，這提醒了我們在模型設計時需保持實驗性的思維。
