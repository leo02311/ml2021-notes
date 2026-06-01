---
title: "第27堂課：Unknown Title (Video 27)"
tags:
  - MachineLearning
  - ML2021
---

# 第27堂課：Unknown Title (Video 27)

在本堂課程中，李宏毅教授深入探討了**自監督學習（Self-Supervised Learning, SSL）**的核心技術，並以自然語言處理（NLP）中極具代表性的兩個模型家族——**BERT 系列**與**GPT 系列**為主軸，詳細剖析其運作原理、訓練機制、應用場景，以及為何這些模型能夠展現出驚人的語言理解與生成能力。此外，課程也進一步延伸至多語系 BERT 的神奇對齊特性、Seq2Seq 預訓練模型（如 BART、T5），並簡介了自監督學習在語音（Speech）與電腦視覺（CV）領域的最新進展。

---

## 1. 知識圖譜 (Knowledge Graph)

```mermaid
graph TD
    "Self-Supervised Learning" --> "BERT Series"
    "Self-Supervised Learning" --> "GPT Series"
    "Self-Supervised Learning" --> "Seq2Seq Models"
    "Self-Supervised Learning" --> "Beyond Text"

    "BERT Series" --> "Masked Language Model"
    "BERT Series" --> "Next Sentence Prediction"
    "BERT Series" --> "Downstream Fine-tuning"
    "BERT Series" --> "Multi-lingual BERT"

    "GPT Series" --> "Predict Next Token"
    "GPT Series" --> "In-context Learning"

    "Seq2Seq Models" --> "BART"
    "Seq2Seq Models" --> "T5"

    "Beyond Text" --> "Speech: SUPERB"
    "Beyond Text" --> "CV: SimCLR & BYOL"

    "Downstream Fine-tuning" --> "Case 1: Sequence Classification"
    "Downstream Fine-tuning" --> "Case 2: Sequence Labeling"
    "Downstream Fine-tuning" --> "Case 3: NLI (Natural Language Inference)"
    "Downstream Fine-tuning" --> "Case 4: Extraction-based QA"

    "In-context Learning" --> "Few-shot"
    "In-context Learning" --> "One-shot"
    "In-context Learning" --> "Zero-shot"
```

---

## 2. 什麼是自監督學習 (Self-Supervised Learning)?

機器學習宗師 Yann LeCun 曾指出，「無監督學習（Unsupervised Learning）」是一個容易讓人混淆且定義過於寬泛的詞彙。因此，現在學界更傾向使用**「自監督學習（Self-Supervised Learning）」**。

### 2.1 核心概念與運作機制

在自監督學習中，我們不需要人工標記的標籤（Label）$\hat{y}$。系統會**自動從輸入數據 $x$ 本身創造出監督信號**。其運作方式如下：
1. 將原始輸入數據 $x$ 分割或損毀為兩個部分：$x'$ 與 $x''$。
2. 將其中一部分 $x'$ 輸入模型。
3. 模型輸出預測結果 $y$。
4. 將模型的輸出 $y$ 與原本的另一部分 $x''$（在此作為 Ground Truth）進行比對，計算損失函數並更新參數。

```
       [ 原始數據 x ]
          /       \
     (損毀/分割)  (保留作為 Ground Truth)
        /           \
     [ x' ]         [ x'' ]
       |              |
    [ 模型 ]          |
       |              |
     [ y ] <--誤差最小化--> [ x'' ]
```

---

## 3. BERT 系列：遮罩語言模型 (Masked Language Model)

BERT（Bidirectional Encoder Representations from Transformers）是自監督學習在 NLP 領域的重大突破，其本質上是一個 **Transformer Encoder**。

### 3.1 訓練機制 1：遮罩輸入 (Masking Input)

在訓練 BERT 時，我們會隨機將輸入文本中 $15\%$ 的 Token 進行「遮罩（Masking）」：
* **方法 A**：有 $80\%$ 的機率將該 Token 替換為特殊的 `[MASK]` 標記。
* **方法 B**：有 $10\%$ 的機率將該 Token 隨機替換為任意其他的 Token。
* **方法 C**：有 $10\%$ 的機率保持該 Token 不變。

#### 數學優化目標

BERT 的目標是根據上下文預測被遮罩位置的原始 Token。假設被遮罩的 Token 原始為 $w_t$，BERT 輸出該位置的 Contextualized Embedding，並通過一個線性分類器（Linear Classifier）與 Softmax 函數，輸出在整個字典（Vocabulary）上的概率分佈 $P(\cdot \mid \text{Context})$。我們通過最小化**交叉熵（Cross Entropy）**來訓練模型：

$$\mathcal{L}_{\text{MLM}} = -\sum_{i \in \text{Masked}} \log P(w_i \mid \text{Context})$$

### 3.2 訓練機制 2：下一句預測 (Next Sentence Prediction, NSP)

BERT 的另一個預訓練任務是判斷兩句文本是否為連續的上下句：
* 輸入格式：`[CLS] Sentence 1 [SEP] Sentence 2`
* 模型取出 `[CLS]` 位置的輸出向量，通過一個線性二分類器，預測「Yes（是連續句）」或「No（非連續句）」。

> **課程補充**：後續的研究（例如 RoBERTa）指出，NSP 任務對於模型性能的提升並非必要，甚至可能帶來負面影響。因此後來的模型常將其捨棄，或替換為**句子順序預測（Sentence Order Prediction, SOP）**（如 ALBERT）。

---

## 4. 如何使用 BERT：四大下游任務 (Downstream Tasks)

預訓練完成後，BERT 內部的參數已經具備強大的通用語言表示能力。接著，我們可以使用少量的標記數據，針對特定的**下游任務**進行**微調（Fine-tuning）**。

### Case 1: 序列分類 (Sequence-level Classification)
* **應用範例**：情感分析（Sentiment Analysis）。
* **輸入**：單一句子 `[CLS] this is good`。
* **機制**：取出 `[CLS]` 對應的輸出向量，輸入一個隨機初始化的線性分類器，輸出分類結果（如 positive / negative）。微調時，BERT 的預訓練參數與分類器參數一同更新。
* **效果**：相較於從頭隨機初始化（Train from scratch），使用預訓練的 BERT 初始化能讓模型收斂速度顯著加快，且最終 Loss 更低。

### Case 2: 序列標記 (Token-level Labeling)
* **應用範例**：詞性標記（POS Tagging）。
* **輸入**：單一句子 `[CLS] I saw a saw`。
* **機制**：對句子中的每一個 Token 輸出的特徵向量，都各自連接同一個線性分類器，預測其詞性（如 N, V, DET, N）。

### Case 3: 自然語言推理 (Natural Language Inference, NLI)
* **應用範例**：判斷前提（Premise）與假設（Hypothesis）之間的關係（蘊含 Entailment、矛盾 Contradiction、中立 Neutral）。
* **輸入**：兩句拼接的序列 `[CLS] Premise [SEP] Hypothesis`。
* **機制**：利用 `[CLS]` 的輸出向量進行三分類預測。

### Case 4: 抽取式問答 (Extraction-based Question Answering, QA)
* **問題定義**：輸入為問題 $Q = \{q_1, \dots, q_M\}$ 與段落 $D = \{d_1, \dots, d_N\}$。模型需輸出兩個整數：答案在段落中的起始位置 $s$ 與結束位置 $e$。最終答案為 $A = \{d_s, \dots, d_e\}$。
* **運作機制**：
  1. 將輸入拼接為：`[CLS] Question [SEP] Document`。
  2. 學習兩個全新的關鍵向量（隨機初始化）：**Start Vector** $\mathbf{v}_s$（下圖中橘色）與 **End Vector** $\mathbf{v}_e$（下圖中藍色）。
  3. 將段落中每個 Token $d_i$ 經過 BERT 後產生的輸出特徵向量記為 $\mathbf{h}_i$。
  4. **計算起始位置**：計算 $\mathbf{v}_s$ 與所有 $\mathbf{h}_i$ 的內積（Inner Product），並通過 Softmax 得到概率分佈：
     $$P(s = i) = \frac{\exp(\mathbf{v}_s^T \mathbf{h}_i)}{\sum_{j=1}^N \exp(\mathbf{v}_s^T \mathbf{h}_j)}$$
     選取機率最大的位置作為起始點 $s$。
  5. **計算結束位置**：同理，利用 $\mathbf{v}_e$ 與 $\mathbf{h}_i$ 計算：
     $$P(e = i) = \frac{\exp(\mathbf{v}_e^T \mathbf{h}_i)}{\sum_{j=1}^N \exp(\mathbf{v}_e^T \mathbf{h}_j)}$$
     選取點 $e$。
  6. **限制**：若出現 $e < s$，則視為無解或非法預測。

```
  [Softmax] ---> [ 0.3,  0.5,  0.2 ]  (尋找最大機率，例如 s=2)
     ^
  [內積]
   /  \
[v_s] [h_1, h_2, h_3]  <-- BERT 對 Document 的輸出向量
```

---

## 5. 為什麼 BERT 能工作？語境化特徵與多語對齊

### 5.1 語境化單字嵌入 (Contextualized Word Embedding)

傳統的 Word2Vec 屬於**靜態嵌入**，同一個單字（例如「蘋果」）在不同語境下（「吃蘋果」vs「蘋果手機」）擁有完全相同的向量表示，這顯然不符合語言學規律。

BERT 則實現了**語境化單字嵌入（Contextualized Word Embedding）**。正如語言學家 John Rupert Firth 所言：
> *"You shall know a word by the company it keeps." (欲知一詞，先看其伴。)*

BERT 通過 Self-Attention 機制，能夠根據上下文計算出動態的向量表示。實驗顯示，在「吃蘋果」與「蘋果茶」中，「蘋果」的餘弦相似度（Cosine Similarity）極高；而在「蘋果手機」與「蘋果股價」中，「蘋果」的向量更接近「iPhone」或「科技股」，彼此形成了清晰的語意叢集。

### 5.2 多語對齊的神奇現象 (Multi-lingual BERT)

如果我們在沒有任何跨語言平行對照語料的情況下，僅僅在一大堆不同語言（例如 104 種語言）的獨立文本上訓練一個 Multi-lingual BERT，會發生什麼事？

**答案是：零樣本跨語言遷移（Zero-shot Cross-lingual Transfer）**。

#### 實驗範例：
* **訓練階段**：僅使用**英文**的 QA 問答對數據對 Multi-lingual BERT 進行微調（Fine-tune）。
* **測試階段**：直接輸入**中文**的問答數據進行測試。
* **結果**：模型在完全沒有看過中文 QA 訓練資料的情況下，居然能夠正確進行中文閱讀理解！

#### 科學解釋：語言空間的平移對齊
研究發現，不同語言在 Multi-lingual BERT 的高維 Embedding 空間中，其實共享了**非常相似的幾何結構**，只是整體存在一個**系統性的平移偏差（Translation Offset）**。
1. 如果我們計算所有中文句子的平均向量 $\mathbf{\mu}_{\text{ZH}}$ 以及所有英文句子的平均向量 $\mathbf{\mu}_{\text{EN}}$。
2. 兩者之差 $\mathbf{v}_{\text{shift}} = \mathbf{\mu}_{\text{ZH}} - \mathbf{\mu}_{\text{EN}}$ 即代表了「中文」與「英文」之間的語言特徵轉換向量。
3. 如果我們將英文單字 $x_{\text{EN}}$ 的 Embedding 加上 $\mathbf{v}_{\text{shift}}$，並輸入至重建解碼器（Reconstruction Decoder），就能在**完全無監督（Unsupervised）**的情況下，將英文單字翻譯成對應的中文單字！

```
[ 英文空間 ] ----( 加上語言平移向量 v_shift )----> [ 中文空間 ]
  "rabbit"   ---------------------------------->   "兔"
  "jump"     ---------------------------------->   "跳"
```

---

## 6. Seq2Seq 模型的自監督預訓練

對於 Encoder-Decoder 架構的模型（如機器翻譯、摘要生成等 Seq2Seq 任務），我們需要同時預訓練 Encoder 與 Decoder。

### 6.1 核心想法
輸入一個**受損的序列（Corrupted Sequence）**到 Encoder，並要求 Decoder **重構出原始序列**。

### 6.2 MASS / BART 採用的破壞策略
* **Text Infilling**：隨機將連續的數個 Token 替換為單一 `[MASK]` 標記。
* **Token Deletion**：直接刪除某些 Token。
* **Sentence Permutation**：將句子順序隨機打亂。
* **Document Rotation**：隨機選擇一個 Token 作為起點，將其之前的文本移至段落尾部。

### 6.3 T5 (Text-to-Text Transfer Transformer)
T5 將所有的 NLP 任務（無論是分類、問答還是翻譯）都統一架構為**「Text-to-Text」**的形式（輸入文字，輸出也是文字）。它在海量的 C4（Colossal Clean Crawled Corpus）數據集上進行了詳盡的系統性實驗，為不同預訓練任務的損毀率（如 $15\%$）與損毀長度提供了黃金標準。

---

## 7. GPT 系列：下一個 Token 預測 (Autoregressive)

與 BERT 雙向（Bidirectional）關注上下文不同，GPT（Generative Pre-trained Transformer）是一個 **Transformer Decoder** 模型，採用**自迴歸（Autoregressive）**的方式進行單向預測。

### 7.1 預測機制
GPT 的任務非常簡單：**預測下一個 Token**。
給定前文 $w_1, w_2, \dots, w_t$，預測 $w_{t+1}$ 的機率：

$$P(w_{t+1} \mid w_1, w_2, \dots, w_t)$$

因為只能看見左側的歷史資訊，GPT 在 Self-Attention 中使用了 **Masked Self-Attention**，強行阻斷向右側未來資訊的注意力傳遞。

### 7.2 驚人的「上下文學習」 (In-context Learning)

隨著模型規模從 GPT-1、GPT-2 暴增到 GPT-3（1750 億參數），研究人員發現了一個顛覆傳統的現象：**In-context Learning**。

對於 GPT-3，我們**不需要調整模型參數（不需要進行梯度下降 Gradient Descent）**，只需在輸入（Prompt）中提供任務說明與幾個範例，模型就能自動學會如何執行全新任務：

* **Few-shot Learning 範例輸入**：
  ```text
  Translate English to French:       <-- 任務描述
  sea otter => loutre de mer         <-- 範例 1
  peppermint => menthe poivrée       <-- 範例 2
  plush giraffe => girafe peluche     <-- 範例 3
  cheese =>                          <-- Prompt (引導模型輸出目標回答)
  ```
* **運作機制**：模型在 inference 時，僅憑強大的預訓練模式辨識能力，就能在不更新任何權重的情況下，輸出正確的法文 `fromage`。

---

## 8. 隨堂測驗

### 測驗 1：BERT 與 GPT 的本質差異
請問 BERT 與 GPT 在網路架構（Network Architecture）與注意力機制（Attention Mechanism）上有何最本質的差異？這如何影響兩者的應用擅長點？

<details>
<summary><b>點擊展開解答</b></summary>

* **架構差異**：
  * **BERT** 基於 **Transformer Encoder**。它使用的是**雙向注意力機制（Bidirectional Attention）**，每個 Token 都可以同時關注其左側和右側的所有資訊。
  * **GPT** 基於 **Transformer Decoder**。它使用的是**遮罩注意力機制（Masked Attention）**，每個 Token 只能關注其左側（歷史）的資訊，右側（未來）的資訊被 Mask 阻斷。
* **影響與應用**：
  * **BERT** 擅長**自然語言理解（NLU）**任務（如分類、標記、問答），因為雙向特徵能提供更完整、更深刻的語意表徵。
  * **GPT** 擅長**自然語言生成（NLG）**任務，因為其自迴歸（Autoregressive）的預測機制與文本生成的逐字輸出流程完全吻合。
</details>

---

### 測驗 2：BERT 抽取式問答（QA）的向量計算
在將 BERT 應用於抽取式問答（Case 4）時，我們需要隨機初始化哪兩個向量？若輸入的段落長度為 $N$，我們該如何利用這兩個向量與 BERT 的輸出向量來確定答案的起點與終點？

<details>
<summary><b>點擊展開解答</b></summary>

1. **初始化的向量**：需要額外學習兩個隨機初始化的向量，分別是 **Start Vector** $\mathbf{v}_s$ 與 **End Vector** $\mathbf{v}_e$。
2. **起點計算**：將 $\mathbf{v}_s$ 與段落中各個 Token 經由 BERT 輸出產生的特徵向量 $\mathbf{h}_i$（$i \in \{1, \dots, N\}$）進行**內積（Inner Product）**計算。隨後將這 $N$ 個內積值通過 **Softmax** 函數轉化為機率分佈，機率最大（或值最大）的位置即為答案的起始位置 $s$。
3. **終點計算**：同理，使用 $\mathbf{v}_e$ 與 $\mathbf{h}_i$ 進行內積並通過 Softmax 運算，挑選出機率最高的位置作為答案的結束位置 $e$。
4. **答案擷取**：最終預測的答案區間即為 $d_s$ 到 $d_e$。
</details>

---

### 測驗 3：多語系 BERT 的跨語言對齊原理
根據課程中所提的研究，Multi-lingual BERT 在沒有任何平行翻譯語料的情況下，為什麼能夠實現「英文訓練、中文測試」的零樣本遷移？我們如何利用其 Embedding 進行無監督的單字翻譯？

<details>
<summary><b>點擊展開解答</b></summary>

1. **對齊原理**：研究發現，不同語言在 Multi-lingual BERT 的高維向量空間中，共享了**極為相似的幾何流形結構（Geometric Structure）**，只是在空間中存在整體的系統性偏移。
2. **無監督翻譯方法**：
   * 第一步：計算大量中文句子向量的平均值，得到中文的中心向量 $\mathbf{\mu}_{\text{ZH}}$。
   * 第二步：計算大量英文句子向量的平均值，得到英文的中心向量 $\mathbf{\mu}_{\text{EN}}$。
   * 第三步：計算兩者的差值，作為語言轉換的偏移向量：$\mathbf{v}_{\text{shift}} = \mathbf{\mu}_{\text{ZH}} - \mathbf{\mu}_{\text{EN}}$。
   * 第四步：若要翻譯一個英文單字，只需將該單字的 Embedding 加上 $\mathbf{v}_{\text{shift}}$，並將此新向量送入重建解碼器（Reconstruction Decoder），即可在無監督的情況下重構出對應的中文單字。
</details>