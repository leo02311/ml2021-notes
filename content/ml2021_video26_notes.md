---
title: "Self-supervised Learning (aka Foundation Model) (Part 1 of 3)"
tags:
  - MachineLearning
  - ML2021
  - SelfSupervisedLearning
  - BERT
  - GPT
---

# 第26堂課：Self-supervised Learning (aka Foundation Model) (Part 1 of 3)

本堂課由李宏毅教授介紹**自監督學習 (Self-Supervised Learning)** 的概念，探討如何透過模型自行從數據中產生監督訊號，並詳細分析了 BERT 與 GPT 系列模型的核心技術與應用。

---

## 一、 自監督學習的核心概念

傳統監督學習 (Supervised Learning) 依賴人工標記的標籤 $y$，而自監督學習則是讓系統學習從輸入 $x$ 的一部分預測另一部分。Yann LeCun 將其定義為「使用部分輸入作為剩餘部分的監督訊號」。

```mermaid
graph TD
    "自監督學習" --> "監督訊號來源"
    "監督訊號來源" --> "利用部分輸入預測其餘部分"
    "應用領域" --> "自然語言處理 (NLP)"
    "應用領域" --> "語音處理 (Speech)"
    "應用領域" --> "電腦視覺 (CV)"
    "核心模型系列" --> "BERT系列"
    "核心模型系列" --> "GPT系列"
```

---

## 二、 BERT 系列：從編碼器 (Encoder) 出發

BERT (Bidirectional Encoder Representations from Transformers) 是基於 Transformer Encoder 的預訓練模型，擁有 340M 參數。

### 1. 預訓練方法 (Pre-train)
*   **Masked Token Prediction**: 隨機遮蔽部分 tokens (使用 `[MASK]` 或隨機替換)，透過 Transformer Encoder 輸出並接上線性層 (Linear)，最小化與真實詞彙的交叉熵 (Cross Entropy)。
*   **Next Sentence Prediction (NSP)**: 判斷兩句句子是否連續（在 RoBERTa 等後續改進中，該方法被認為效果不佳；ALBERT 則改用 SOP: Sentence Order Prediction）。

### 2. 下游任務 (Downstream Tasks) 的微調 (Fine-tuning)
透過大量的無標記資料進行預訓練，再針對少量標記資料的任務進行微調：
*   **Case 1 (序列分類)**: 如情緒分析，將 `[CLS]` 輸出的 embedding 接上 Linear 層。
*   **Case 2 (序列標記)**: 如詞性標注 (POS tagging)，對每個 token 的輸出分別進行分類。
*   **Case 3 (自然語言推論 NLI)**: 判斷前提 (Premise) 與假設 (Hypothesis) 的關係 (entailment, contradiction, neutral)。
*   **Case 4 (問答系統 QA)**: 提取式問答，預測答案在文件中的開始位置 $s$ 與結束位置 $e$。

---

## 三、 GPT 系列：從解碼器 (Decoder) 出發

GPT (Generative Pre-trained Transformer) 系列專注於生成任務，透過「預測下一個 Token (Predict Next Token)」來進行預訓練。

*   **生成能力**: GPT 系列如 GPT-2, GPT-3 具有極強的文字生成能力，能根據提示 (Prompt) 進行創意寫作。
*   **In-context Learning**: 無需梯度下降，透過給予任務描述與少許範例（Few-shot learning），模型即可在推論時執行指定任務。

---

## 四、 多語言與跨領域應用

*   **多語言 BERT (Multi-lingual BERT)**: 在多種語言資料上進行訓練，使模型具備跨語言能力。
    *   **零樣本閱讀理解 (Zero-shot Reading Comprehension)**: 即使僅使用英文資料進行訓練，模型在中文問答測試上也能有不錯表現，顯示模型學到了跨語言的對齊 (Alignment)。
*   **跨領域應用**: 
    *   將 BERT 應用於生物資訊（蛋白質、DNA 序列）與音樂分類，效果往往優於隨機初始化的模型。

---

## 五、 自監督學習的發展趨勢

除了文字領域，自監督學習也廣泛應用於語音與影像：
*   **語音處理**: 透過 SUPERB (Speech processing Universal PERformance Benchmark) 框架，比較不同自監督預訓練模型在語音辨識、說話者識別、情感分類等任務的表現。
*   **電腦視覺**: 如 SimCLR (Maximize agreement between two views) 與 BYOL (Bootstrap your own latent)，透過對比學習 (Contrastive learning) 等方式提取影像表徵。

---

## 隨堂測驗

1. **問：根據李宏毅教授的觀點，為什麼 BERT 應用在 DNA 序列分類時，表現通常優於隨機初始化的模型？**
   <details>
   <summary>點擊展開解答</summary>
   因為 BERT 在預訓練階段已經學會了處理序列資訊（Contextualized embedding），即便資料類型從自然語言換成 DNA 序列，模型捕捉序列特徵的能力仍然能幫助下游分類任務。
   </details>

2. **問：在 GPT 的 "Few-shot" Learning 中，訓練過程中是否需要進行梯度下降 (Gradient Descent)？**
   <details>
   <summary>點擊展開解答</summary>
   不需要。GPT 的 Few-shot Learning 屬於 "In-context" Learning，模型是透過提示 (Prompt) 中的範例直接進行推論，不需更新模型權重。
   </details>

3. **問：BERT 進行 Masked Token Prediction 時，輸入端除了 `[MASK]` token 之外，還有什麼方式來破壞輸入資料以供模型預測？**
   <details>
   <summary>點擊展開解答</summary>
   使用隨機的其他 token (Random tokens) 進行替換。
   </details>

## 來自課程原聲的重點摘要

## 來自課程原聲的重點摘要

* **關於自監督學習模型（Self-supervised Learning Models）的命名規則：**
  * 教授以芝麻街（Sesame Street）的角色來為這些模型命名。例如：ELMo 是取自芝麻街中的紅色怪獸角色；BERT 是為了要跟 ELMo 湊在一起，取自芝麻街中與 ELMo 是好朋友的角色。
  * 這種取名方式通常帶有一點隨性或惡趣味，並沒有嚴謹的命名邏輯。

* **模型規模的直觀比喻：**
  * **GPT-3 的體積：** 教授將 BERT 比作一個「超級巨大的巨人」，而 GPT-3 則是比 BERT 大上 10 倍的巨型模型。
  * **如何感受參數的大小：** 為了讓學生對參數數量級有感，教授將模型的大小視覺化。他將 BERT 的參數數量（3.4 億）比喻為一個「1 米高」的身高，而 GPT-3 的體積則大約是台北 101 大樓的高度，藉此讓學生理解這些模型在硬體算力上的驚人差異。
  * **更驚人的 Switch Transformer：** 當模型大到 1.6 萬億（1.6 Trillion）參數時，其複雜度已經超越了人類大腦的預估神經元數量（1,000 億）。

* **教授特別強調的觀點：**
  * **這門課並非在討論卡通：** 儘管教授大量運用芝麻街的角色來舉例，但他強調這純粹是為了方便記憶與趣味性。這些模型的命名規則反映了深度學習領域中，研究者們喜歡透過特定創意來為研究成果命名的文化。
  * **「不看」的隱喻：** 在提到模型原理或潛在風險時，教授會幽默地請同學「閉上眼睛、摀住耳朵」，這種互動方式是用來緩解課程中較為枯燥或艱深部分的學習壓力，同時也提醒同學即將進入較為艱深的技術細節。

* **學習上的注意事項：**
  * 同學們不需要過度鑽研這些怪異的命名，重點在於理解這些模型（如 BERT, GPT）背後的「自監督學習」架構及其在不同任務中的應用價值。
