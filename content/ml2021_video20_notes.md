好的，我將扮演專業技術筆記整理專家，為您整理李宏毅教授機器學習課程第20堂的逐字稿，輸出為精美的 Markdown 筆記。

---
title: 第20堂課：Unknown Title (Video 20)
tags:
  - MachineLearning
  - ML2021
  - Decoder
  - AutoregressiveModel
  - NonAutoregressiveModel
  - Transformer
  - AttentionMechanism
  - CrossAttention
  - TeacherForcing
  - ScheduledSampling
  - BeamSearch
  - CopyMechanism
  - GuidedAttention
---

# 【機器學習 2021】第20堂課：Unknown Title (Video 20)

本堂課將深入探討機器學習模型中的 **Decoder (解碼器)**，這是繼兩週前討論 **Encoder (編碼器)** 之後的重要內容。我們將聚焦於其運作原理、內部結構、訓練方法，以及多種優化技巧。

## 課程概覽：Encoder與Decoder

回顧兩週前，我們討論了 **Encoder** 的概念。Encoder 的主要功能是將輸入序列（例如語音訊號或原始文本）轉換為一串高維度的向量序列。這些向量包含了輸入的語義資訊。

接下來，**Decoder** 的任務便是接收這些由 Encoder 產生的向量，並將其轉換成我們期望的輸出序列（例如文字文本、翻譯結果或合成語音）。

## Autoregressive (AR) Decoder 詳解

AR Decoder 是最常見的一種解碼器，其核心思想是「**逐步生成**」輸出序列。每個時間步的輸出都依賴於前一個時間步的輸出。

### 運作流程 (以語音辨識為例)

我們以語音辨識或機器翻譯任務為例，解釋 AR Decoder 的工作方式。

1.  **輸入與輸出單位**
    *   **輸入 (Encoder):** 語音訊號 (e.g., "machine learning")
    *   **Encoder 輸出:** 一系列向量 (代表語音的隱藏資訊)
    *   **Decoder 任務:** 生成文字序列 (e.g., "機器學習")

2.  **特殊符號 (BOS)**
    *   Decoder 啟動時，會先接收一個特殊的 **`begin of sentence` (BOS)** 符號作為初始輸入。這個符號代表了句子開始的信號。
    *   **表示方式:** BOS 通常被表示為一個 **One-hot Vector**，其長度與詞彙表大小相同，BOS 對應的維度為 1，其餘為 0。

3.  **詞彙表 (Vocabulary) 與輸出單位**
    *   **詞彙表:** 定義了 Decoder 可能生成的所有單詞或字元。
    *   **輸出單位選擇:**
        *   **中文語音辨識:** 通常以中文字符為單位 (例如，常見的 2000-5000 個字)。
        *   **英文:** 可選擇字母 (A-Z)、單詞，或更常見的 **Subword (子詞)** 單位，將單詞拆分為詞根和詞綴，減少詞彙表大小。
    *   **輸出向量長度:** 與詞彙表大小相同，每個維度對應一個詞彙。

4.  **Softmax與單詞生成**
    *   Decoder 的輸出是一個向量，其長度與詞彙表大小相同。
    *   這個向量會通過 **Softmax 函數**，將各個詞彙的「分數」轉換為「機率分佈」，所有機率值加總為 1。
    *   **首個輸出:** 選擇機率最高的詞彙作為 Decoder 的第一個輸出。例如，如果 `機` 字的機率最高，則輸出 `機`。

5.  **序列生成機制**
    *   **遞迴輸入:** Decoder 會將自己前一個時間步的輸出 (例如 `機`) 作為下一個時間步的新輸入，與原有的 `BOS` 符號一起送入。
    *   **依序生成:** 根據 `BOS` 和 `機`，Decoder 再生成第二個詞彙 (例如 `器`)。此過程持續重複，將 `BOS`, `機`, `器` 作為輸入，再生成 `學`，以此類推。
    *   **核心特點:** **「Decoder 看見的是自己前一時間步的輸出」**。

6.  **錯誤傳播問題 (Error Propagation)**
    *   由於 Decoder 是依賴自身輸出來進行下一個生成，如果某一步生成了錯誤的詞彙，這個錯誤可能會被傳播到後續的生成步驟中，導致整個輸出序列都出錯。
    *   本課程後續會討論解決方案。

### 內部結構 (Transformer Decoder)

在 Transformer 架構中，Decoder 的內部結構比 Encoder 稍複雜。

1.  **與Encoder的比較**
    *   如果隱藏 Decoder 的中間部分，Encoder 和 Decoder 的基本結構是相似的：都包含 **Multi-head Attention**、**Add & Norm**、**Feed Forward**、**Add & Norm**，並重複 N 次。
    *   **主要差異:** Decoder 在其 Multi-head Attention (Self-Attention) 層中加入了一個 **Mask**，並且多了一個 **Cross-Attention** 層。

2.  **Masked Self-Attention (遮罩式自我注意力)**
    *   **傳統 Self-Attention (Encoder):** 每個輸出向量 `bi` 在生成時，可以考慮完整的輸入序列 `a1` 到 `aN` 的所有資訊。
    *   **Masked Self-Attention (Decoder):** 為了模擬逐步生成的過程，當 Decoder 生成第 `i` 個輸出 `bi` 時，它 **只能考慮 `a1` 到 `ai` 的資訊**，而不能「偷看」`ai+1` 到 `aN` 的未來資訊。
    *   **實現方式:** 在計算注意力分數時，將未來位置的注意力權重設為負無窮大 (或直接設為零)，使其在 Softmax 後機率為零。
    *   **原因:** Decoder 的輸出是「一個接一個」產生的，在生成 `ai` 時，`ai+1`、`ai+2` 等還並不存在，因此不能使用這些資訊。

### 停止機制 (EOS)

既然 Decoder 逐步生成，如何知道何時停止呢？

1.  **特殊符號 `End` (EOS)**
    *   需要在詞彙表中加入另一個特殊符號 **`end of sentence` (EOS)**。
    *   當 Decoder 判斷當前序列已經結束時，它會輸出 `end` 符號。
    *   一旦 `end` 符號被輸出，整個生成過程就宣告結束。
    *   **備註:** `begin` 和 `end` 符號在某些實作中可以是相同的，因為它們分別只出現在輸入和輸出中，不會混淆。

## Non-Autoregressive (NAR) Decoder 簡介 (NAT)

與 AR Decoder 逐步生成不同，NAR Decoder (通常簡稱為 NAT) 的目標是「**一次性生成整個序列**」。

### 運作方式

1.  **一次性生成:** NAT Decoder 不會像 AR 模型那樣將前一個輸出作為下一個輸入，而是嘗試在一個步驟內生成整個輸出序列。
2.  **輸入:** NAT Decoder 的輸入可能是一整排的 `begin` 符號 (例如，四個 `begin` 生成四個字)。

### 長度預測問題與解決方案

由於 NAT 是一次性生成，它需要知道要生成多長的序列。

1.  **額外長度分類器:**
    *   訓練另一個分類器，其輸入與 Encoder 相同，輸出是一個數字，代表預期的輸出長度。
    *   NAT Decoder 收到這個長度後，便生成相應數量的 `begin` 符號作為輸入。
2.  **固定長度與 EOS 截斷:**
    *   設定一個最大輸出長度上限 (例如 300)。
    *   NAT Decoder 始終輸入 300 個 `begin` 符號，並輸出 300 個詞彙。
    *   在輸出序列中找到 `end` 符號的位置，截斷其後的詞彙。

### 優點 (平行化、速度、長度控制)

1.  **平行化 (Parallelization):** 由於所有詞彙是同時生成的，NAT Decoder 可以更好地利用平行計算資源，顯著加速推論速度。
    *   AR 需要 100 步生成 100 個詞彙，NAT 只需 1 步。
2.  **速度快:** 相比 AR Decoder，NAT 在生成長序列時速度更快。
3.  **長度控制:**
    *   在語音合成 (Speech Synthesis) 等任務中，可以透過調整長度分類器的輸出，直接控制生成語音的速度 (例如，將長度除以 2 語音加快一倍，乘以 2 語音減慢一倍)。
    *   例如，在 FastSpeech 模型中應用。

### 缺點 (性能問題、Multi-modality)

1.  **性能較差:** 雖然看起來很強大，但 NAT Decoder 的性能通常不如 AR Decoder。
    *   研究者們正在努力提升 NAT 的性能，使其能與 AR 媲美，但通常需要更多技巧。
2.  **Multi-modality (多模態問題):** 這是 NAT 性能不佳的其中一個主要原因，但本次課程不會深入討論。

## Encoder與Decoder的互動：Cross-Attention

Decoder 中間的特殊部分，連接 Encoder 和 Decoder 的橋樑，就是 **Cross-Attention (交叉注意力)**。

### Cross-Attention機制

1.  **角色:** Cross-Attention 讓 Decoder 能夠「讀取」Encoder 輸出的資訊。
2.  **運作原理:**
    *   Decoder 的中間層會產生 **Query (Q)**。
    *   Encoder 的輸出向量序列 (A1, A2, A3...) 會用來生成 **Key (K)** 和 **Value (V)**。
    *   Decoder 的 Q 會與 Encoder 的 K 進行點積運算，計算出注意力分數 (Alpha)。
    *   這些注意力分數會與 Encoder 的 V 進行加權求和，得到一個上下文向量 (Context Vector)。
    *   這個上下文向量包含了 Decoder 在當前生成步驟所需，從 Encoder 輸出中提取的相關資訊。
    *   **關鍵點:** Q 來自 Decoder，K 和 V 來自 Encoder。

### 歷史發展與應用 (Listen, Attend and Spell)

Cross-Attention 的概念早在 Transformer 之前就已存在。

1.  **`Listen, Attend and Spell` (LAS) 論文 (2016):**
    *   這是早期將 Sequence-to-Sequence 模型成功應用於語音辨識的里程碑式工作。
    *   該模型雖然使用 LSTM 作為 Encoder 和 Decoder，但已經包含了 Cross-Attention 機制。
    *   論文名稱的含義: 機器「聽 (Listen)」聲音，然後透過「注意力 (Attend)」機制，最後「拼寫 (Spell)」出文字。
2.  **注意力可視化:** 該論文展示了 Cross-Attention 的注意力權重圖。
    *   當生成文字時 (例如 `H`, `O`, `W`)，注意力權重會隨著時間從語音輸入的左側（開頭）逐漸向右側（結尾）移動。
    *   這符合人類聽覺和發音的直觀理解：隨著語音播放，我們會依序專注於聲音訊號的不同部分來理解和生成文字。

### Attention可視化與行為模式

*   注意力權重圖通常會顯示從左上角向右下角移動的對角線模式，表示 Decoder 在生成輸出時，會逐步關注 Encoder 輸出的相應部分。
*   **多層連接:** 在原始 Transformer 論文中，Decoder 的每一層 Cross-Attention 都會使用 Encoder 最後一層的輸出作為 K 和 V。但這並非唯一方式，研究者也在探索更多樣化的連接方法。

## Decoder模型訓練

我們之前討論的是模型訓練完成後的「推論 (Inference)」。現在來看看如何訓練 Decoder。

### 訓練資料與目標

1.  **訓練資料:** 需要大量的輸入-輸出對，例如語音-文本對。
    *   例如，輸入一段語音說「機器學習」，對應的正確輸出是文本「機器學習」。
2.  **目標:** 訓練模型，使其在給定特定輸入後，能夠輸出與正確答案盡可能接近的序列。

### 損失函數 (Cross-Entropy)

1.  **類似分類任務:** Decoder 的訓練可以被視為一系列的分類任務。
    *   在每個時間步，Decoder 嘗試從詞彙表中「分類」出正確的下一個詞彙。
    *   例如，有 4000 個中文字，這就是一個 4000 類別的分類問題。
2.  **計算交叉熵 (Cross-Entropy):**
    *   **目標:** 讓 Decoder 輸出分佈 (Softmax 後) 盡可能接近正確詞彙的 **One-hot Vector** (Ground Truth)。
    *   **損失:** 計算 Decoder 輸出分佈與 Ground Truth 之間的交叉熵，並最小化這個值。交叉熵值越小，表示輸出分佈與正確答案越接近。
3.  **多個分類任務:**
    *   對於一個長度為 L 的輸出序列，將有 L 個詞彙需要分類。
    *   總損失是每個詞彙分類任務的交叉熵之和。
    *   **EOS 訓練:** 訓練時也要讓模型學習在正確的時機輸出 `end` 符號，因此在序列末尾，也會有一個對 `end` 符號的交叉熵計算。

### Teacher Forcing (訓練與推論的不一致)

1.  **訓練時的輸入:** 在訓練期間，Decoder 在生成下一個詞彙時，其輸入總是使用 **Ground Truth (正確答案)**，而不是它自己前一步的預測輸出。
    *   例如，給定 `begin`，希望輸出 `機`。訓練時，即使 Decoder 輸出了錯誤的詞彙，下一步的輸入仍然是 `begin` 和 **正確的 `機`**，而不是 Decoder 錯誤的輸出。
2.  **命名:** 這種訓練方式被稱為 **Teacher Forcing**。
3.  **問題點:**
    *   **訓練:** Decoder 總是看到「正確」的輸入。
    *   **推論:** Decoder 看到的是「自己」的輸出，如果它犯錯，就會看到「錯誤」的輸入。
    *   這種訓練與推論之間輸入的 **不一致性 (Mismatch)** 稱為 **Exposure Bias**。
    *   Exposure Bias 會導致模型在推論時，一旦犯錯，就很難恢復，因為它從未見過「錯誤」的輸入狀態。

## Decoder模型訓練與推論技巧

### Copy Mechanism (複製機制)

在某些任務中，Decoder 不僅需要生成新詞，還需要從輸入中「複製」一些詞彙。

1.  **應用場景:**
    *   **聊天機器人 (Chatbot):** 當用戶說「我是 **Kololo**」，機器人回覆「你好 **Kololo**，很高興認識你」。機器人不需要從零創造 `Kololo` 這個詞，直接從輸入複製更合理。
    *   **摘要 (Summarization):** 摘要文章時，很多關鍵詞或短語都是直接從原文複製而來。
2.  **優點:** 降低模型生成不常見詞彙的難度，提高生成正確率。
3.  **相關技術:**
    *   **Pointer Network (指針網絡):** 早期解決這類問題的模型。
    *   **Copy Network (複製網絡):** 允許模型選擇生成新詞或複製輸入詞。

### Guided Attention (引導式注意力)

有時模型的注意力行為不符合預期，例如語音合成時跳字。我們可以引導注意力。

1.  **問題背景:** 在語音辨識或語音合成中，如果模型在輸入中「漏掉」或「跳過」了某部分資訊，這會導致結果的嚴重錯誤 (例如，語音合成時漏讀一個字)。
    *   例如，輸入「發財」四次，但模型只輸出「發財」三次。
2.  **適用場景:** 尤其適用於語音辨識和語音合成，因為這些任務對輸入信息的完整處理有較高的要求。
3.  **核心思想:** 強制模型的注意力 (Cross-Attention) 按照某種預期模式運行。
    *   在語音合成/辨識中，理想的注意力應該是從輸入序列的「左到右」逐步移動。
4.  **實現方式 (關鍵詞):**
    *   **Monotonic Attention (單調注意力)**
    *   **Location-aware Attention (位置感知注意力)**
    *   這些技術旨在在訓練過程中引入額外的損失或約束，鼓勵注意力行為從左到右、有秩序地進行。

### Beam Search (集束搜索)

在推論時，如何從 Decoder 輸出的詞彙機率分佈中選擇最優序列？

1.  **Greedy Decoding (貪婪解碼):**
    *   在每個時間步，只選擇當前機率最高的詞彙作為輸出。
    *   **缺點:** 局部最優不代表全局最優。即使第一步選擇的詞彙機率稍低，但它可能導向一個整體機率更高的序列。
    *   **比喻:** 讀博士。短期選擇 (找工作) 看似收益高，但長期選擇 (讀博) 可能帶來更好的未來。
2.  **Beam Search (集束搜索):**
    *   維護一個固定大小的「集束 (Beam)」，在每個時間步，保留前 K 個（K 為 Beam Size）機率最高的候選序列。
    *   從這些候選序列中，再擴展生成下一個詞彙，並重新選擇前 K 個最優序列。
    *   **優點:** 找到比 Greedy Decoding 更好的近似解，但不是窮舉所有可能路徑。
    *   **缺點:** 計算成本更高；且有研究指出，在某些生成式任務中 (如文本生成)，Beam Search 可能導致重複和退化（Degeneration），生成不自然的文本。
    *   **適用性:**
        *   **有明確答案的任務 (如語音辨識):** 通常更有效，因為目標是找到唯一正確的文本。
        *   **需要創造性的任務 (如文本補全、故事生成):** 可能效果不佳，因為它傾向於選擇「安全」的、重複的選項，抑制了模型的創造性。

### Decoder中的隨機性 (Randomness)

在某些任務中，推論時引入隨機性反而能提升效果，這反直覺但真實。

1.  **反直覺的例子：語音合成 (Text-to-Speech, TTS)**
    *   直觀上，在測試時加入雜訊應該會讓結果變差。
    *   但實際經驗發現，在 TTS 的 Decoder 中引入一些隨機雜訊，可以讓生成的語音聽起來更自然、不那麼像機器槍掃射。
    *   **原因:** 完美的聲音可能不存在於訓練數據中，適度的不完美（隨機性）反而能捕捉人類語音的自然變化。
2.  **與 Dropout 的區別:** 訓練時的 Dropout 是為了讓模型更魯棒，而這裡討論的是推論時在 Decoder 內部加入隨機因素。

### 訓練目標與評估指標的差距 (Cross-Entropy vs. BLEU Score)

1.  **訓練損失 (Cross-Entropy):** 計算每個單詞的機率與其真實標籤之間的差異。
2.  **評估指標 (BLEU Score):** 將整個生成的句子作為一個整體，與正確答案進行比較，評估翻譯質量等。
3.  **問題:** 最小化交叉熵並不保證最大化 BLEU Score，因為它們是兩個不同的函數。
    *   在實作中，通常會選擇在驗證集上 BLEU Score 最高的模型，而不是交叉熵最低的模型。
4.  **解決方案：強化學習 (Reinforcement Learning, RL)**
    *   如果評估指標 (如 BLEU Score) 是不可微分的，無法直接作為損失函數進行優化，可以將其視為 RL 中的 **獎勵 (Reward)**。
    *   將 Decoder 視為一個 **Agent**，通過 RL 的方法來優化這個不可微分的獎勵。這是一個較為困難但有效的方向。

### Exposure Bias 與 Scheduled Sampling

回顧 `Teacher Forcing` 導致的 `Exposure Bias` 問題：訓練時 Decoder 看到的是正確答案，推論時卻看到自己的（可能錯誤的）輸出。

1.  **問題根源:** 模型在訓練期間從未見過錯誤的輸入，導致在推論時一旦犯錯，便會非常「驚訝」，進而導致錯誤傳播。
2.  **解決方案：Scheduled Sampling (排程採樣)**
    *   在訓練過程中，Decoder 的輸入會以某種機率選擇使用 **Ground Truth** 或 **模型自己的預測輸出**。
    *   隨著訓練的進行，使用模型自身輸出的機率會逐漸增加。
    *   這樣，模型在訓練時就能逐漸適應看到自己可能錯誤的輸出，從而減少 Exposure Bias。
    *   **備註:** `Scheduled Sampling` 對於 LSTM 等 RNN 模型較為有效，但對 Transformer 的平行化能力有潛在影響，需注意實作細節。且 Transformer 中的 Scheduled Sampling 與 RNN 中的可能有所不同。

## 總結

本堂課詳細介紹了 Decoder 的核心概念、Autoregressive 和 Non-Autoregressive 兩種主要類型，以及它們的運作機制。我們深入探討了 Transformer Decoder 的結構，特別是 Masked Self-Attention 和連接 Encoder 的 Cross-Attention。此外，課程還涵蓋了 Decoder 模型訓練的基礎（如損失函數與 Teacher Forcing），以及多種優化訓練與推論的實用技巧，包括 Copy Mechanism、Guided Attention、Beam Search、Decoder 中的隨機性，以及如何處理訓練目標與評估指標不一致的問題（如透過強化學習）。最後，我們重新審視了 Exposure Bias 問題及其解決方案 Scheduled Sampling。

## 隨堂測驗

### 測驗一：Autoregressive (AR) Decoder 與 Non-Autoregressive (NAR) Decoder 的比較

請說明 AR Decoder 和 NAR Decoder 在運作機制上的主要差異，並各自列出一個優點和一個缺點。

<details>
<summary>點擊展開解答</summary>

**Autoregressive (AR) Decoder**
*   **運作機制:** 逐步生成輸出序列，每個時間步的輸出都依賴於前一個時間步的模型輸出。
*   **優點:** 通常性能較好，生成序列的連貫性和語法正確性更高。
*   **缺點:** 由於是序列式生成，無法實現完全平行化，推論速度相對較慢，尤其對於長序列。

**Non-Autoregressive (NAR) Decoder**
*   **運作機制:** 一次性生成整個輸出序列，不依賴於前一個時間步的輸出。
*   **優點:** 可以實現高度平行化，推論速度快，尤其適合生成長序列。
*   **缺點:** 性能通常不如 AR 模型，可能在生成質量和連貫性方面存在挑戰（例如 Multi-modality 問題）。

</details>

---

### 測驗二：Masked Self-Attention 的目的

在 Transformer Decoder 中，為什麼需要引入 Masked Self-Attention (遮罩式自我注意力) 機制？這與 Encoder 中的 Self-Attention 有何不同？

<details>
<summary>點擊展開解答</summary>

**Masked Self-Attention 的目的：**
Masked Self-Attention 的目的是為了模擬 Decoder 逐步生成輸出序列的過程。當 Decoder 在生成第 `i` 個詞彙時，它只能存取和利用輸入序列中以及之前已經生成詞彙的資訊，而不能「偷看」尚未生成或未來的資訊。這符合 Decoder 的因果性（Causality）要求。

**與 Encoder 中 Self-Attention 的不同：**
*   **Encoder 中的 Self-Attention:** 每個輸出元素（或其隱藏狀態）在生成時，可以考慮完整的輸入序列中的所有元素（即雙向上下文）。
*   **Decoder 中的 Masked Self-Attention:** 每個輸出元素在生成時，只能考慮它自身以及之前位置的元素，而不能考慮它之後位置的元素（即單向上下文）。這個「遮罩」操作會將未來位置的注意力權重設為一個極小的負值（例如負無窮大），使其在 Softmax 後的機率為零，從而有效阻止模型在生成時「看到未來」。

</details>

---

### 測驗三：Exposure Bias 與 Scheduled Sampling

什麼是 **Exposure Bias**？它是在模型訓練的哪個環節產生的？請簡述 **Scheduled Sampling** 如何嘗試解決這個問題。

<details>
<summary>點擊展開解答</summary>

**Exposure Bias (曝險偏差):**
Exposure Bias 是指在 Sequence-to-Sequence 模型訓練和推論之間存在輸入不一致的問題。
*   **訓練時:** 模型通常採用 **Teacher Forcing**，即 Decoder 在生成序列時，總是接收到 **正確的（ground truth）** 前一個詞彙作為輸入。
*   **推論時:** Decoder 卻是接收 **自己前一步的預測輸出** 作為輸入。

這種訓練與推論之間的輸入差異導致模型在訓練時從未見過「錯誤」的輸入狀態。一旦在推論時模型生成了一個錯誤的詞彙，這個錯誤就可能因為模型從未學習過如何處理這種錯誤輸入而傳播開來，導致後續生成也出現問題。

**Scheduled Sampling (排程採樣) 如何解決 Exposure Bias:**
Scheduled Sampling 旨在緩解 Exposure Bias 問題。它在訓練過程中，讓 Decoder 的輸入以一定的機率選擇使用 **ground truth (正確答案)** 或 **模型自己前一步的預測輸出**。
*   **機制:** 隨著訓練的進行，使用模型自身預測輸出作為輸入的機率會逐漸增加。
*   **目的:** 這樣做使得模型在訓練過程中，也能逐漸接觸並學習如何處理自己可能犯錯時所產生的「不正確」輸入，從而提高其在推論時面對錯誤的魯棒性，減少錯誤傳播的影響。

</details>

---
```mermaid
graph TD
    A[課程主題: Decoder] --> B{兩種Decoder類型}

    B --> C[Autoregressive (AR) Decoder]
    C --> AR1[運作機制: 逐步生成]
    AR1 --> AR2[輸入: BOS + 前一個輸出]
    AR2 --> AR3[輸出: 詞彙分數分佈 (Softmax)]
    AR3 --> AR4[選擇最高分詞彙]
    AR4 --> AR5[循環直到 EOS]
    C --> AR6[內部結構: Transformer Decoder]
    AR6 --> AR7[特點: Masked Self-Attention]
    C --> AR8[潛在問題: 錯誤傳播]

    B --> D[Non-Autoregressive (NAT) Decoder]
    D --> NAT1[運作機制: 一次生成整個序列]
    NAT1 --> NAT2[輸入: 多個Begin Tokens]
    NAT2 --> NAT3[輸出: 整個序列]
    D --> NAT4[優點: 平行化, 速度快, 長度控制]
    D --> NAT5[缺點: 性能較差, Multi-modality]
    D --> NAT6[解決長度問題: 額外分類器 或 固定長度+EOS]

    E[Encoder] --> F[Cross-Attention]
    AR6 --> F
    F --> CA1[Query (Q) 來自 Decoder]
    F --> CA2[Key (K) & Value (V) 來自 Encoder]
    CA2 --> CA3[Encoder輸出]
    F --> CA4[功能: Decoder讀取Encoder資訊]

    G[模型訓練] --> T1[訓練資料: (輸入, 正確輸出) 對]
    G --> T2[損失函數: Cross-Entropy]
    G --> T3[訓練方法: Teacher Forcing]
    T3 --> T4[問題: Exposure Bias]
    T4 --> T5[解決方案: Scheduled Sampling]

    G --> T6[推論方法: Greedy Decoding / Beam Search]
    T6 --> T7[Greedy Decoding: 局部最優]
    T6 --> T8[Beam Search: 近似解, 保留多個候選]
    T8 --> T9[Beam Search的局限性: 有時無用, 抑制創意]

    H[訓練與推論技巧] --> K1[Copy Mechanism]
    K1 --> K1_1[應用: Chatbot, 摘要]
    K2[Guided Attention]
    K2 --> K2_1[應用: 語音辨識, 語音合成]
    K2_1 --> K2_2[強制Attention行為: 左到右]
    K3[Decoder隨機性]
    K3 --> K3_1[應用: 語音合成 (TTS)]
    K3_1 --> K3_2[目的: 提高生成品質, 避免機器感]
    K4[優化目標與評估指標不一致]
    K4 --> K4_1[Cross-Entropy vs. BLEU Score]
    K4_1 --> K4_2[解決方案: 強化學習 (RL)]