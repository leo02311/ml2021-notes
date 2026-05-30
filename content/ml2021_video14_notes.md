好的，身為專業的技術筆記整理專家，這就為您將李宏毅教授的機器學習課程逐字稿，整理成排版精美、結構清晰的 Markdown 筆記。

---
title: 第14堂課：自注意力機制 (Self-Attention) - 處理變長序列輸入
tags:
  - MachineLearning
  - ML2021
  - SelfAttention
  - Transformer
  - SequenceModeling
  - DeepLearning
---

# 【機器學習 2021】第14堂課：[ML 2021 (English version)] Lecture 10: Self-attention (1/2)

這堂課將介紹一種常見的神經網路架構：**自注意力機制 (Self-Attention)**。它主要用於解決輸入為變長序列向量的問題，這在許多機器學習任務中都非常普遍。

## 課程概述：自注意力機制 (Self-Attention)

### 為什麼需要處理序列輸入？

到目前為止，我們討論的神經網路輸入通常是**固定長度的向量**（例如 YouTube 觀看次數預測、圖像處理）。然而，現實世界中有許多問題的輸入是**一串向量 (a row of vectors)**，且這串向量的**長度是可變的**。

**問題核心：**
當模型輸入的向量序列長度不固定時，傳統的固定輸入尺寸網路無法直接處理。

### 序列資料的表示

教授列舉了幾種常見的變長序列輸入例子及其表示方法：

1.  **文字 (Text)**
    *   **概念：** 一個句子由多個單字組成，每個單字的長度不同。
    *   **表示：** 每個單字可以被表示為一個向量。
        *   **One-hot Encoding:**
            *   為每個單字分配一個極長的向量，其中只有一個維度為 1。
            *   **問題：** 無法體現單字間的語義關係（如 "cat" 和 "dog" 關係不比 "cat" 和 "apple" 近）。
        *   **Word Embedding:**
            *   為每個單字分配一個包含語義資訊的向量。
            *   具有語義相似性的單字在向量空間中會比較接近（如 "cat" 和 "dog" 會聚類在一起）。
            *   本課程不深入討論其生成方式，但可從網路下載預訓練模型。
    *   **結果：** 一個句子被表示為一串長度不一的向量。

2.  **語音 (Audio)**
    *   **概念：** 聲學訊號是連續的，需要將其切分成離散單元。
    *   **表示：**
        *   將語音訊號切成一個個**窗格 (window)**，通常為 25 毫秒。
        *   每個窗格的資訊被轉換成一個**向量 (vector)**，稱為 **音框 (frame)**。
        *   窗格通常會以 10 毫秒的步長滑動，形成一串音框。
        *   例如：1 秒語音包含 100 個音框，1 分鐘語音包含 6000 個音框。
    *   **結果：** 一段語音訊號被表示為一串向量。

3.  **圖形 (Graph)**
    *   **概念：** 圖形由節點和邊組成，節點數量可變。
    *   **表示：** 每個節點可以被看作一個向量。
        *   **社交網路：** 每個節點代表一個人，邊代表關係。節點向量可由個人資料（性別、年齡、職業、貼文等）組成。
        *   **分子 (Molecule)：** 每個球代表一個原子，原子之間有鍵結。每個原子可用 One-hot 向量表示（例如 H: [1,0,0,0], C: [0,1,0,0], O: [0,0,1,0]）。
    *   **結果：** 一個圖形被表示為一串向量。

### 序列模型的輸出類型

根據輸入與輸出的數量關係，序列模型大致可分為三種類型：

1.  **序列標註 (Sequence Labeling)：輸入與輸出長度相同**
    *   **描述：** 輸入 N 個向量，輸出 N 個標籤（每個輸入向量對應一個標籤）。
    *   **標籤類型：** 可以是數值 (迴歸) 或類別 (分類)。
    *   **範例：**
        *   **文字：** 詞性標註 (Part-of-Speech Tagging, POS Tagging)。例如，在 "I saw a saw" 中，第一個 "saw" 是動詞，第二個 "saw" 是名詞。模型需為每個單字輸出其詞性。
        *   **語音：** 為每個音框決定其音素類型。
        *   **圖形：** 預測社交網路中每個人的屬性（例如是否會購買某產品）。
    *   **本日課程重點：** 本堂課主要探討此類型。

2.  **序列分類 (Sequence Classification)：輸入 N 向量，輸出 1 個標籤**
    *   **描述：** 輸入整個序列，只輸出一個代表整個序列的標籤。
    *   **範例：**
        *   **文字：** 情感分析 (Sentiment Analysis)。判斷一個句子是正面還是負面評論。
        *   **語音：** 語者辨識 (Speaker Recognition)。辨識一段語音是誰說的。
        *   **圖形：** 判斷一個分子是否具有毒性或親水性。

3.  **序列到序列 (Sequence-to-Sequence, Seq2Seq)：輸入 N 向量，輸出 N' 標籤 (N' 由模型決定)**
    *   **描述：** 輸入 N 個向量，模型自行決定輸出的 N' 個標籤數量。輸入和輸出長度可能不同。
    *   **範例：**
        *   **機器翻譯：** 輸入一種語言的句子，輸出另一種語言的句子。
        *   **語音辨識：** 輸入一段語音，輸出文字段落。
    *   **後續課程重點：** 本堂課不深入探討，將在作業 5 和未來課程中再次討論。

## 傳統神經網路的限制與問題

針對**序列標註 (Sequence Labeling)** 任務，若嘗試使用傳統的全連接網路 (Fully Connected Network, FCN) 會有以下問題：

### 針對個別向量的全連接網路 (FCN)

*   **做法：** 將序列中的每個向量獨立輸入 FCN，各自得到輸出。
*   **問題：** 完全不考慮序列中各向量之間的**依賴性與上下文資訊**。
    *   例如，在詞性標註 "I saw a saw" 中，兩個 "saw" 的詞性不同，但如果獨立輸入 FCN，它們會得到相同的輸出，因為它們的輸入向量完全一樣。模型會無法區分。

### 考慮固定窗格的全連接網路

*   **做法：** 為了引入上下文資訊，將當前向量與其前後固定數量的向量（形成一個「窗格」）拼接起來，再輸入 FCN。
    *   例如，在作業 2 中，判斷一個音框的音素時，會考慮其前後 5 個音框，共 11 個音框的資訊。
*   **問題：**
    *   **窗格大小限制：** 無法捕捉到**整個序列的資訊**，只能處理局部上下文。
    *   **變長序列問題：**
        *   如果序列長度不固定，就難以定義一個「覆蓋整個序列」的固定窗格。
        *   若要覆蓋最長序列，窗格會變得非常大，導致 FCN 參數過多。
        *   **參數爆炸 (Computationally Expensive)：** 大量的參數不僅計算成本高，也容易造成**過度擬合 (Overfitting)**。

## 解決方案：自注意力機制 (Self-Attention)

**自注意力機制 (Self-Attention)** 旨在解決上述問題，它能：

*   **處理變長序列：** 不需要固定窗格。
*   **考慮全局上下文：** 每個輸出向量都考慮了整個輸入序列的資訊。

### 自注意力機制的工作流程

1.  **輸入：** 一串向量 `a1, a2, ..., aN`。這些可以是原始輸入，也可以是前面層次的隱藏層輸出。
2.  **輸出：** 數量與輸入相同的一串向量 `b1, b2, ..., bN`。
3.  **特性：** 每個輸出向量 `bi` 都是透過**考慮所有輸入向量 `a1` 到 `aN`** 後生成的。這意味著 `bi` 包含了整個序列的上下文資訊。
    *   例如，生成 `b1` 時，會考慮 `a1, a2, a3, a4` 的所有資訊。

### 自注意力機制的應用層次

自注意力機制可以與全連接網路交替使用，形成更複雜的網路架構：

*   `Self-Attention` -> `FCN` -> `Self-Attention` -> `FCN` -> ...
*   `Self-Attention` 負責處理整個序列的全局資訊。
*   `FCN` 負責處理特定位置的局部資訊。

### 與 Transformer 的關係

自注意力機制是 Google 在 **"Attention Is All You Need"** 這篇論文中提出的 **Transformer** 架構中最重要的模組之一。Transformer 的強大性能使其在自然語言處理等領域取得了巨大的成功。雖然自注意力概念在之前的一些論文中已有類似結構（如 Self-Matching），但 "Attention Is All You Need" 使其廣為人知並成為主流。

## 自注意力機制的工作原理

接下來將詳細解釋如何從輸入序列 `a1, ..., aN` 生成輸出向量 `b1`（此過程適用於生成所有 `bi`）。

### 核心概念：Query, Key, Value

自注意力機制的核心思想是：根據一個向量 (Query) 去**查詢 (query)** 其他向量 (Key) 的相關性，然後根據這些相關性從所有向量中**提取 (value)** 重要的資訊。

### 輸出向量 B_i 的生成流程 (以 b1 為例)

目標：生成向量 `b1`。

#### 步驟一：計算注意力分數 (Attention Score)

為了生成 `b1`，我們需要找出序列中其他向量與 `a1` 的**相關程度**。這個相關程度用一個**注意力分數 (attention score) `alpha`** 來表示。

1.  **生成 Query 向量 (q)**:
    *   將 `a1` 乘以一個權重矩陣 `W_Q`，得到 `q1`。
    *   `q1 = W_Q * a1`
    *   `q` 被稱為 **Query**，就像你在搜尋引擎中輸入的關鍵字。

2.  **生成 Key 向量 (k)**:
    *   將**所有**輸入向量 `a_j` (包括 `a1` 自己) 分別乘以另一個權重矩陣 `W_K`，得到 `k_j`。
    *   `k_j = W_K * a_j` (例如 `k1 = W_K * a1`, `k2 = W_K * a2`, ...)
    *   `k` 被稱為 **Key**，如同資料庫中的索引。

3.  **計算注意力分數 (alpha_ij)**:
    *   使用 `q1` 與所有的 `k_j` (包括 `k1`) 進行**點積 (Dot Product)** 運算，得到它們之間的注意力分數 `alpha_1j`。
    *   `alpha_1j = q1 . k_j` (點積)
    *   **常用計算方式：** 點積 (Dot Product) 是最常見且 Transformer 採用的方法。
    *   **其他計算方式：** 加法注意力 (Additive Attention) 等，但本課程主要聚焦點積。
    *   **實務考量：** `a1` 與 `k1` (自己與自己的相關性) 通常也會計算，這在作業中可以嘗試是否會影響結果。

#### 步驟二：Softmax 正規化

得到一系列的 `alpha_1j` 分數後，通常會對它們進行 Softmax 運算，得到**正規化後的注意力分數 `alpha'_1j`**。

*   `alpha'_1j = softmax(alpha_1j)`
*   Softmax 將所有分數轉換成一個介於 0 到 1 之間的**機率分佈**，且總和為 1。
*   **注意：** 雖然 Softmax 是常見做法，但並非強制。其他激活函數（如 ReLU）在某些情況下可能表現更好，這是可自行嘗試決定的超參數。

#### 步驟三：生成 Value 向量

從**所有**輸入向量 `a_j` 中提取**資訊內容**，形成 `v_j` 向量。

*   將**所有**輸入向量 `a_j` 分別乘以另一個權重矩陣 `W_V`，得到 `v_j`。
*   `v_j = W_V * a_j` (例如 `v1 = W_V * a1`, `v2 = W_V * a2`, ...)
*   `v` 被稱為 **Value**，代表了該向量所包含的實際資訊。

#### 步驟四：加權求和得到輸出

將正規化後的注意力分數 `alpha'_1j` 作為權重，對所有 `v_j` 進行**加權求和**，得到最終的輸出向量 `b1`。

*   `b1 = sum(alpha'_1j * v_j)` (從 `j=1` 到 `N`)
*   **意義：** 若 `alpha'_1j` 越高，表示 `a_j` 對 `a1` 越重要，其對 `b1` 的貢獻就越大。這使得 `b1` 能夠綜合整個序列中與 `a1` 最相關的資訊。

### Mermaid 知識圖譜

```mermaid
graph TD
    subgraph 核心概念與問題
        A[自注意力機制 Self-Attention] --> P1[處理變長序列輸入]
        A --> P2[考量全局上下文資訊]
        P1 --> P3[打破傳統網路固定輸入長度限制]
        P2 --> P4[超越固定窗格 FCN 限制]
        P3 & P4 --> B(傳統神經網路限制)
    end

    subgraph 變長序列輸入範例
        VLI[變長序列輸入]
        VLI --> VLI_Text[文字: 句子]
        VLI_Text --> VLI_Text_Rep[表示: 詞向量 (Word Embedding)]
        VLI_Text_Rep --> VLI_Text_Rep_Types[類型: One-hot, Word Embedding]
        
        VLI --> VLI_Audio[語音: 聲學訊號]
        VLI_Audio --> VLI_Audio_Rep[表示: 音框 (Frame)]
        
        VLI --> VLI_Graph[圖形: 社交網路 / 分子]
        VLI_Graph --> VLI_Graph_Rep[表示: 節點向量]
    end

    subgraph 序列模型輸出類型
        SMO[序列模型輸出]
        SMO --> SMO_Labeling[類型一: 序列標註 (Sequence Labeling)]
        SMO_Labeling --> SMO_Labeling_Desc[輸入 N 輸出 N (本日重點)]
        SMO_Labeling --> SMO_Labeling_Ex1[範例: 詞性標註 (POS Tagging)]
        SMO_Labeling --> SMO_Labeling_Ex2[範例: 音素分類]
        
        SMO --> SMO_Classification[類型二: 序列分類 (Sequence Classification)]
        SMO_Classification --> SMO_Classification_Desc[輸入 N 輸出 1]
        SMO_Classification --> SMO_Classification_Ex1[範例: 情感分析]
        SMO_Classification --> SMO_Classification_Ex2[範例: 語者辨識]
        
        SMO --> SMO_Seq2Seq[類型三: 序列到序列 (Seq2Seq)]
        SMO_Seq2Seq --> SMO_Seq2Seq_Desc[輸入 N 輸出 N' (未來討論)]
        SMO_Seq2Seq --> SMO_Seq2Seq_Ex1[範例: 機器翻譯]
        SMO_Seq2Seq --> SMO_Seq2Seq_Ex2[範例: 語音辨識]
    end

    subgraph 自注意力機制詳解 (以生成輸出 b1 為例)
        Input_Seq[輸入序列 a1, ..., aN]

        subgraph 步驟一: 計算注意力分數 (Alpha)
            Input_a1[當前輸入向量 a1] --> WQ_Matrix(權重矩陣 W_Q)
            WQ_Matrix --> Query_q1[Query q1]

            Input_All_aj[所有輸入向量 aj] --> WK_Matrix(權重矩陣 W_K)
            WK_Matrix --> Keys_kj[Keys k1, ..., kN]

            Query_q1 --> Dot_Product[點積運算]
            Keys_kj --> Dot_Product
            Dot_Product --> Alpha_Scores[注意力分數 α1j]
            Alpha_Scores -- 意義 --> Correlation_Degree[a1 與 aj 的相關程度]
        end

        subgraph 步驟二: Softmax 正規化
            Alpha_Scores --> Softmax_Function(Softmax 函數)
            Softmax_Function --> Alpha_Prime[正規化注意力分數 α'1j]
            Alpha_Prime -- 特性 --> Prob_Distribution[0-1 之間，總和為 1]
        end

        subgraph 步驟三: 生成 Value 向量
            Input_All_aj --> WV_Matrix(權重矩陣 W_V)
            WV_Matrix --> Values_vj[Values v1, ..., vN]
            Values_vj -- 意義 --> Extracted_Info[向量 aj 包含的資訊]
        end

        subgraph 步驟四: 加權求和得到輸出
            Alpha_Prime --> Weighted_Sum_Op[加權求和]
            Values_vj --> Weighted_Sum_Op
            Weighted_Sum_Op --> Output_b1[輸出向量 b1]
            Output_b1 -- 特性 --> Context_Aware[包含整個序列上下文資訊]
        end
    end

    subgraph 自注意力機制應用與相關
        SA_Application[自注意力機制應用]
        SA_Application --> SA_Stacking[可堆疊與 FCN 交替使用]
        SA_Application --> SA_Global_Context[處理全局資訊]

        SA_Related[自注意力機制相關]
        SA_Related --> Transformer_Arch[Transformer 架構]
        SA_Related --> Attention_Paper["Attention Is All You Need" 論文]
    end

    Input_Seq --> 步驟一
    Input_Seq --> 步驟三
    Output_b1 --> Output_Seq[輸出序列 b1, ..., bN]
    Output_Seq --> SMO_Labeling_Desc
```

## 隨堂測驗

### 測驗一

為什麼在處理自然語言的詞性標註（Part-of-Speech Tagging）任務時，僅使用獨立的全連接網路 (Fully Connected Network, FCN) 為每個單字判斷詞性會面臨困難？

<details>
<summary>點擊查看解答</summary>
當僅使用獨立的 FCN 時，每個單字都被視為獨立的輸入，模型無法考慮到單字在句子中的上下文資訊。例如，在 "I saw a saw" 這個句子中，兩個 "saw" 的詞性不同（一個是動詞，一個是名詞），但如果它們的輸入向量完全相同，FCN 會給出相同的輸出，導致無法正確區分詞性。因此，缺乏上下文理解是主要困難。
</details>

### 測驗二

自注意力機制在計算注意力分數 `alpha_ij` 時，使用了 Query (`q_i`) 和 Key (`k_j`) 向量。請解釋 Query、Key、Value 這三種向量各自在自注意力機制中的抽象意義或角色。

<details>
<summary>點擊查看解答</summary>
*   **Query (查詢 `q`)**：代表「我正在尋找什麼資訊」。它是當前向量（例如 `a_i`）的表示，用於尋找序列中其他相關的資訊。
*   **Key (鍵 `k`)**：代表「我擁有什麼資訊」。它是序列中所有其他向量（例如 `a_j`）的表示，用於與 Query 進行匹配，判斷相關性。
*   **Value (值 `v`)**：代表「我能提供什麼資訊」。它是序列中所有其他向量（例如 `a_j`）的資訊內容本身，當 Key 與 Query 匹配成功後，會根據相關程度從這些 Value 向量中提取資訊。

簡單來說，Query 向量去「問」Key 向量「你跟我有多像？」，然後根據 Key 的回答（注意力分數）來決定從 Value 向量中「拿取」多少資訊。
</details>

### 測驗三

在自注意力機制的工作原理中，計算出注意力分數 `alpha_1j` 後，通常會使用 Softmax 函數進行正規化，得到 `alpha'_1j`。課程中提到 Softmax 並非唯一選擇，也可以嘗試其他激活函數（如 ReLU）。請問，為什麼 Softmax 函數在這裡是常用的選擇？如果改用 ReLU，可能帶來什麼樣的潛在影響？

<details>
<summary>點擊查看解答</summary>
**Softmax 函數是常用選擇的原因：**
1.  **機率分佈特性：** Softmax 將所有注意力分數轉換為一個總和為 1 的機率分佈，每個 `alpha'_1j` 介於 0 到 1 之間。這使得 `alpha'_1j` 可以直接被解釋為加權求和時的「權重」或「重要性程度」，符合直覺。
2.  **梯度穩定性：** 在某些情況下，Softmax 有助於梯度的傳播。

**改用 ReLU 的潛在影響：**
1.  **非正規化權重：** ReLU 的輸出範圍是 `[0, infinity)`，它不會將分數正規化成機率分佈，且總和可能不為 1。這意味著每個 `alpha'_1j` 將是獨立的非負值，它們不再代表相對的機率，而是絕對的「重要性」或「激活程度」。
2.  **稀疏性：** ReLU 函數會將所有負值設為 0。這可能導致注意力權重變得更稀疏，即只有少數幾個具有正相關性的向量會被賦予權重，而其他向量的權重則為 0。這可能強化模型對特定關鍵資訊的專注，但也可能丟失某些細微的上下文資訊。
3.  **實驗性選擇：** 正如課程所提，激活函數的選擇往往是實驗性的。ReLU 在某些任務上可能因為其稀疏性和計算效率而帶來更好的性能。
</details>