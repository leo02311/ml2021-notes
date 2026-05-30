---
title: "第15堂課：[ML 2021 (English version)] Lecture 11: Self-attention (2/2)"
tags:
  - MachineLearning
  - ML2021
---

# 【機器學習 2021】第15堂課：[ML 2021 (English version)] Lecture 11: Self-attention (2/2)

## 1. 自注意力機制 (Self-Attention Mechanism) 概述

自注意力機制的核心思想是讓模型在處理序列中的每個元素時，能夠考慮到序列中所有其他元素的相關性，並根據相關性給予不同的權重。與傳統的序列模型（如 RNN）不同，自注意力機制能夠同時處理序列中的所有元素，實現高度並行化。

### 1.1 `b_i` 的計算流程 (以 `b2` 為例)

自注意力機制並不需要依序計算 `b1` 到 `b4`，這些輸出向量是**同時計算**的。以下以 `b2` 的計算為例：

1.  **生成 `q_i`, `k_j`, `v_j` 向量**
    *   將輸入向量 `a_i` 乘上不同的轉換矩陣 `Wq`, `Wk`, `Wv`，分別得到 Query 向量 `q_i`、Key 向量 `k_i` 和 Value 向量 `v_i`。
    *   例如，`a2` 乘上 `Wq` 得到 `q2`。
    *   對於所有輸入 `a1` 到 `a4`，都會生成對應的 `q_i`, `k_i`, `v_i` (即 `q1..q4`, `k1..k4`, `v1..v4`)。

2.  **計算注意力分數 (Attention Scores)**
    *   使用 `q2` 與所有 `k_j` 向量進行點積 (dot product) 計算。
    *   例如：`q2` · `k1`, `q2` · `k2`, `q2` · `k3`, `q2` · `k4`。
    *   這些點積結果即為注意力分數，表示 `a2` 與其他 `a_j` 的相關程度。

3.  **歸一化注意力分數**
    *   對上述四個注意力分數進行歸一化，例如使用 Softmax 函數，使它們的和為 1。
    *   歸一化後的注意力分數表示為 `α'2,1`, `α'2,2`, `α'2,3`, `α'2,4`。

4.  **計算加權和 (Weighted Sum)**
    *   將歸一化後的注意力分數分別乘上對應的 Value 向量 `v_j`。
    *   將所有這些加權後的 `v_j` 向量加總，即可得到輸出向量 `b2`。
    *   `b2 = α'2,1 * v1 + α'2,2 * v2 + α'2,3 * v3 + α'2,4 * v4`

## 2. 自注意力機制：矩陣運算視角

上述逐點計算過程可以高效地透過矩陣乘法來實現。

### 2.1 生成 Q, K, V 矩陣

1.  **輸入矩陣 `I`**: 將所有輸入向量 `a1` 到 `a4` 堆疊成一個矩陣 `I`，其中每個 `a_i` 作為 `I` 的一欄。
2.  **生成 Query 矩陣 `Q`**: `Q = I * Wq`
3.  **生成 Key 矩陣 `K`**: `K = I * Wk`
4.  **生成 Value 矩陣 `V`**: `V = I * Wv`
    *   `Wq`, `Wk`, `Wv` 是模型需要學習的網絡參數。`Q`, `K`, `V` 矩陣的每一欄分別對應 `q_i`, `k_i`, `v_i` 向量。

### 2.2 計算注意力分數矩陣 `A`

*   將 `K` 矩陣轉置 (transpose) 為 `K^T`，然後與 `Q` 矩陣相乘：
    `A = Q * K^T`
    *   這個矩陣 `A` 的每個元素 `A_ij` 都代表 `q_i` 和 `k_j` 的點積，即 `a_i` 對 `a_j` 的注意力分數。

### 2.3 歸一化注意力分數矩陣 `A'`

*   對 `A` 矩陣的每一列（或每一行，根據具體實現）獨立地應用 Softmax 函數進行歸一化。
*   歸一化後的結果表示為 `A'` (有時稱為 Attention Matrix)。

### 2.4 計算輸出矩陣 `O`

*   將 Value 矩陣 `V` 與歸一化後的注意力分數矩陣 `A'` 相乘：
    `O = V * A'`
    *   `O` 矩陣的每一欄即為對應的輸出向量 `b_i` (例如，`O` 的第一欄是 `b1`，第二欄是 `b2` 等)。

### 2.5 可學習參數

在整個自注意力層中，唯一需要透過訓練資料學習的參數是轉換矩陣 `Wq`, `Wk`, `Wv`。

## 3. 多頭自注意力機制 (Multi-Head Self-Attention)

### 3.1 緣由：多樣的相關性

*   單一的 Query `q` 可能只能捕捉一種「相關性」的定義。
*   然而，不同的相關性類型可能同時存在（例如，語法上的相關性、語義上的相關性）。
*   因此，我們需要多個「頭」(heads) 來同時捕捉不同類型的相關性。

### 3.2 運作方式

1.  **生成多個 `Q, K, V` 轉換**
    *   對於每個輸入 `a_i`，不再只生成一組 `q_i, k_i, v_i`。
    *   而是生成多組，例如兩頭：`q_i1, k_i1, v_i1` 和 `q_i2, k_i2, v_i2`。
    *   這可以透過在原始 `Wq, Wk, Wv` 矩陣之後，再引入各自獨立的轉換矩陣來實現 (例如 `Wq_head1`, `Wq_head2` 等)。

2.  **獨立計算注意力**
    *   每個頭獨立地進行自注意力計算。
    *   例如，第一組 (`q_i1, k_j1, v_j1`) 會計算出 `b_i1`。
    *   第二組 (`q_i2, k_j2, v_j2`) 會計算出 `b_i2`。
    *   重點是，`q_i1` 只會關注 `k_j1`，不會關注 `k_j2`。

3.  **連接並轉換輸出**
    *   將所有頭的輸出向量 (`b_i1, b_i2, ...`) 連接 (concatenate) 起來。
    *   然後將連接後的向量通過一個最終的線性轉換矩陣，得到最終的輸出 `b_i`。

## 4. 位置編碼 (Positional Encoding)

### 4.1 問題：缺乏位置信息

*   自注意力機制在設計上**不包含任何位置信息**。無論輸入向量 `a_i` 在序列中的哪個位置，其計算過程都是相同的。
*   例如，`a1` 和 `a4` 之間的「距離」在自注意力眼中與 `a2` 和 `a3` 之間沒有區別。
*   然而，在許多任務中（例如詞性標註），位置信息非常重要。

### 4.2 解決方案：加入位置向量

*   為每個位置定義一個獨特的**位置向量 `e_i`**。
*   將這個位置向量 `e_i` **加到**對應的輸入向量 `a_i` 上：`a'_i = a_i + e_i`。
*   這樣，處理後的輸入 `a'_i` 就包含了其所在位置的信息。

### 4.3 位置向量的生成方式

1.  **手動設計 (Handcrafted)**
    *   最早的 Transformer 模型 (Attention Is All You Need) 使用了基於 Sine 和 Cosine 函數的數學公式來生成位置向量。
    *   這種方法的優點是具有外推性，即使面對比訓練時更長的序列也能生成位置信息。
2.  **從資料學習 (Learned Positional Encoding)**
    *   將位置向量視為模型的參數，讓模型在訓練過程中自行學習這些向量。
    *   這是一個活躍的研究領域，有許多新的方法被提出，例如基於 RNN 生成或透過其他網絡結構學習。

## 5. 自注意力機制的應用與比較

### 5.1 語音處理 (Speech Processing)

*   **挑戰**: 語音訊號轉換成向量序列後，長度可能非常大 (例如，1秒音訊 = 100向量，一個詞 = 數千向量)。
    *   自注意力機制的計算複雜度是序列長度 `L` 的平方 (`O(L^2)`)。
    *   這會導致高昂的計算成本和記憶體消耗。
*   **解決方案**: **截斷式自注意力 (Truncated Self-Attention)**
    *   限制每個 Query `q_i` 只關注其周圍一定範圍內的 Key `k_j`，而不是整個序列。
    *   關注範圍 (window size) 可由人為設定。
    *   原理是：在語音識別中，通常只需考慮局部上下文即可判斷音素。

### 5.2 影像處理 (Image Processing)

*   **應用方式**: 將圖像視為一個向量集合。
    *   例如，一個 5x10 像素的 RGB 圖像，可以看作是 50 個 3 維向量的集合 (每個像素點一個向量)。
    *   一旦圖像被表示為向量集合，就可以應用自注意力機制。
*   **與卷積神經網絡 (CNN) 的比較**
    *   **CNN**: 具有固定的「感受野」(receptive field)。每個神經元只考慮其感受野範圍內的信息。感受野的範圍和大小是手動設計的。
    *   **自注意力**: 「感受野」是自動學習的。每個 Query 會根據注意力分數，自動決定要關注圖像中的哪些像素點，甚至可以是整個圖像。
    *   **關係**:
        *   CNN 可以被視為自注意力機制的一個「受限版本」(restricted version)。
        *   自注意力機制是 CNN 的「彈性版本」(flexible version)。
        *   已有數學證明指出，CNN 是自注意力機制的一個特例，通過適當的參數設計，自注意力可以模擬 CNN。
    *   **資料量與性能**:
        *   **資料量小**: CNN (限制性模型) 更不易過擬合，表現可能優於自注意力。
        *   **資料量大**: 自注意力 (彈性模型) 能從大量資料中學習更複雜的關係，表現最終超越 CNN。
        *   **案例**: Google 論文 "An Image Is Worth 16x16 Words" (ViT) 實驗證明，在千萬級以上訓練資料量時，自注意力 (Transformer) 表現優於 CNN。
    *   **結合**: **Conformer** 模型同時結合了自注意力與 CNN 的優勢，在一些任務中取得了更好的效果。

### 5.3 循環神經網絡 (Recurrent Neural Network, RNN)

*   **相似性**: 兩者都用於處理序列數據。自注意力的輸出向量序列和 RNN 的隱藏狀態序列，都可以經過全連接層進行後續處理。
*   **關鍵差異**:
    1.  **長距離依賴性 (Long-Range Dependency)**
        *   **RNN**: 訊息需要從序列的開頭依序傳遞到結尾。長序列中容易出現梯度消失/爆炸，導致「遺忘」遠距離信息。即使是雙向 RNN，也無法從根本上避免這個問題。
        *   **自注意力**: 直接計算任何兩個位置之間的相關性。無論距離多遠，只要 Query 和 Key 匹配，就能輕易提取信息，沒有「遺忘」問題。
    2.  **並行化 (Parallelization)**
        *   **RNN**: 由於其循環結構，每個時間步的計算都依賴前一個時間步的輸出，因此無法並行計算，必須依序進行。
        *   **自注意力**: 序列中的所有輸出向量 (`b1` 到 `b4`) 都可以**同時**計算，因為它們的計算彼此獨立。這大大提高了計算效率。
*   **趨勢**: 由於並行化和長距離依賴處理能力的優勢，自注意力機制在許多應用中逐漸取代了 RNN。

### 5.4 圖神經網絡 (Graph Neural Network, GNN)

*   **應用方式**: 將圖中的每個節點視為一個向量。
*   **特點**: 圖數據除了節點向量外，還有「邊」信息，表示節點之間的連接關係。
*   **自注意力與圖結構**:
    *   在圖上應用自注意力時，可以利用邊信息來限制注意力計算。
    *   例如，只計算有邊連接的節點之間的注意力分數，未連接的節點注意力分數直接設為零。
    *   這是一種利用領域知識 (domain knowledge) 的方式，因為邊已經暗示了節點之間的相關性。
*   **關係**: 這種在圖結構上應用的自注意力機制，其實就是一種**圖神經網絡**。GNN 本身是一個非常複雜且龐大的領域。

## 6. 進階與高效自注意力 (Efficient Transformers)

### 6.1 挑戰：計算成本

*   自注意力機制的主要缺點依然是其 `O(L^2)` 的計算複雜度，在處理非常長的序列時會造成性能瓶頸。

### 6.2 解決方案：多樣化的 `xxx-formers`

*   為了降低計算量，學術界提出了各種改進版本的自注意力機制，它們通常以 `xxx-former` 命名 (例如 Linformer, Performer, Reformer 等)。
*   這些變體旨在減少注意力矩陣的計算量或儲存空間。

### 6.3 性能與速度的權衡

*   通常情況下，加速的 `xxx-former` 版本會犧牲一些模型性能。
*   目前的研究目標是尋找能夠在保持高性能的同時，大幅提高計算效率的自注意力變體。

## 隨堂測驗 (Quiz)

### 測驗一

在自注意力機制中，計算輸出向量 `b_i` 涉及到哪些核心步驟？請列出主要步驟並簡述其目的。
<details>
<summary>解答</summary>
核心步驟如下：
1.  **生成 Query (Q), Key (K), Value (V) 向量**：將輸入向量 `a_i` 轉換為 `q_i, k_i, v_i`，分別用於查詢相關性、被查詢以及提供加權和內容。
2.  **計算注意力分數**：將 Query `q_i` 與所有 Key `k_j` 進行點積，量化 `a_i` 與 `a_j` 之間的相關性。
3.  **歸一化注意力分數**：使用 Softmax 等函數將注意力分數歸一化，使其總和為1，作為加權的權重。
4.  **計算加權和**：將歸一化後的注意力分數與對應的 Value 向量 `v_j` 相乘並加總，得到最終的輸出向量 `b_i`。
</details>

### 測驗二

自注意力機制與卷積神經網絡 (CNN) 在影像處理應用上有何異同？特別是在處理感受野和對訓練資料量的需求方面，兩者有何主要區別？
<details>
<summary>解答</summary>
**相同點**：兩者都能應用於影像處理，並從影像中提取特徵。
**不同點**：
*   **感受野 (Receptive Field)**：
    *   **CNN**：具有固定且人為設計的感受野。每個濾波器只關注其局部範圍內的像素信息。
    *   **自注意力**：具有動態且自動學習的感受野。它會根據輸入數據自動決定要關注哪些相關的像素（甚至可以是整個圖像），而不限於局部區域。
*   **對訓練資料量的需求**：
    *   **CNN**：由於其結構具有較強的歸納偏置和限制，在資料量較少時不易過擬合，性能通常優於自注意力。
    *   **自注意力**：模型彈性較大，需要大量訓練資料才能充分發揮其潛力，避免過擬合。在資料量極大時，其性能往往能超越 CNN。
簡言之，CNN 可視為自注意力的一個受限版本，而自注意力是CNN的一個更彈性的版本。
</details>

### 測驗三

為何自注意力機制需要「位置編碼」(Positional Encoding)？位置編碼如何解決這個問題？
<details>
<summary>解答</summary>
*   **問題**：自注意力機制在設計上**不包含任何位置信息**。這意味著無論輸入序列中的元素位於哪個位置，其處理方式都是相同的，模型無法區分元素在序列中的順序或相對位置。然而，在許多任務（如語言理解）中，位置信息至關重要。
*   **解決方式**：為了解決這個問題，自注意力機制引入了「位置編碼」。它為序列中的每個位置定義一個獨特的**位置向量 `e_i`**。這個位置向量會被**加到**對應的輸入向量 `a_i` 上，形成新的輸入 `a'_i = a_i + e_i`。這樣，模型在處理 `a'_i` 時，就能夠同時獲取原始的內容信息和其所在位置的信息。
</details>

## 知識圖譜 (Mermaid Knowledge Graph)

```mermaid
graph TD
    subgraph 自注意力核心流程
        Input[輸入向量序列 (A)] --> PositionalEncoding[位置編碼];
        PositionalEncoding --> QKV_Generation(生成 Q, K, V 矩陣);
        QKV_Generation --> Q(Query 矩陣);
        QKV_Generation --> K(Key 矩陣);
        QKV_Generation --> V(Value 矩陣);
        QKV_Generation --- Wq_Param(Wq: Learnable Parameter);
        QKV_Generation --- Wk_Param(Wk: Learnable Parameter);
        QKV_Generation --- Wv_Param(Wv: Learnable Parameter);

        Q --> AttentionCalculation(計算注意力分數 A);
        K --> AttentionCalculation;
        AttentionCalculation --> AttentionNormalization(歸一化注意力分數 A');
        AttentionNormalization --> WeightedSum(計算加權和);
        V --> WeightedSum;
        WeightedSum --> Output[輸出向量序列 (O)];
    end

    subgraph 多頭自注意力
        QKV_Generation --> MultiHead_Split(將 Q, K, V 分割成多個頭);
        MultiHead_Split --> Head_One[頭 1 注意力];
        MultiHead_Split --> Head_N[頭 N 注意力];
        Head_One & Head_N --> ConcatHeads(連接所有頭的輸出);
        ConcatHeads --> LinearTransform(線性轉換);
        LinearTransform --> Output_MultiHead[多頭自注意力輸出];
        Output_MultiHead --- Output;
    end

    subgraph 位置編碼類型
        PositionalEncoding --> Handcrafted_PE(手動設計 (e.g., Sine/Cosine));
        PositionalEncoding --> Learned_PE(從資料學習);
    end

    subgraph 自注意力應用與比較
        Output --> App_Speech[語音處理];
        App_Speech --> Truncated_SelfAttention[截斷式自注意力];
        Output --> App_Image[影像處理];
        App_Image --> Compare_CNN(與 CNN 比較);
        Compare_CNN --> CNN_Restricted[CNN (受限且固定感受野)];
        Compare_CNN --> SA_Flexible[自注意力 (彈性且學習感受野)];
        Compare_CNN --> Data_Impact[資料量影響性能];
        Data_Impact --> Small_Data_CNN(小資料量: CNN 較優);
        Data_Impact --> Large_Data_SA(大資料量: 自注意力較優);
        App_Image --> Hybrid_Conformer[混合模型: Conformer];

        Output --> App_RNN[取代 RNN];
        App_RNN --> SA_Advantage[優勢: 並行化, 長距離依賴];
        Output --> App_Graph[圖資料 (GNN)];
        App_Graph --> Edge_Restriction[利用邊信息限制注意力];
    end

    subgraph 進階自注意力 (Efficient Transformers)
        Output --> Advanced_SA[優化計算成本 (xxx-formers)];
        Advanced_SA --> Speed_Performance_Tradeoff[速度 vs 性能權衡];
    end
```
