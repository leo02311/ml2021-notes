---
title: "第13堂課：【機器學習2021】自注意力機制 (Self-attention) (下) - 自注意力機制 Self-Attention"
tags:
  - MachineLearning
  - ML2021
  - SelfAttention
  - Transformer
---

# 【機器學習 2021】第13堂課：【機器學習2021】自注意力機制 (Self-attention) (下)

這堂課深入講解了自注意力機制 (Self-Attention) 的運作原理，從向量層面逐步過渡到矩陣運算，並探討了其多種變形、應用以及與其他主流模型的比較。

## 一、自注意力機制 (Self-Attention) 運作流程

Self-Attention 旨在將輸入的一系列向量 $a^1, a^2, \dots, a^L$ 轉換為另一系列向量 $b^1, b^2, \dots, b^L$。這裡強調一點，所有的輸出向量 $b^1$ 到 $b^L$ 是**一次同時**被計算出來的，而非依序生成。

以計算 $b^2$ 為例，其操作流程如下：

1.  **產生 Query (查詢) 向量 $q^2$**:
    *   將輸入向量 $a^2$ 乘上一個轉換矩陣 $W^Q$ 得到查詢向量 $q^2$。
    *   $q^2 = W^Q a^2$

2.  **計算注意力分數 (Attention Score)**:
    *   $q^2$ 會與所有輸入向量 $a^1, \dots, a^L$ 所對應的 Key (鍵) 向量 $k^1, \dots, k^L$ 進行點積 (dot product) 運算，得到一系列注意力分數。
    *   $k^i = W^K a^i$ (其中 $W^K$ 是另一個轉換矩陣)
    *   分數計算範例：
        *   $q^2$ 與 $k^1$ 點積得到 $q^2 \cdot k^1$
        *   $q^2$ 與 $k^2$ 點積得到 $q^2 \cdot k^2$
        *   ...
        *   $q^2$ 與 $k^L$ 點積得到 $q^2 \cdot k^L$

3.  **分數正規化 (Normalization)**:
    *   將這些原始注意力分數進行正規化，例如透過 Softmax 函數，得到最終的注意力權重 $\alpha'$。
    *   例如，從 $q^2 \cdot k^1, q^2 \cdot k^2, \dots, q^2 \cdot k^L$ 透過 Softmax 得到 $\alpha'^2_1, \alpha'^2_2, \dots, \alpha'^2_L$。

4.  **加權求和 (Weighted Sum) 得到輸出 $b^2$**:
    *   將這些正規化後的注意力權重 $\alpha'$ 分別乘上所有輸入向量 $a^1, \dots, a^L$ 所對應的 Value (值) 向量 $v^1, \dots, v^L$，再將結果全部加總，即得到 $b^2$。
    *   $v^i = W^V a^i$ (其中 $W^V$ 是第三個轉換矩陣)
    *   $b^2 = \sum_{i=1}^{L} \alpha'^2_i v^i = \alpha'^2_1 v^1 + \alpha'^2_2 v^2 + \dots + \alpha'^2_L v^L$

這個過程對每個 $b^j$ 都一樣，只需將主角 $a^2$ 換成 $a^j$ 即可。

## 二、從矩陣角度看自注意力機制 (Self-Attention)

上述的步驟可以被簡化為一系列的矩陣乘法運算。

### 2.1 產生 Q, K, V 矩陣

1.  **輸入矩陣 I**: 將輸入向量 $a^1, \dots, a^L$ 橫向堆疊成一個矩陣 $I$。若每個 $a^i$ 是 $D$ 維向量，序列長度為 $L$，則 $I$ 的形狀通常為 $L \times D$ (每行代表一個向量)。

2.  **轉換矩陣 $W^Q, W^K, W^V$**: 這是 Self-Attention 層中唯一需要學習的參數。它們將輸入向量轉換為 Query, Key, Value 向量。
    *   $W^Q$: 形狀為 $D \times d_k$
    *   $W^K$: 形狀為 $D \times d_k$
    *   $W^V$: 形狀為 $D \times d_v$
    (其中 $d_k$ 是 Key/Query 的維度，$d_v$ 是 Value 的維度，通常 $d_k = d_v = D$)

3.  **計算 Q, K, V 矩陣**:
    *   $Q = I W^Q$ (Query 矩陣，形狀 $L \times d_k$)
    *   $K = I W^K$ (Key 矩陣，形狀 $L \times d_k$)
    *   $V = I W^V$ (Value 矩陣，形狀 $L \times d_v$)
    *   這些矩陣的每一行分別代表一個 $q^i, k^i, v^i$ 向量。

### 2.2 計算注意力分數矩陣 A

1.  **計算原始注意力分數矩陣 A**: 每個 Query 向量與每個 Key 向量進行點積，形成一個 $L \times L$ 的注意力分數矩陣。
    *   $A = Q K^T / \sqrt{d_k}$
    *   這裡除以 $\sqrt{d_k}$ 是為了縮放，防止點積結果過大，導致 Softmax 梯度過小。
    *   矩陣 $A$ 中的每個元素 $A_{ij}$ 代表 $q^i$ 與 $k^j$ 的關聯程度 ($q^i \cdot k^j$)。

2.  **正規化得到注意力權重矩陣 A'**: 對 $A$ 的每一行 (或每一列，取決於設計) 進行 Softmax 運算，確保權重和為 1。
    *   $A' = \text{Softmax}(A)$
    *   此 $A'$ 通常被稱為 **Attention Matrix**。

### 2.3 計算輸出矩陣 O

1.  **加權求和得到輸出矩陣 O**: 將注意力權重矩陣 $A'$ 乘上 Value 矩陣 $V$。
    *   $O = A' V$
    *   矩陣 $O$ 的形狀為 $L \times d_v$，其每一行 $O_i$ 即為對應輸入 $a^i$ 的輸出向量 $b^i$。

### 2.4 矩陣運算總結

整個 Self-Attention 層的操作可以概括為：
$$
O = \text{Softmax}\left(\frac{I W^Q (I W^K)^T}{\sqrt{d_k}}\right) I W^V
$$

其核心流程是：
`Input I` $\rightarrow$ `Q, K, V 矩陣` $\rightarrow$ `注意力分數矩陣 A` $\rightarrow$ `正規化注意力矩陣 A'` $\rightarrow$ `輸出矩陣 O`

值得注意的是，Self-Attention 層中**唯一**需要訓練的參數是 $W^Q, W^K, W^V$ 這三個轉換矩陣。

## 三、多頭自注意力機制 (Multi-head Self-Attention)

多頭 (Multi-head) Self-Attention 是 Self-Attention 的一種進階變形，在現代 Transformer 模型中被廣泛應用。

### 3.1 概念

*   **多種相關性**: 相關性可能有多種定義。例如，在句子中，一個詞可能與另一個詞有語法上的關聯，也可能與第三個詞有語義上的關聯。
*   **多個「頭」**: 為了捕捉這些不同種類的相關性，我們不只產生一個 $q, k, v$ 向量，而是產生多個「頭」的 $q, k, v$ 向量組，讓不同的頭負責學習不同類型的相關性。

### 3.2 運作流程

1.  **生成多個 Query, Key, Value 投影**:
    *   對每個輸入 $a^i$，不再只產生一組 $q^i, k^i, v^i$。
    *   例如，如果有 $H$ 個頭，每個 $a^i$ 會生成 $H$ 組 $q^{i,h}, k^{i,h}, v^{i,h}$ (其中 $h$ 是頭的編號)。
    *   這通常透過將 $a^i$ 乘上 $H$ 組不同的投影矩陣 ($W^{Q,h}, W^{K,h}, W^{V,h}$)，或將 $a^i$ 投影到更高維度後再分割成 $H$ 個部分來實現。

2.  **平行執行多個 Self-Attention**:
    *   每個頭 (e.g., 頭 1) 獨立執行 Self-Attention 流程：$q^{i,1}$ 只與 $k^{j,1}$ 計算注意力分數，然後用這些分數對 $v^{j,1}$ 進行加權求和，得到輸出 $b^{i,1}$。
    *   所有 $H$ 個頭同時進行這些運算，互不干擾。

3.  **拼接與最終轉換**:
    *   將所有頭的輸出向量 (e.g., $b^{i,1}, b^{i,2}, \dots, b^{i,H}$) 拼接 (concatenate) 起來。
    *   將拼接後的向量再通過一個最終的線性轉換矩陣 $W^O$ (Outpu t Projection)，得到最終的輸出 $b^i$。

### 3.3 超參數考量

*   **頭的數量 (Number of Heads)**: 這是需要調整的超參數。不同的任務可能需要不同數量的頭。例如，翻譯和語音辨識可能需要更多的頭，而某些簡單任務可能一個頭就足夠。

## 四、位置編碼 (Positional Encoding)

### 4.1 問題

*   **缺失位置資訊**: 傳統的 Self-Attention 機制對輸入序列中的每個向量 $a^i$ 都是獨立處理的，它無法區分一個向量是出現在序列的最前面還是最後面，或者與其他向量的相對距離。所有的位置對於 Self-Attention 而言都是「天涯若比鄰」。
*   **重要性**: 位置資訊在許多序列任務中至關重要，例如詞性標記時，知道一個詞在句首或句尾有助於判斷其詞性。

### 4.2 解決方案：將位置資訊嵌入輸入

1.  **生成位置向量 (Positional Vector)**: 為序列中的每個位置 $i$ 創建一個獨特的向量 $e^i$。
2.  **疊加到輸入向量**: 將這個位置向量 $e^i$ 加到對應位置的輸入向量 $a^i$ 上，形成新的輸入 $a'^i = a^i + e^i$。
3.  這樣，Self-Attention 層在處理 $a'^i$ 時，就能感知其在序列中的位置。

### 4.3 Positional Encoding 的實現

*   **Handcrafted (人工設計)**: 原始 Transformer 論文採用了正弦 (sin) 和餘弦 (cos) 函數來生成 $e^i$。這種方法的好處是不受序列長度限制。
*   **Learnable (可學習)**: 也可以將 Positional Encoding 向量設為模型的參數，讓模型透過訓練數據自動學習這些向量。
*   **研究熱點**: Positional Encoding 仍然是一個活躍的研究領域，不斷有新的方法被提出，例如基於 RNN 生成、或稱為 FLOATER 的方法。目前沒有一個公認的最佳方法。

## 五、Self-Attention 的應用與變形

### 5.1 語音辨識 (Speech Recognition) - 截斷式 Self-attention (Truncated Self-attention)

*   **問題**: 語音訊號轉換成的向量序列 (e.g., Mel-spectrogram) 長度非常可觀（1秒聲音約100個向量），導致標準 Self-Attention 的計算複雜度 $O(L^2)$ 過高，難以訓練。
*   **解決方案**: **截斷式 Self-attention**。在計算注意力時，只考慮輸入向量周圍一小段範圍內的向量，而非整個序列。
*   **依據**: 辨識一個音素往往只需要其前後少量語音資訊，無需整個句子上下文。這有助於大幅降低計算量和記憶體消耗。

### 5.2 影像處理 (Image Processing)

*   **觀點轉換**: Self-Attention 適用於輸入為「一排向量」的場景。雖然影像常被視為高維張量 (Tensor)，但也可以將其重新概念化為「一個向量的集合」。
    *   例如，一張 $W \times H$ 的彩色圖片，可以看作是 $W \times H$ 個像素點，每個像素點本身就是一個 3 維向量 (RGB)。
*   **應用**: Vision Transformer (ViT) 等模型已成功將 Self-Attention 應用於影像處理任務，顯示其泛化能力。

## 六、Self-Attention 與其他模型的比較

### 6.1 與卷積神經網路 (CNN) 的比較

*   **感受野 (Receptive Field)**:
    *   **CNN**: 每個神經元只考慮其**局部感受野**內的資訊，感受野的大小是人工設計的。
    *   **Self-Attention**: 透過注意力機制，可以考慮**整張影像**或整個序列的資訊 (全局感受野)，就像感受野是「自動學習」出來的，網絡自己決定哪些像素是相關的。
*   **關係**:
    *   數學上，CNN 可以被視為 Self-Attention 的一個特例 (subset)。Self-Attention 是更彈性 (flexible) 的 CNN，而 CNN 是受限制 (restricted) 的 Self-Attention。
    *   只要設定合適的參數，Self-Attention 就能模擬 CNN 的行為。
*   **數據量與過擬合**:
    *   **Self-Attention**: 彈性大，需要**更多**的訓練資料。數據量不足時容易過擬合。
    *   **CNN**: 彈性小 (具有更強的歸納偏置，inductive bias)，在訓練資料量較少時，通常表現會比 Self-Attention 更好，不易過擬合。
    *   實驗結果 (Google ViT 論文) 顯示，當訓練資料量極大時，Self-Attention 表現會超越 CNN。
*   **混合模型**: 實際應用中，可以結合兩者優點，例如 Conformer 模型同時使用 Self-Attention 和 CNN。

### 6.2 與遞迴神經網路 (RNN) 的比較

*   **處理方式**:
    *   **RNN**: 順序處理序列，前一個時間步的輸出會作為下一個時間步的輸入 (透過記憶體向量)。若要考慮遠端資訊，需要將資訊一路「帶」到最後，且容易有梯度消失/爆炸問題。
    *   **Self-Attention**: 可以直接在序列中任意兩個位置之間建立聯繫，無論距離多遠，都能直接抽取資訊 (天涯若比鄰)，沒有「遺忘」的問題。
*   **平行化 (Parallelization)**:
    *   **RNN**: 無法平行化。每個時間步的計算都依賴於前一個時間步的結果，必須依序進行。
    *   **Self-Attention**: 輸出序列中的每個向量都可以同時計算，高度平行化，因此運算效率更高。
*   **趨勢**: 由於其平行處理能力和捕捉長距離依賴的優勢，Self-Attention 在許多序列任務中逐漸取代了 RNN 的地位。
*   **關係**: Self-Attention 透過某些設計和限制，也可以模擬 RNN 的行為。

## 七、Self-Attention 在圖 (Graph) 上的應用

*   **圖即向量集合**: 圖 (Graph) 也可以被視為一組節點向量，因此 Self-Attention 也能應用於圖結構數據。
*   **結合邊資訊**: 在圖上應用 Self-Attention 時，除了節點本身的特徵向量外，我們還擁有邊 (Edge) 的資訊，這代表了節點之間的明確關聯。
*   **限制注意力範圍**: 可以利用圖的邊資訊來限制 Self-Attention 的計算。例如，只計算有邊相連的節點之間的注意力分數，對於沒有直接相連的節點，其注意力分數可直接設為 0。
*   **Graph Neural Network (GNN)**: 這種結合圖結構資訊的 Self-Attention，是某一種類型的圖神經網絡 (GNN)。它利用了圖中已知的關聯性，避免模型重新學習這些已知的關係。

## 八、Self-Attention 的變形與未來研究

### 8.1 挑戰：計算複雜度

*   Self-Attention 最大的問題在於其計算複雜度為 $O(L^2)$，這對於長序列來說是巨大的負擔。
*   因此，如何減少 Self-Attention 的計算量，是目前研究的重點方向。

### 8.2 高效變形 (Efficient Transformers)

*   **xxformer 家族**: 為了優化計算效率，產生了許多 Self-Attention 的變形，通常被冠以 "-former" 的名稱，如 Linformer, Performer, Reformer 等。
*   **速度與效能的權衡**: 這些高效變形通常能在速度上超越原始 Transformer，但往往會犧牲一定的模型效能 (performance)。
*   **未來研究**: 如何設計出既快又好 (兼顧速度與效能) 的 Self-Attention 變形，仍然是一個開放且活躍的研究問題。有許多論文 (如 "Long Range Arena" 和 "Efficient Transformers: A Survey") 專門探討和比較這些變形。

```mermaid
graph TD
    A[自注意力機制 Self-Attention] --> SA_Flow[運作流程]
    A --> SA_Matrix[矩陣視角]
    A --> MHSA[多頭自注意力機制]
    A --> PE[位置編碼]
    A --> App[應用與變形]
    A --> Comp[與其他模型比較]
    A --> Graph[在圖上的應用]
    A --> Future[變形與未來研究]

    SA_Flow --> Q_Vec[查詢 Q 向量]
    SA_Flow --> K_Vec[鍵 K 向量]
    SA_Flow --> V_Vec[值 V 向量]
    Q_Vec -- 點積計算分數 --> Attn_Scores[注意力分數]
    K_Vec -- 提供上下文 --> Attn_Scores
    Attn_Scores -- Softmax正規化 --> Alpha_Prime[注意力權重 α']
    V_Vec -- 加權求和 --> Output_B[輸出向量 b]
    Alpha_Prime -- 加權求和 --> Output_B

    SA_Matrix --> Input_I[輸入 I 矩陣]
    Input_I -- 乘以 Wq, Wk, Wv --> QKV_Matrix[Q, K, V 矩陣]
    QKV_Matrix -- Q K轉置計算 --> Raw_Attn_A[原始注意力矩陣 A]
    Raw_Attn_A -- Softmax正規化 --> Attn_Matrix_A_Prime[正規化注意力矩陣 A']
    Attn_Matrix_A_Prime -- 乘以 V 矩陣 --> Output_O[輸出 O 矩陣]
    QKV_Matrix -- Wq, Wk, Wv 可學習參數 --> SA_Matrix

    MHSA --> Diff_Heads[多個獨立的頭]
    MHSA --> Concat_Output[拼接各頭輸出]
    MHSA --> Linear_Transform[線性轉換 W^O]
    Diff_Heads -- 捕捉不同相關性 --> Attn_Scores

    PE --> Pos_Info[解決位置資訊缺失]
    PE --> Add_To_Input[將位置向量加到輸入]
    PE --> Handcrafted_PE[人工設計: Sin/Cos]
    PE --> Learnable_PE[可學習位置編碼]

    App --> Speech_Attn[語音辨識]
    App --> Image_Attn[影像處理]
    Speech_Attn --> Truncated_Attn[截斷式自注意力]
    Truncated_Attn -- 降低 O(L^2) --> App
    Image_Attn --> Pixel_As_Vector[像素視為向量集合]
    Image_Attn --> Vision_Transformer[應用於 Vision Transformer]

    Comp --> VS_CNN[與 CNN 比較]
    Comp --> VS_RNN[與 RNN 比較]
    VS_CNN --> Local_vs_Global[局部 vs. 全局感受野]
    VS_CNN --> Subset_Relationship[CNN 是 Self-Attention 的特例]
    VS_CNN --> Data_Amount_Effect[數據量影響表現]
    VS_RNN --> Parallelization[平行處理能力]
    VS_RNN --> Long_Term_Dep[長距離依賴處理]
    VS_RNN --> RNN_Replacement[逐漸取代 RNN]

    Graph --> Edge_Info[利用圖的邊資訊]
    Graph --> Limited_Attn[僅計算相鄰節點注意力]
    Graph --> GNN_Type[一種類型的 Graph Neural Network]

    Future --> Compute_Complexity[O(L^2) 計算量大]
    Future --> Efficient_Transformers[高效 Transformer 變形]
    Efficient_Transformers --> XXformer[如 Linformer, Performer]
    Efficient_Transformers --> Speed_Performance_Tradeoff[速度與效能權衡]
```

## 隨堂測驗

### 測驗一

在自注意力機制 (Self-Attention) 中，當我們要計算一組輸出向量 $b^1, b^2, b^3, b^4$ 時，它們的計算順序為何？

A. 必須先計算 $b^1$，然後 $b^2$，依序進行。
B. $b^1$ 到 $b^4$ 可以一次同時計算出來。
C. 順序無關緊要，但通常會先計算 $b^1$。
D. 順序由模型自行決定，可能依序也可能同時。

<details>
  <summary>點擊查看解答</summary>
  **正確答案：B**
  解釋：Self-Attention 的一個重要特性是，其所有輸出向量 $b^1$ 到 $b^L$ 可以透過平行運算一次性生成，這也是它相較於 RNN 的一個主要優勢。
</details>

### 測驗二

為何 Self-Attention 模型需要「位置編碼 (Positional Encoding)」？它是如何將位置資訊傳達給模型的？

A. 為了降低模型的計算複雜度，透過位置編碼來優化注意力分數的計算。
B. 因為 Self-Attention 機制本身不包含任何位置信息，所有輸入向量對它來說是沒有順序關係的。它透過將位置向量加到輸入向量上來提供位置信息。
C. 位置編碼有助於減少模型訓練時的過擬合問題，它透過對 Q, K, V 矩陣進行額外的編碼來實現。
D. 為了在多頭自注意力機制中區分不同的頭，每個頭被賦予一個獨特的位置編碼。

<details>
  <summary>點擊查看解答</summary>
  **正確答案：B**
  解釋：Self-Attention 模型的設計使得它對輸入序列中向量的相對或絕對位置是無感的。為了讓模型感知位置信息，通常會生成一個獨特的位置向量 (Positional Vector)，並將其疊加到對應的輸入向量上，從而將位置信息嵌入到輸入中。
</details>

### 測驗三

比較 Self-Attention (SA) 和卷積神經網絡 (CNN) 在處理影像時的特性，尤其是在訓練資料量較少的情況下，哪一個模型的表現可能更好？簡述其原因。

<details>
  <summary>點擊查看解答</summary>
  **答案：**
  在訓練資料量較少的情況下，**CNN** 的表現通常會比 Self-Attention 更好。

  **原因：**
  1.  **彈性與歸納偏置 (Inductive Bias)**：
      *   **Self-Attention** 具有非常高的彈性，它可以「學習」任意兩個像素之間的關係，就像其感受野是自適應學習的，因此需要大量的數據來充分學習這些複雜的模式。
      *   **CNN** 具有較強的「局部性」歸納偏置，它預設了影像中的相關資訊通常存在於局部區域（即感受野）。這種對局部性的內建偏置限制了模型的彈性，使其在數據量有限時，由於搜尋空間較小，更不容易過擬合。
  2.  **數據需求**：
      *   由於 Self-Attention 更為靈活，它需要更多的訓練資料才能收斂並達到最佳性能。
      *   CNN 的限制性使其在數據量較少時，依然能利用其局部性優勢，表現出較好的泛化能力。
</details>