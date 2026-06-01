---
title: "第10堂課：卷積神經網路 (Convolutional Neural Networks, CNN)"
tags:
  - MachineLearning
  - ML2021
  - CNN
  - ConvolutionalNeuralNetworks
  - DeepLearning
  - ImageProcessing
---

# 第10堂課：卷積神經網路 (Convolutional Neural Networks, CNN)

## 1. 引言：Network 架構與 CNN 簡介

本堂課將深入探討神經網路（Network）的架構設計，特別是 **Convolutional Neural Network (CNN)**。CNN 主要應用於影像處理，我們將透過這個例子來理解其設計理念以及如何提升模型表現。

### 什麼是 CNN？
- **縮寫：** CNN
- **全稱：** Convolutional Neural Network
- **主要用途：** 專門設計用於影像相關任務，但應用範圍已擴展至其他領域。

### CNN 的應用領域
- **影像分類：** 給定圖片，判斷圖片內容。
- **AlphaGo：** 應用於圍棋AI。
- **語音處理、文字處理：** 近年來也有應用，但需針對特性調整設計。

## 2. 影像分類問題與傳統 Fully Connected Network 的挑戰

### 影像分類目標
- **任務：** 給予機器一張圖片，判斷其內容。
- **輸入：** 固定大小的圖片，例如 $100 \times 100$ 像素解析度。即使圖片大小不一，通常也會先 Rescale 到固定大小。
- **輸出：** 每個類別用 **One-Hot Vector** 表示 ($\hat{y}$)。
    - 向量長度決定可辨識的物體種類數量（例如：$2000$ 種物體，向量長度即為 $2000$）。
    - 輸出經 **Softmax** 函數處理後得到 $y'$。
- **目標：** 最小化 $y'$ 與 $\hat{y}$ 之間的 **Cross Entropy**。

### 影像的電腦表示：三維 Tensor
- **概念：** 對於機器而言，一張圖片是一個三維的 Tensor。
    - **維度一：** 圖片寬度 (Width)
    - **維度二：** 圖片高度 (Height)
    - **維度三：** 圖片的 **Channel** 數目
        - 彩色圖片：R (紅), G (綠), B (藍) 三個 Channel。
        - 黑白圖片：1 個 Channel。
- **數值：** 每個 Channel 的每個 Pixel 都有一個數值，代表該顏色的強度。
- **例子：** $100 \times 100$ 解析度的彩色圖片，包含 $100 \text{ (寬)} \times 100 \text{ (高)} \times 3 \text{ (Channel)}$ 個數值。

### Fully Connected Network 的問題
若將三維影像 Tensor 直接「拉直」（Flatten）成一個巨大的向量作為 Fully Connected Network 的輸入，將面臨以下問題：

1.  **參數爆炸 (Huge Number of Parameters)：**
    - 圖片輸入向量長度：$100 \times 100 \times 3 = 30{,}000$ 維。
    - 假設第一層有 $1000$ 個 Neuron。
    - 第一層的 Weight 數量：$1000 \text{ (Neuron)} \times 30{,}000 \text{ (輸入維度)} = 3 \times 10^7$ (3千萬) 個參數。
    - 這麼多參數會導致模型訓練困難且需要龐大的計算資源。

2.  **Overfitting 風險增加：**
    - 雖然多參數能增加模型彈性與能力。
    - 但參數越多，模型的彈性越大，越容易在訓練資料上表現良好，但在未見過的資料上表現不佳，即發生 Overfitting。

## 3. CNN 的核心概念 (第一種觀點：基於觀察的簡化)

為了解決 Fully Connected Network 的問題，CNN 引入了基於影像特性設計的簡化。

### 觀察一：重要 Pattern 具局部性 (Local Connectivity)
- **核心思想：** 辨識影像中的物體，通常是透過偵測其局部出現的關鍵 Pattern (例如：鳥嘴、眼睛、鳥爪)。
- **Neuron 僅需關注影像局部：**
    - 偵測一個「鳥嘴」Pattern，並不需要檢視整張圖片。
    - Neuron 只需要看圖片的 **一小部分** 即可完成其偵測任務。

#### Receptive Field (感受野) 的概念
- **定義：** 每個 Neuron 只關心其定義的特定局部區域，這個區域稱為 Receptive Field (RF)。
- **運作方式：**
    1.   Neuron 定義一個 RF (例如 $3 \times 3 \times 3$ 區域)。
    2.  將 RF 內的 $3 \times 3 \times 3 = 27$ 個數值拉直成向量，作為 Neuron 的輸入。
    3.  Neuron 為這 $27$ 維向量的每個維度分配一個 Weight，加上 Bias，再通過 Activation Function 產生輸出。
- **RF 的靈活性與常見設定：**
    - **重疊：** Receptive Field 之間可以彼此重疊。
    - **多個 Neuron 守備：** 同一個 RF 範圍可以由多個不同的 Neuron 共同守備，以偵測不同的 Pattern。
    - **大小：** RF 可以有不同大小 (例如 $3 \times 3$, $11 \times 11$)，但一般不會設太大 (常見 $3 \times 3$)。
    - **Channel 考量：** RF 通常會考慮所有 Channel (R, G, B)。
    - **形狀：** 通常為相連的正方形或長方形區域。
    - **Kernel Size：** Receptive Field 的高與寬合稱為 **Kernel Size**。

#### RF 掃描機制：Stride 與 Padding

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/vid10_cnn_dark.png">
  <img alt="CNN Convolution Window Sliding" src="../assets/vid10_cnn_light.png">
</picture>

- **Stride (步長)：** 決定 Receptive Field 移動的距離。例如 $\text{Stride} = 2$ 表示每次移動兩格。
    - 為了確保 RF 之間有足夠重疊，以避免 Pattern 剛好落在 RF 之間而被遺漏，$\text{Stride}$ 通常設為 $1$ 或 $2$。
- **Padding (補值)：** 當 RF 超出影像邊界時，為了不遺漏邊緣的 Pattern，會用特定數值（最常見是 0，稱為 Zero Padding）來補齊超出部分的像素值。

### 觀察二：相同 Pattern 具空間不變性 (Parameter Sharing / Weight Sharing)
- **核心思想：** 同一個 Pattern (例如鳥嘴) 可能出現在圖片的不同位置。
- **問題：** 如果每個位置都需要一個獨立的 Neuron 去偵測「鳥嘴」，則會產生大量重複工作的 Neuron，導致參數過多。
- **解決方案：參數共享 (Parameter Sharing)：**
    - 讓不同 Receptive Field 的 Neuron 共享相同的參數 (Weights)。
    - 雖然參數相同，但由於它們的輸入 (來自不同 RF) 不同，所以輸出也會不同。
    - 這種共享參數的 Neuron 集合，其實就是第二種觀點中提到的 **Filter**。

#### 共享參數的具體實現：Filter
- **Filter：** 一組共享的參數。每個 Receptive Field 都會有一組 Neuron 負責守備 (例如 64 個 Neuron)，這些 Neuron 在不同的 RF 位置上，會共享同一組參數，這組參數即為一個 Filter。
- **Filter 的命名：** Filter1, Filter2, Filter3...

### 總結：Convolutional Layer 的誕生
- **Convolutional Layer = Receptive Field (局部連接) + Parameter Sharing (權重共享)**
- **CNN 的模型偏差 (Model Bias)：**
    - 相較於 Fully Connected Network，Convolutional Layer 的彈性較小 (限制 Neuron 只能看局部，並強制共享參數)。
    - 這意味著 CNN 的模型偏差較大，但這對於影像任務而言是好的。
    - 因為 CNN 是專為影像特性設計的，儘管有較大的偏差，但在影像任務上仍能表現優異，且不容易 Overfitting。
    - **警示：** 若將 CNN 用於非影像任務，需仔細思考該任務是否也具備影像的這些特性。

## 4. CNN 的核心概念 (第二種觀點：Filter 的掃描與 Feature Map)

這是更常見的 CNN 介紹方式，本質上與第一種觀點相同，只是描述方式不同。

### Filter 的定義與作用
- **Filter 結構：**
    - 每個 Filter 都是一個小型的三維 Tensor，其大小為 `3 x 3 x Channel_Size`。
    - `Channel_Size` 取決於輸入影像的 Channel 數目（彩色為 3，黑白為 1）。
- **Filter 偵測 Pattern：** 每個 Filter 的作用就是在影像中捕捉特定的局部 Pattern。Filter 內的數值就是模型需要學習的參數。

### Convolution 操作實例
- **假設：** $6 \times 6$ 黑白圖片 ($\text{Channel} = 1$)，Filter 大小 $3 \times 3$。
- **步驟：**
    1.  將 Filter 放在圖片的左上角。
    2.  將 Filter 內的所有數值與圖片該區域的數值進行 **內積 (Inner Product)** 運算，得到一個單一數值。
    3.  Filter 根據 **$\text{Stride}$** 往右移動（例如 $\text{Stride} = 1$），重複步驟 2。
    4.  掃完一行後，Filter 往下移動（同樣根據 $\text{Stride}$），重複掃描整行。
    5.  直到 Filter 掃遍整張圖片為止。
- **結果：** 一個 Filter 掃描圖片後，會產生一個數值矩陣，其中每個數值代表 Filter 在對應區域偵測到的 Pattern 强度。

### Feature Map 的生成
- **概念：** 每個 Filter 掃描完圖片後，都會產生一個數值矩陣，這個矩陣稱為 **Feature Map (特徵圖)**。
- **多個 Filter：** 如果一個 Convolutional Layer 包含多個 Filter (例如 64 個 Filter)，則會產生 64 個 Feature Map。
- **新的「圖片」：** 這些 Feature Map 可以被視為一張新的「圖片」，只是它的 Channel 數目不再是 RGB，而是與 Filter 的數量相同（例如 64 個 Channel）。

### 多層 Convolution 的效果：擴大感受野
- **問題：** 如果 Filter 大小一直設為 $3 \times 3$，如何偵測更大的 Pattern？
- **解答：** 透過堆疊多層 Convolutional Layer。
    - 例如，第一層使用 $3 \times 3$ Filter，其輸出 Feature Map。
    - 第二層再使用 $3 \times 3$ Filter 處理第一層的 Feature Map。
    - 雖然每一層都只看 $3 \times 3$ 的範圍，但在原始影像上，第二層的 $3 \times 3$ Filter 實際上已經「看到」了原始影像的 $5 \times 5$ 範圍。
    - **結論：** Network 疊得越深，即使 Filter 尺寸不變，其在原始影像上的有效 **感受野** 也會越來越大，從而能偵測到更大的 Pattern。

## 5. 比較兩種 CNN 概念

| 特性             | 第一種觀點 (Neuron 簡化)             | 第二種觀點 (Filter 掃描)          |
| :--------------- | :----------------------------------- | :-------------------------------- |
| **核心概念**     | Receptive Field + Parameter Sharing  | Filter 進行 Convolution           |
| **參數**         | Neuron 共享的 Weight                 | Filter 本身就是一組參數           |
| **操作**         | Neuron 關注其 Receptive Field 範圍內 | Filter 掃過圖片，計算內積         |
| **結果**         | Neuron 輸出 (對應特定 Pattern 存在與否) | Feature Map (所有 Neuron 輸出的集合) |
| **掃描機制**     | 不同 Neuron 守備不同 Receptive Field | 同一個 Filter 掃過整張圖片      |

- **Bias：** 在實作上，Filter 也通常會包含 Bias 數值。
- **命名由來：** 「Convolutional Layer」之名來自於 Filter 掃過圖片的過程，這個數學操作即是 Convolution。

## 6. Pooling Layer：縮小影像維度

### Pooling 的觀察基礎
- **核心思想：** 對影像進行 Subsampling (例如將圖片縮小到 $1/4$)，通常不會影響我們對影像中物體的辨識。例如，一隻鳥的圖片縮小後，仍然是一隻鳥。
- **目的：** 透過縮小影像尺寸來減少運算量，同時保持關鍵特徵。

### Pooling 的運作方式 (Max Pooling 為例)
- **特性：**
    - **無參數：** Pooling Layer 本身不包含任何可學習的參數 (Weight)。因此，它更像是一個操作 (Operator) 或激活函數 (Activation Function)，而非傳統的 Layer。
    - **固定行為：** 它的運言規則是固定好的，不需根據資料學習。
- **Max Pooling：**
    1.  將 Feature Map 中的數值，按固定大小分組 (例如 $2 \times 2$)。
    2.  在每一組中，選擇 **最大** 的數值作為該組的代表。
- **其他 Pooling 方式：**
    - **Min Pooling：** 選擇最小值。
    - **Average Pooling：** 選擇平均值。
    - **尺寸：** Pooling 的分組大小 (例如 $2 \times 2$, $3 \times 3$, $4 \times 4$) 可自定義。

### Pooling 的作用與應用模式
- **作用：**
    - 將 Feature Map 的空間尺寸縮小 (例如 $4 \times 4$ 變成 $2 \times 2$)。
    - Feature Map 的 Channel 數目保持不變。
    - 減少模型的參數和運算量。
- **應用模式：** 通常是 Convolutional Layer 與 Pooling Layer 交替使用 (例如：兩次 Convolution 後接一次 Pooling)。

### Pooling 的趨勢：Full Convolutional Networks
- **缺點：** Pooling 會丟失一些精細的空間信息，對於需要處理微細物件的任務，可能會損害模型表現。
- **現代趨勢：** 隨著運算能力的提升，許多先進的影像辨識網路開始捨棄 Pooling Layer，轉而使用 **Full Convolutional Neural Network**。
- **原因：** Pooling 最主要目的是減少運算量，若運算資源充足，不使用 Pooling 有助於保留更多細節，可能獲得更好的效能。

## 7. CNN 網路的整體架構

一個經典的影像辨識 CNN 網路通常包含以下步驟：

1.  **多層 Convolution 與 Pooling 的交替：** 處理輸入圖片，提取不同層次的特徵。
2.  **Flatten (拉平)：** 將最後一個 Pooling Layer 的輸出 (一個多 Channel 的圖片 Tensor) 拉直成一個巨大的向量。
3.  **Fully Connected Layer (全連接層)：** 將 Flatten 後的向量輸入一或多個 Fully Connected Layer，進行高層次的特徵組合與分類。
4.  **Softmax Layer：** 最終輸出為各類別的機率分佈，作為影像辨識的結果。

```mermaid
graph TD
    Input_Image['影像輸入 (3D Tensor)'] --> Conv1['Convolutional Layer 1']
    Conv1 --> Feature_Map_1['Feature Map 1']
    Feature_Map_1 -- "輸出" --> Pool1['Pooling Layer 1']
    Pool1 --> Conv2['Convolutional Layer 2']
    Conv2 --> Feature_Map_2['Feature Map 2']
    Feature_Map_2 -- "輸出" --> Pool2['Pooling Layer 2']
    Pool2 --> Flatten['Flatten Layer']
    Flatten --> FC1['Fully Connected Layer 1']
    FC1 --> FC_Output['最終分類層 (Softmax)']
    FC_Output --> Output['分類結果']

    subgraph "核心觀察"
        Obs_Local['局部 Pattern 觀察']
        Obs_Spatial['Pattern 空間不變性觀察']
        Obs_Subsample['影像 Subsampling 觀察']
    end

    subgraph "CNN 組件原理"
        RF['Receptive Field']
        PS['Parameter Sharing']
        Filter['Filter / Kernel']
        Feature_Map['Feature Map']
    end

    Obs_Local --> RF
    Obs_Spatial --> PS
    PS --> Filter
    RF --> Conv_Principle['Convolutional Layer 原理']
    Filter --> Conv_Principle

    Conv_Principle --> Conv1
    Conv_Principle --> Conv2

    Obs_Subsample --> Pool_Principle['Pooling Layer 原理']
    Pool_Principle --> Pool1
    Pool_Principle --> Pool2
```

## 8. CNN 的應用範例：AlphaGo

### 圍棋盤面的表示
- **輸入：** 棋盤上的黑子、白子位置。
- **輸出：** 下一步最佳落子位置（一個 $19 \times 19$ 個類別的分類問題）。
- **盤面表示成圖片：**
    - 棋盤可以視為 $19 \times 19$ 解析度的圖片。
    - 每個棋盤位置 (Pixel) 以 **$48$ 個 Channel** 來描述，包含該位置的各種棋局特性 (例如是否被叫吃、周圍顏色等)。
    - 因此，AlphaGo 的輸入是一個 $19 \times 19 \times 48$ 的三維 Tensor。

### CNN 適用於圍棋的特性
圍棋與影像有許多共通的特性，使得 CNN 適用於此任務：

1.  **局部 Pattern：** 圍棋中許多重要 Pattern (例如「叫吃」) 只需要看棋盤的局部區域即可判斷。
    - AlphaGo 的第一層 Filter 大小為 $5 \times 5$，顯示設計者認為許多關鍵 Pattern 在 $5 \times 5$ 範圍內可被偵測。
2.  **Pattern 空間不變性：** 相同的 Pattern (例如「叫吃」) 可以出現在棋盤的任何位置。
    - 這與影像中「鳥嘴」可在不同位置出現的特性相似。

### AlphaGo 的網路架構揭密：無 Pooling 層的設計
- **論文發現：** 仔細閱讀 AlphaGo 的原始論文附件會發現，其網路架構**並沒有使用 Pooling Layer**。
- **AlphaGo 的結構：**
    - 輸入：$19 \times 19 \times 48$ 的 Image。
    - 第一層：Zero Padding, Kernel Size $5 \times 5$, $192$ 個 Filter, $\text{Stride} = 1$, 激活函數 ReLU。
    - 第二層至第十二層：Zero Padding, Kernel Size $3 \times 3$, $192$ 個 Filter, $\text{Stride} = 1$, 激活函數 ReLU。
    - 最後層：Softmax (用於分類)。
- **重要啟示：**
    - 類神經網路的設計**存乎一心**。
    - 不要盲目套用其他領域的經驗（例如影像辨識常用 Pooling）。
    - 必須深入思考任務本身的特性，設計最合適的網路架構。圍棋作為一個精細的任務，捨棄 Pooling 有助於保留棋盤上每一點的細微資訊，反而更適合。

## 9. CNN 的限制與解決方案

### 無法處理影像縮放與旋轉
- **問題：** CNN 本身並不具備處理影像縮放 (Scaling) 或旋轉 (Rotation) 變化的能力。
    - 例如，如果訓練時只看過特定大小的狗圖片，當狗被放大或旋轉後，CNN 可能就無法正確辨識。
    - 從數學角度看，即使形狀相同，但放大或旋轉後，拉直成向量的數值會完全不同，CNN 會將其視為完全不同的輸入。
- **Translation Invariance (平移不變性) vs. Scaling/Rotation Invariance：** CNN 由於 Filter 掃描的機制，對平移變動有一定程度的魯棒性（Pattern 平移，Filter 也能在新的位置找到），但對縮放和旋轉則不然。

### Data Augmentation (數據增強) 的必要性
- **目的：** 為了讓 CNN 能夠處理縮放和旋轉問題，必須在訓練時進行 Data Augmentation。
- **方法：**
    - 將訓練圖片進行隨機裁剪、放大、縮小。
    - 將訓練圖片進行隨機旋轉。
    - 讓 CNN 在訓練階段就見識到同一個物件在不同大小、不同角度下的呈現方式。

### 更多進階架構
- 存在一些架構專門處理 Scaling 和 Rotation 問題，例如 **Spatial Transformer Layer**。

## 10. 其他應用領域

### 語音與文字處理
- CNN 近年來也廣泛應用於語音識別和自然語言處理任務。
- **重要考量：** 若要將 CNN 用於語音或文字處理，必須**重新設計**其 Receptive Field 和參數共享機制，使其符合語音或文字資料的特性。
- **警示：** 不應直接將影像領域的 CNN 架構套用到語音或文字上，因為它們的資料結構和 Pattern 特性不同。

---

## 隨堂測驗

### 測驗一
為什麼傳統的 Fully Connected Network (全連接層) 不適合直接處理高解析度影像，會面臨哪些主要問題？

<details>
<summary>點擊查看解答</summary>
傳統的 Fully Connected Network 直接處理高解析度影像會面臨兩大問題：

1.  **參數爆炸 (Huge Number of Parameters)：** 影像（如 $100 \times 100 \times 3$ 的圖片）被拉直成一個巨大的向量作為輸入，如果第一層有許多 Neuron，將導致模型參數數量非常龐大。例如，$100 \times 100 \times 3$ 維的輸入向量和 $1000$ 個 Neuron 的第一層，會產生 $3 \times 10^7$ 個連接權重，這使得模型難以訓練且計算資源需求高。
2.  **Overfitting (過度擬合) 風險增加：** 參數數量過多雖然增加了模型的彈性，但也使其更容易在訓練資料上學習到過於細節或噪音的模式，導致在未見過的新資料上表現不佳。
</details>

### 測驗二
請說明構成 Convolutional Layer 的兩個核心觀察（或稱簡化），並解釋它們如何解決傳統網路的問題。

<details>
<summary>點擊查看解答</summary>
構成 Convolutional Layer 的兩個核心觀察是：

1.  **重要 Pattern 具局部性 (Local Connectivity)：** 許多影像中的關鍵特徵（如鳥嘴、眼睛）只需要觀察圖片的一小部分區域即可被偵測。
    *   **解決方案：Receptive Field (感受野)**。每個 Neuron 只連接到輸入影像的一個局部區域（稱為 Receptive Field），而不是整張圖片。這大幅減少了每個 Neuron 的連接數量，從而減少了模型的總參數。
2.  **相同 Pattern 具空間不變性 (Parameter Sharing / Weight Sharing)：** 同樣的 Pattern（例如「圓圈」）可能出現在圖片的不同位置。
    *   **解決方案：參數共享 (Parameter Sharing)**。在不同的 Receptive Field 上，負責偵測相同 Pattern 的 Neuron 共享同一組參數（即 Filter）。這意味著我們不需要為每個位置的每個 Pattern 訓練獨立的參數，進一步大幅減少了模型的總參數。

這兩個簡化有效地減少了模型參數，降低了 Overfitting 的風險，同時利用了影像的固有特性，使模型在影像任務上更有效率。
</details>

### 測驗三
李宏毅教授在課程中特別提到 AlphaGo 的 CNN 結構中缺少了哪一個常見的組件？這個發現對於我們理解神經網路設計有何重要啟示？

<details>
<summary>點擊查看解答</summary>
李宏毅教授提到，根據 AlphaGo 的原始論文附件，其 CNN 結構中**沒有使用 Pooling Layer**。

這個發現的重要性在於：

*   **設計思維的重要性：** 它強調了在設計神經網路架構時，不能盲目地套用其他領域（如影像辨識）的常用組件。每個問題都有其獨特的特性。
*   **理解組件目的：** Pooling Layer 的主要目的是為了減少運算量並提供一些平移不變性。然而，對於圍棋這種需要精確判斷棋盤上每一個細微變化的任務，捨棄 Pooling 有助於保留棋盤上所有位置的詳細資訊，這對最終的棋局判斷至關重要。
*   **Context-Dependent 設計：** 成功的網路架構是針對特定任務和數據特性進行深思熟慮後設計的。這提醒我們，設計類神經網路是一個結合理論與實踐的藝術，需要深入理解問題本身。
</details>