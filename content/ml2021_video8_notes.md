作為專業的技術筆記整理專家，我已將李宏毅教授的機器學習課程逐字稿整理成排版精美的 Markdown 筆記。

---
title: 第8堂課：【機器學習2021】類神經網路訓練不起來怎麼辦 (四)：損失函數 (Loss) 也可能有影響
tags:
  - MachineLearning
  - ML2021
  - Classification
  - Softmax
  - CrossEntropy
---

# 【機器學習 2021】第8堂課：【機器學習2021】類神經網路訓練不起來怎麼辦 (四)：損失函數 (Loss) 也可能有影響

## 課程簡介：分類問題 (Classification)

本堂課將以簡短的方式快速介紹分類問題的處理方法。若想深入了解，可參考過去的完整課程錄影。

### 從迴歸看分類問題 (Classification as Regression)

*   **迴歸定義**：輸入向量，輸出一個數值 `y`，目標是讓 `y` 接近真實標籤 `ŷ`。
*   **分類的類比**：
    *   可以嘗試將分類問題視為迴歸：輸入一個東西，輸出一個數值 `y`。
    *   將類別數值化：例如 Class1 編號為 1，Class2 編號為 2，Class3 編號為 3。
    *   目標：讓 `y` 接近類別的編號。
*   **問題點**：
    *   這種數值化方式隱含了類別之間的相似性。例如，Class1=1, Class2=2, Class3=3，會預設 Class1 與 Class2 較接近，Class1 與 Class3 較遠。
    *   若類別間無實際數值關係（如水果種類），此方法不適用。

### 類別表示法：One-hot Vector

為了避免預設類別關係的問題，常見的做法是使用 One-hot Vector 來表示類別。

*   **原理**：每個類別都用一個向量表示，其中只有一個元素為 1，其餘為 0。
*   **範例**：若有三個類別
    *   Class1: `[1, 0, 0]`
    *   Class2: `[0, 1, 0]`
    *   Class3: `[0, 0, 1]`
*   **優勢**：
    *   消除了類別間相似性的預設。
    *   任何兩個 One-hot Vector 之間的距離都是相同的，不會暗示某些類別更相似。

### 神經網路輸出層 (Network Output for Classification)

當目標 `ŷ` 是 One-hot Vector (一個多維向量) 時，神經網路的輸出也應該是一個多維向量。

*   **設計**：
    *   如果 `ŷ` 是一個 K 維向量 (K 個類別)，則網路的輸出 `y` 也應為 K 個數值 (`y₁, y₂, ..., yK`)。
    *   實現方式：將原本輸出單一數值的網路結構重複 K 次，每次使用不同的權重與偏置，即可產生 K 個輸出值。

### Soft-max 函式

在分類問題中，神經網路的原始輸出 `y` 通常會再通過一個 Soft-max 函式，得到 `y'`，然後再計算 `y'` 與 `ŷ` 之間的距離。

#### 為何需要 Soft-max？

*   **簡化解釋**：`ŷ` (One-hot Vector) 的值只有 0 和 1。為了讓網路輸出 `y` 能與 `ŷ` 更好地比較相似度，我們需要將 `y` 的值也歸一化到 0 到 1 之間。Soft-max 函式就扮演了這個角色。
*   **深層解釋 (課程中簡略)**：Soft-max 具有豐富的歷史淵源，通常從生成模型、Logistic Regression 等角度深入解釋其背後的假設。若要了解，可參考相關課程錄影。

#### Soft-max 運作原理

Soft-max 函式將輸入的 `y₁`, `y₂`, `y₃` (Logits) 轉換為 `y₁'`, `y₂'`, `y₃'`。

1.  **取指數 (Exponential)**：對每個輸入 `yᵢ` 取指數運算：`exp(yᵢ)`。
    *   無論 `yᵢ` 是正數還是負數，`exp(yᵢ)` 都會是正數。
2.  **歸一化 (Normalization)**：將每個 `exp(yᵢ)` 除以所有 `exp(yᵢ)` 的總和。
    *   `yᵢ' = exp(yᵢ) / Σⱼ exp(yⱼ)`

**圖示化說明：**
`y₁`, `y₂`, `y₃` -> `exp(y₁)`, `exp(y₂)`, `exp(y₃)` -> `Summation = exp(y₁) + exp(y₂) + exp(y₃)` ->
`y₁' = exp(y₁) / Summation`
`y₂' = exp(y₂) / Summation`
`y₃' = exp(y₃) / Summation`

#### Soft-max 的特性

*   **輸出範圍**：所有 `yᵢ'` 都介於 0 到 1 之間。
*   **總和為 1**：所有 `yᵢ'` 的總和為 1 (類似機率分佈)。
*   **拉大數值差距**：Soft-max 會讓輸入中較大的值變得更大 (接近 1)，較小的值變得更小 (接近 0)，從而放大不同類別之間的區別。
    *   **範例**：輸入 `[3, 1, -3]` 經過 Soft-max 後變成 `[0.88, 0.12, 0.00]`，可以看到 `-3` 被壓到趨近於 0。

#### Logit

Soft-max 函式的輸入 `yᵢ` 通常被稱為 **Logit**。

#### Soft-max 與 Sigmoid 的關係 (for 2 Classes)

當只有兩個類別時，Soft-max 函式與 Sigmoid 函式是等價的。可以自行推導驗證。

### 分類常用的損失函數 (Loss Functions for Classification)

在得到 `y'` (Soft-max 輸出) 和 `ŷ` (One-hot label) 後，我們需要計算兩者之間的「距離」`e`，這個距離就是損失 (Loss)。

#### 平均平方誤差 (Mean Squared Error, MSE)

*   **公式**：`e = Σᵢ (yᵢ' - ŷᵢ)²`
*   **問題**：雖然 MSE 也可以用來衡量兩個向量的距離，但在分類問題中，特別是當模型的預測離正確答案很遠時，MSE 的梯度可能會非常小，導致模型訓練緩慢或停滯。

#### 交叉熵 (Cross-entropy, CE)

Cross-entropy 是分類問題中最常用的損失函數。

*   **公式**：`e = Σᵢ ŷᵢ * log(yᵢ')`
    *   **註**：通常 Cross-entropy 的標準公式為 `-Σᵢ ŷᵢ * log(yᵢ')`。教授的講法是當 `ŷ` 與 `y'` 相同時，該值越小越好，這與帶負號的公式在最小化目標上是一致的。
*   **與最大似然 (Maximize Likelihood) 的關係**：最小化 Cross-entropy 與最大化似然函數是等價的。
*   **優勢：最佳化角度考量**
    *   相較於 MSE，Cross-entropy 在模型預測與真實標籤差異較大時，仍能提供足夠大的梯度。
    *   這使得梯度下降演算法能夠更有效地調整參數，加速訓練過程，避免卡在梯度平坦區。
    *   **錯誤曲面範例**：當 Loss 很大（例如初始訓練階段）時，MSE 的錯誤曲面可能非常平坦，導致梯度趨近於 0，使訓練難以啟動。而 Cross-entropy 在相同情況下仍具有明顯斜率，有助於模型學習。

#### Soft-max 與 Cross-entropy 的整合

*   **PyTorch 設計**：在 PyTorch 中，`nn.CrossEntropyLoss` 函式內部已經包含了 Soft-max 運算。
*   **使用注意事項**：如果你在神經網路的輸出層已經手動加入了 Soft-max 函式，然後再使用 PyTorch 的 `nn.CrossEntropyLoss`，會導致 Soft-max 被套用兩次，這通常不是我們想要的行為。

---

## 知識圖譜 (Knowledge Graph)

```mermaid
graph TD
    subgraph 機器學習分類核心
        分類問題["分類問題 (Classification Problem)"]
        迴歸類比法["迴歸類比法"]
        數值化表示問題["數值化表示問題"]
        類別表示方式["類別表示方式"]
        OneHotVector["One-hot Vector"]
        多輸出神經網路["多輸出神經網路"]
        Softmax函數["Soft-max 函數"]
    end

    subgraph Softmax詳解
        Logits輸入[y₁ y₂ y₃ (Logits)] --> Softmax計算原理["Soft-max 計算原理"]
        Softmax計算原理 --> 取指數運算["取指數 (exp)"]
        取指數運算 --> 歸一化處理["歸一化 (Normalization)"]
        歸一化處理 --> 機率分佈輸出[y₁' y₂' y₃' (機率分佈)]
        Softmax特性["Soft-max 函式特性"]
        值介於零壹["值介於 0-1"]
        總和為壹["總和為 1"]
        拉大數值差距["拉大數值差距"]
        兩類等價Sigmoid["兩類等價於 Sigmoid"]
    end

    subgraph 分類損失函數
        損失函數選擇["損失函數選擇"]
        MSE損失["平均平方誤差 (MSE)"]
        交叉熵損失["交叉熵 (Cross-entropy)"]
        優化挑戰["優化挑戰"]
        梯度平坦區["MSE 梯度平坦區"]
        優化優勢["交叉熵優化優勢"]
    end

    subgraph 實作與理論
        最大似然關係["與最大似然 (Maximize Likelihood) 關係"]
        PyTorch整合設計["PyTorch 整合設計"]
        CrossEntropyLoss內建["CrossEntropyLoss 內建 Soft-max"]
        重複Softmax風險["重複 Soft-max 風險"]
    end

    分類問題 -- 需要 --> 類別表示方式
    分類問題 -- 可嘗試 --> 迴歸類比法
    迴歸類比法 -- 引發 --> 數值化表示問題
    數值化表示問題 -- 解決方案 --> OneHotVector
    OneHotVector -- 定義目標 --> 多輸出神經網路
    多輸出神經網路 -- 輸出 --> Logits輸入
    Logits輸入 -- 經處理 --> Softmax函數
    Softmax函數 -- 細節 --> Softmax詳解
    Softmax詳解 --> Softmax特性
    機率分佈輸出 -- 作為輸入 --> 損失函數選擇
    OneHotVector -- 作為目標 --> 損失函數選擇
    損失函數選擇 -- 包含 --> MSE損失
    損失函數選擇 -- 包含 --> 交叉熵損失
    MSE損失 -- 造成 --> 優化挑戰
    優化挑戰 -- 表現為 --> 梯度平坦區
    交叉熵損失 -- 擁有 --> 優化優勢
    交叉熵損失 -- 理論基礎 --> 最大似然關係
    交叉熵損失 -- 實作於 --> PyTorch整合設計
    PyTorch整合設計 -- 提供 --> CrossEntropyLoss內建
    CrossEntropyLoss內建 -- 使用不當導致 --> 重複Softmax風險

    classDef blue fill:#add8e6;
    class 機器學習分類核心 blue
    class Softmax詳解 blue
    class 分類損失函數 blue
    class 實作與理論 blue
```

---

## 隨堂測驗

### 測驗一

在機器學習分類問題中，使用 One-hot Vector 表示類別的主要目的是什麼？

<details>
<summary>點擊展開解答</summary>
C) 為了避免隱含地假設不同類別之間存在鄰近性或序數關係。One-hot Vector 使所有類別之間的距離相等，不會預設任何類別比其他類別「更像」或「更遠」。
</details>

### 測驗二

關於 Soft-max 函式，下列哪一個敘述是 **不正確** 的？

<details>
<summary>點擊展開解答</summary>
D) 對於一個兩類分類問題，使用 Soft-max 和使用 Sigmoid 函式是等價的，它們之間沒有根本上的不同。
</details>

### 測驗三

從最佳化 (Optimization) 的角度來看，為何在分類問題中，交叉熵 (Cross-entropy) 通常比平均平方誤差 (Mean Squared Error, MSE) 更受推薦？

<details>
<summary>點擊展開解答</summary>
C) 交叉熵在模型預測離真實標籤較遠時，仍能產生足夠大的梯度，這有助於梯度下降演算法更有效地調整模型參數，避免卡在梯度平坦區。
</details>