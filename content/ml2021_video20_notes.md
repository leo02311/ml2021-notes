---
title: "第20堂課：Transformer (Part 2 of 2)"
tags:
  - MachineLearning
  - ML2021
  - DeepLearning
  - Optimization
---

# 第20堂課：Transformer (Part 2 of 2)

本課程由李宏毅教授講解深度學習中關鍵的技術——**批次正規化 (Batch Normalization, BN)**。透過調整特徵分佈，BN 能夠有效解決訓練過程中的損失函數景觀（Loss Landscape）問題，從而加速神經網路的收斂。

## 1. 為什麼需要正規化？

在神經網路訓練中，如果輸入特徵（$x_1, x_2$）的分佈範圍差異過大（例如 $x_1$ 很小而 $x_2$ 很大），會導致損失函數的誤差表面（Error Surface）在不同方向上出現極端的陡峭或平緩。

*   **平滑的景觀**：當特徵分佈均勻時，損失函數的誤差表面較為接近圓形，梯度下降（Gradient Descent）的路徑能更直觀地指向最小值。
*   **陡峭的景觀**：當特徵分佈不均時，誤差表面會被拉長，導致某些方向變化極快，另一些方向變化極慢，造成訓練過程中的震盪與收斂困難。

透過 **Feature Normalization**，我們可以將每個維度的資料進行標準化：
$$\tilde{x}_i^r \leftarrow \frac{x_i^r - m_i}{\sigma_i}$$
其中 $m_i$ 為平均值，$\sigma_i$ 為標準差，確保所有維度的平均值為 $0$，變異數為 $1$。

## 2. 深度學習中的 Batch Normalization

在深層神經網路中，不僅是輸入層，每一層的輸出（激活函數之前）也可能因為參數更新而發生分佈偏移。

### 演算法流程
對於一個 Batch 的資料 $\{z^1, z^2, z^3\}$，我們執行以下步驟：
1. **計算平均值**：$\mu = \frac{1}{3} \sum_{i=1}^3 z^i$
2. **計算標準差**：$\sigma = \sqrt{\frac{1}{3} \sum_{i=1}^3 (z^i - \mu)^2}$
3. **標準化**：$\tilde{z}^i = \frac{z^i - \mu}{\sigma}$
4. **縮放與平移（Scale & Shift）**：為了保持神經網路的表達能力，BN 加入了可學習參數 $\gamma$ 與 $\beta$：
   $$\hat{z}^i = \gamma \odot \tilde{z}^i + \beta$$

### 測試階段的處理
由於測試時可能只有單筆資料而非一個完整的 Batch，我們無法即時計算該 Batch 的 $\mu$ 和 $\sigma$。因此，我們在訓練過程中記錄這些統計量的**移動平均（Moving Average）**，在測試時直接使用累積的平均值 $\bar{\mu}$ 和標準差 $\bar{\sigma}$ 來進行標準化。

## 3. 知識圖譜

```mermaid
graph TD
    "Batch Normalization" --> "核心目標"
    "Batch Normalization" --> "訓練流程"
    "Batch Normalization" --> "測試策略"
    "核心目標" --> "解決誤差表面陡峭問題"
    "核心目標" --> "加速模型收斂"
    "訓練流程" --> "計算平均值與變異數"
    "訓練流程" --> "標準化 z"
    "訓練流程" --> "Scale and Shift"
    "測試策略" --> "使用移動平均"
    "測試策略" --> "不依賴當前 Batch"
```

## 4. 關於 Internal Covariate Shift 的反思

原始論文提出 BN 是為了減少「內部協變量偏移」（Internal Covariate Shift）。然而，後續研究指出，BN 之所以有效，更多是因為它平滑了損失函數的景觀。正如科學史上青黴素的發現，BN 在訓練上的顯著效果可能帶有某種「偶然性」（Serendipitous），後續對於歸一化方案的設計仍有廣闊的探索空間。

---

## 隨堂測驗

**Q1: 在 Batch Normalization 中，使用參數 $\gamma$ 與 $\beta$ 的目的是什麼？**
<details>
<summary>點擊查看解答</summary>
為了恢復神經網路在執行正規化後可能損失的表達能力。若沒有縮放與平移，標準化後的數據均為零均值單位變異，可能限制了後續激活函數的運作空間。
</details>

**Q2: 為什麼在測試階段不能直接使用當前輸入資料計算 $\mu$ 與 $\sigma$？**
<details>
<summary>點擊查看解答</summary>
因為 Batch Normalization 依賴於一個「Batch」的統計量。在測試時，輸入的可能僅有單筆資料（Batch size = 1），此時計算出來的統計量無法代表訓練集的分佈，因此必須使用訓練過程中累積的移動平均值。
</details>

**Q3: 根據課程內容，除了 Batch Normalization，還有哪些其他的正規化方法？**
<details>
<summary>點擊查看解答</summary>
課程中提到的方法包含：Batch Renormalization、Layer Normalization、Instance Normalization、Group Normalization、Weight Normalization 與 Spectrum Normalization。
</details>