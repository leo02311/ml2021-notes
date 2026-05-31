---
title: "第17堂課：批次正規化 (Batch Normalization)"
tags:
  - MachineLearning
  - ML2021
  - DeepLearning
  - BatchNormalization
---

# 第17堂課：批次正規化 (Batch Normalization)

本課程由李宏毅教授講解深度學習中關鍵的技術：**Batch Normalization (批次標準化)**。

## 一、 為什麼需要正規化 (Normalization)？

在優化問題中，損失函數的平面 (Loss Landscape) 若過於崎嶇或陡峭，會導致梯度下降 (Gradient Descent) 難以收斂。

*   **輸入特徵的影響**：當輸入特徵 $x_1$ 與 $x_2$ 的數值範圍差異巨大時（例如 $x_1 \in [1, 2], x_2 \in [100, 200]$），會導致權重 $w_1, w_2$ 更新時的損失函數平面變得不均勻（如橢圓狀），這會讓模型難以優化。
*   **特徵正規化 (Feature Normalization)**：針對每個維度 $i$，透過減去平均值 $m_i$ 並除以標準差 $\sigma_i$，使該維度的數值分佈平均為 $0$、變異數為 $1$。這能讓損失函數平面更接近圓形，加速梯度下降的收斂速度。

公式如下：
$$\tilde{x}_i^r \leftarrow \frac{x_i^r - m_i}{\sigma_i}$$

## 二、 深度學習中的 Batch Normalization

在深度神經網絡中，即便輸入層做了正規化，隱藏層的輸出 (例如 $z^1, z^2, z^3$) 仍然可能因為經過權重計算後，分佈發生偏移或範圍變化，導致後續層難以訓練。

### 1. 運作機制
針對一個 Batch 中的樣本，計算該批次在某個層的輸出統計量：
*   平均值：$\mu = \frac{1}{3} \sum_{i=1}^{3} z^i$
*   標準差：$\sigma = \sqrt{\frac{1}{3} \sum_{i=1}^{3} (z^i - \mu)^2}$

接著對輸出進行標準化與縮放平移：
*   標準化：$\tilde{z}^i = \frac{z^i - \mu}{\sigma}$
*   調整分佈：$\hat{z}^i = \gamma \odot \tilde{z}^i + \beta$
    *   其中 $\gamma$ 與 $\beta$ 是可學習的參數，允許模型學習恢復原始的分佈特性。

### 2. 測試階段的處理
在測試時，我們可能只有單一樣本，無法計算 Batch 的 $\mu$ 與 $\sigma$。因此，我們在訓練過程中會記錄每一批次的 $\mu$ 與 $\sigma$，並計算其**移動平均 (Moving Average)**。測試時直接使用這些累積的統計量。

## 三、 關於 Internal Covariate Shift 的討論

過去認為 Batch Normalization 的成功是因為它解決了「Internal Covariate Shift」（內部協變量偏移）。但後續研究顯示，該說法存在爭議。現有實驗指出，Batch Normalization 真正有效的原因更傾向於它平滑了損失函數的表面 (Landscape)，使得訓練過程更穩定。這種正向影響甚至被形容為某種「偶然的」(serendipitous) 發現。

## 四、 知識圖譜

```mermaid
graph TD
    "Batch Normalization" --> "核心目的"
    "Batch Normalization" --> "運作原理"
    "Batch Normalization" --> "測試策略"
    "核心目的" --> "平滑損失函數平面"
    "核心目的" --> "加速收斂"
    "運作原理" --> "計算Batch統計量"
    "計算Batch統計量" --> "平均值mu"
    "計算Batch統計量" --> "標準差sigma"
    "運作原理" --> "參數縮放gamma與beta"
    "測試策略" --> "移動平均Moving Average"
```

---

## 隨堂測驗

**Q1: 為何要進行特徵正規化 (Feature Normalization)？**
<details>
<summary>點擊查看解答</summary>
答：為了讓不同維度的特徵處於相似的數值範圍，這能使損失函數的表面更平滑（更接近圓形），從而讓梯度下降演算法收斂得更快。
</details>

**Q2: 在測試 (Testing) 階段，Batch Normalization 如何獲取 $\mu$ 和 $\sigma$？**
<details>
<summary>點擊查看解答</summary>
答：測試階段無法取得足夠的 Batch，因此會使用訓練過程中記錄下來的 $\mu$ 和 $\sigma$ 的「移動平均值 (Moving Average)」來進行正規化。
</details>

**Q3: Batch Normalization 公式中 $\gamma$ 與 $\beta$ 的作用是什麼？**
<details>
<summary>點擊查看解答</summary>
答：它們是可學習的參數。正規化後的數值強制為平均 $0$、變異數 $1$，可能過度限制了模型的表達能力，引入 $\gamma$ 與 $\beta$ 允許神經網路學習恢復適合的分佈，保留靈活性。
</details>