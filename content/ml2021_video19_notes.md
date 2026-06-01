---
title: "第19堂課：Transformer (Part 1 of 2)"
tags:
  - MachineLearning
  - ML2021
  - BatchNormalization
  - DeepLearning
---

# 第19堂課：Transformer (Part 1 of 2)

本堂課主要介紹深度學習中極為重要的技術——**批量歸一化 (Batch Normalization, BN)**。透過對神經網路中的特徵進行歸一化，可以有效改善訓練時的優化困難，提升收斂速度。

## 1. 為什麼需要 Normalization？

在優化問題中，如果損失函數（Loss Function）的等高線呈現極度狹長（狹窄的峽谷狀），梯度下降會因為梯度在某些維度過大而產生震盪，導致收斂緩慢；反之，若等高線呈現圓形，梯度方向會直接指向最小值，優化過程會變得非常平滑。

*   **輸入特徵的差異**：若輸入特徵的數值範圍差距過大（例如 $x_1$ 範圍是 $[1, 2]$，$x_2$ 範圍是 $[100, 200]$），會導致 Loss Surface 變得極不平坦。
*   **Feature Normalization**：透過對每個維度 $i$ 計算平均值 $m_i$ 與標準差 $\sigma_i$，進行正規化：
    $$\tilde{x}_i^r \leftarrow \frac{x_i^r - m_i}{\sigma_i}$$
    使得所有維度的特徵平均值為 0，變異數為 1。這能讓優化過程更穩定。

## 2. 深度學習中的 Batch Normalization

在深層神經網路中，不僅輸入層需要歸一化，隱藏層的輸出 $z$ 也同樣面臨不同維度數值範圍不一的問題。

### 核心演算法
對於 Batch 中的每一個樣本輸出 $z^i$，我們計算該 Batch 的平均值 $\mu$ 與標準差 $\sigma$：
$$\mu = \frac{1}{3} \sum_{i=1}^{3} z^i$$
$$\sigma = \sqrt{\frac{1}{3} \sum_{i=1}^{3} (z^i - \mu)^2}$$

接著，進行標準化並加入可學習的參數 $\beta$ 與 $\gamma$：
$$\tilde{z}^i = \frac{z^i - \mu}{\sigma}$$
$$\hat{z}^i = \gamma \odot \tilde{z}^i + \beta$$
其中 $\gamma$ 與 $\beta$ 允許網路學習適當的偏移與縮放，確保歸一化不會限制模型的表達能力。

### 測試階段 (Testing Stage)
在測試時，我們可能沒有一個完整的 Batch 可以計算 $\mu$ 與 $\sigma$。因此，通常使用訓練過程中統計出的**移動平均值 (Moving Average)** 來進行歸一化：
$$\bar{\mu} \leftarrow p\bar{\mu} + (1 - p)\mu^t$$

## 3. 知識圖譜

```mermaid
graph TD
    A['Batch Normalization'] --> B['解決優化困難']
    B --> C['平滑 Loss Surface']
    A --> D['標準化流程']
    D --> E['計算 Batch 統計量 mu, sigma']
    D --> F['應用偏移與縮放 beta, gamma']
    A --> G['測試階段']
    G --> H['使用移動平均 Moving Average']
    A --> I['機制探討']
    I --> J['Internal Covariate Shift (傳統觀點)']
    I --> K['改變 Loss Landscape (實驗證據)']
```

## 4. 關於 Internal Covariate Shift 的辯論

過去認為 BN 的有效性是因為它解決了「內部協變量偏移 (Internal Covariate Shift)」，即防止隱藏層輸入分佈隨著權重更新而大幅改變。然而，後續研究顯示，BN 的主要貢獻在於它**改變了損失函數的表面 (Landscape)**，使其在訓練時更容易優化。這種正面影響在某種程度上甚至被視為是「偶然的 (Serendipitous)」。

---

### 隨堂測驗

**Q1：為什麼在梯度下降時，如果 Loss Surface 的等高線呈現狹長狀會導致訓練困難？**
<details>
<summary>點擊查看解答</summary>
當 Loss Surface 狹長時，梯度下降在某些方向上會產生劇烈震盪，導致無法穩定地朝向最小值收斂，增加了優化的難度與時間。
</details>

**Q2：在測試階段 (Testing Stage)，為什麼不能像訓練時直接計算 Batch 的 $\mu$ 與 $\sigma$？**
<details>
<summary>點擊查看解答</summary>
因為在測試階段，我們通常是一次處理一個樣本（或無法保證有足夠大的 Batch），無法計算出代表性的 Batch 統計量，因此必須使用訓練期間累積的移動平均值。
</details>

**Q3：Batch Normalization 公式中的參數 $\beta$ 與 $\gamma$ 有什麼作用？**
<details>
<summary>點擊查看解答</summary>
這兩個是可學習的參數，目的是為了讓模型能夠自行學習是否需要將正規化後的數值進行平移（$\beta$）或縮放（$\gamma$），確保正規化不會損害網路的表達能力。
</details>