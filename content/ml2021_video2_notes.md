---
title: 第二堂課：深度學習基礎 - 突破線性模型的限制
tags:
  - MachineLearning
  - DeepLearning
  - Optimization
---

# 突破線性模型的限制 (Model Bias)

在上一堂課中，我們認識了線性模型（Linear Model）：$y = b + w \cdot x_1$。
然而，線性模型非常受限，它永遠只能畫出一條直線。真實世界中，特徵 $x$ 與預測目標 $y$ 的關係往往是非線性的（例如：前一天觀看人數太高，隔天反而變少，形成一個有峰值的曲線）。
這種來自於模型本身的限制，我們稱之為 **Model Bias**。

為了減少 Model Bias，我們需要寫出一個**更有彈性、帶有未知參數的複雜函數**。

---

## 2. 使用分段線性函數逼近連續曲線

任何連續的非線性曲線，都可以透過取點連線的方式，近似成**分段線性曲線 (Piecewise Linear Curves)**。
而每一個分段線性曲線，都可以看作是一個常數，加上許多個「藍色函數 (Hard Sigmoid)」的組合。

### 藍色函數 (Hard Sigmoid) 的特性
* 當 $x$ 小於某個閾值時為定值。
* 當 $x$ 大於某個閾值時為另一個定值。
* 中間存在一段斜坡。

只要我們有足夠多的藍色函數（設定不同的轉折點與斜率）並把它們相加，就能完美重構出任何複雜的 Piecewise Linear 曲線。

---

## 3. Sigmoid 函數：讓神經網路平滑化

由於 Hard Sigmoid 是有折角的，數學上較難處理，因此我們引入了一條 S 型的平滑曲線來逼近它，這就是鼎鼎大名的 **Sigmoid Function**：

$$ y = c \cdot \frac{1}{1 + e^{-(b + w x_1)}} $$

* **$w$ (Weight)**：改變斜坡的坡度。
* **$b$ (Bias)**：將 Sigmoid 函數左右平移。
* **$c$ (Constant)**：改變函數的最高高度。

只要組合足夠多個不同參數的 Sigmoid，就能組裝出極度複雜的預測模型。

---

## 4. 線性代數表示法：神經網路的雛形

當我們考慮多個特徵 (Feature) 以及多個 Sigmoid 時，可以將這些複雜的加總用矩陣 (Matrix) 與向量 (Vector) 簡潔地表示：

1. **特徵相乘**：$\mathbf{r} = \mathbf{W}\mathbf{x} + \mathbf{b}$
2. **激勵函數**：$\mathbf{a} = \sigma(\mathbf{r})$ （其中 $\sigma$ 為 Sigmoid）
3. **最終輸出**：$y = \mathbf{c}^T \mathbf{a} + b$

我們將所有的未知數 ($\mathbf{W}, \mathbf{b}, \mathbf{c}, b$) 全部展開並拉直，統稱為一個巨大的向量 **$\theta$**。

---

## 5. Loss 與最佳化 (Optimization)

與線性模型完全相同，現在我們擁有了超多參數 $\theta$，我們依然需要定義 **Loss Function $L(\theta)$** 來衡量參數的好壞。

尋找最佳參數 $\theta^*$ 的暴力破解法已經不可能做到，我們必須仰賴 **梯度下降 (Gradient Descent)**：
1. 隨機初始化 $\theta^0$。
2. 對每個參數計算微分（即計算 Gradient $\mathbf{g} = \nabla L(\theta^0)$）。
3. 更新參數：$\theta^1 = \theta^0 - \eta \mathbf{g}$ （$\eta$ 為 Learning Rate）。

---

## 🧠 知識圖譜 (Knowledge Graph)

```mermaid
graph TD
    A[線性模型 Limitation] -->|Model Bias| B[無法擬合複雜曲線]
    B --> C[解決方案: 彈性函數]
    C --> D[Piecewise Linear Curves]
    D --> E[Hard Sigmoid 組合]
    E --> F[Sigmoid 逼近平滑化]
    
    F --> G[參數矩陣化]
    G --> H["r = Wx + b"]
    H --> I["a = σ(r)"]
    I --> J["y = c^T a + b"]
    
    J --> K[定義 Loss L(θ)]
    K --> L[Gradient Descent 最佳化]
    L --> M[更新參數 θ = θ - η∇L]
```

---

## 📝 課後測驗

**Q1：什麼是 Model Bias？**
<details>
<summary>點擊查看答案</summary>
Model Bias 指的是模型本身天生的限制。例如線性模型只能擬合直線，無法模擬真實世界複雜的非線性關係。
</details>

**Q2：在 Sigmoid 函數中，調整參數 $w$ 與 $b$ 分別會改變圖形的什麼？**
<details>
<summary>點擊查看答案</summary>
調整 $w$ (Weight) 會改變 S 型曲線斜坡的「坡度」，而調整 $b$ (Bias) 則會將曲線「左右平移」。
</details>

**Q3：當模型的參數數量從兩個擴張到一萬個時，為何不能使用暴力嘗試法找最佳解？需要用什麼演算法？**
<details>
<summary>點擊查看答案</summary>
因為參數過多時，組合數呈指數爆炸。我們必須使用「梯度下降 (Gradient Descent)」計算 Loss 對每一個參數的微分 (Gradient)，來指引參數更新的方向。
</details>
