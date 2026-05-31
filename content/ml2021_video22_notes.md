---
title: "Generative Adversarial Network (GAN) (Part 2 of 4)"
tags:
  - MachineLearning
  - ML2021
  - GAN
  - GenerativeModel
---

# 第22堂課：Generative Adversarial Network (GAN) (Part 2 of 4)

本堂課主要介紹生成式模型（Generative Models），特別是生成對抗網路（Generative Adversarial Network, GAN）的原理、架構與應用，並討論生成模型的評估指標及處理非配對數據（Unpaired Data）的技術。

## 1. 生成式模型簡介

### 核心概念
生成式模型的目標是讓模型學會數據的分佈，並能從中進行採樣（Sample）。
- **輸入**：一個簡單的分佈（通常是從標準常態分佈採樣得到的向量 $z$）。
- **網路 (Network)**：作為生成器（Generator, $G$），將簡單的 $z$ 轉換為複雜的 $y$。
- **目標**：讓 $G(z)$ 的輸出分佈 $P_G$ 盡可能接近真實數據的分佈 $P_{data}$。

### 為什麼需要分佈？
在創造性任務（如繪圖、聊天機器人、影片預測）中，相同的輸入可能對應多種合理的輸出。因此，模型不僅需要輸出單一結果，還要能捕捉數據的多樣性，這就是「分佈」存在的意義。

## 2. 生成對抗網路 (GAN)

### 基本原理
GAN 由兩個競爭的網路組成：
- **生成器 (Generator, $G$)**：負責生成數據，試圖欺騙判別器。
- **判別器 (Discriminator, $D$)**：一個二元分類器，判斷輸入是真實數據還是生成器產生的「假」數據。

### 訓練算法
訓練過程包含兩個交替步驟：
1. **更新判別器 $D$**：固定 $G$，將真實數據給予高分，生成數據給予低分。
2. **更新生成器 $G$**：固定 $D$，透過梯度反向傳播，讓 $G$ 產出的內容在 $D$ 中獲得更高的分數（即欺騙 $D$）。

```mermaid
graph LR
    "標準常態分佈 z" --> "生成器 G"
    "真實數據" --> "判別器 D"
    "生成器 G" --> "生成數據 y"
    "生成數據 y" --> "判別器 D"
    "判別器 D" --> "輸出分數"
```

## 3. 理論基礎與挑戰

### 目標函數
我們的目標是 $G^* = \arg \min_G \text{Div}(P_G, P_{data})$，即最小化真實分佈與生成分佈之間的差異。

### JS 散度 (Jensen-Shannon Divergence) 的問題
在原始 GAN 中，判別器的目標與 JS 散度有關。若兩個分佈 $P_G$ 與 $P_{data}$ 沒有重疊（在現實高維空間中極為常見），JS 散度將恆為 $\log 2$，導致梯度消失，無法進行有效訓練。

### Wasserstein Distance (Earth Mover's Distance)
為解決上述問題，WGAN 引入了 Wasserstein 距離，它衡量的是將分佈 $P$ 變換為 $Q$ 所需移動的「土方量」，即便沒有重疊也能給出連續且有意義的梯度。

## 4. 進階生成架構

- **Conditional GAN (CGAN)**：不僅輸入 $z$，還加入條件 $c$ (如：輸入「紅眼」文字，生成紅眼動漫角色)，使生成內容可控。
- **Cycle GAN**：解決非配對數據（Unpaired Data）的問題，透過週期一致性損失（Cycle consistency loss）讓模型能學會「X 領域到 Y 領域」的轉換，無需成對標記資料。
- **StarGAN**：將多領域轉換整合在一個模型中。

## 5. 生成模型的評估

評估生成質量通常是困難的，常用指標包括：
- **Inception Score (IS)**：評估生成的圖像類別是否清晰（Concentrated）且多樣（Diverse）。
- **Fréchet Inception Distance (FID)**：衡量真實數據特徵與生成數據特徵在嵌入空間中的高斯分佈差異，數值越小越好。

---

## 隨堂測驗

**Q1: 為何在生成式任務中我們傾向使用「分佈」而非單一預測？**
<details>
<summary>點擊查看解答</summary>
因為在許多具有「創造性」的任務中，相同的輸入可能對應多種同樣合理的輸出（例如影片預測中物體的轉向，或聊天機器人的多種回答方式），使用分佈可以捕捉這種多樣性。
</details>

**Q2: 為什麼原始 GAN 在判別器訓練得太好時，會導致生成器難以訓練？**
<details>
<summary>點擊查看解答</summary>
當判別器過於強大時，真實分佈與生成分佈的重疊區域變得很小，導致 JS 散度固定在 $\log 2$。此時梯度接近於 0，生成器無法透過反向傳播獲得有效的參數更新訊號。
</details>

**Q3: 在 Cycle GAN 中，為什麼需要「週期一致性損失 (Cycle consistency loss)」？**
<details>
<summary>點擊查看解答</summary>
如果沒有週期一致性，生成器可能會忽略輸入並學習將任意輸入對應到目標領域中的某一張特定圖片（模式崩塌）。透過將轉換後的圖片再轉回原始領域並要求與輸入一致，可以確保生成器確實學習了輸入與輸出之間的轉換關係。
</details>

## 來自課程原聲的重點摘要

## 來自課程原聲的重點摘要

* **GAN 的基本運作機制與目標**
    * **生成器 (Generator) 的目標**：並非最小化或最大化一個簡單的損失函數。給定一個隨機分布（正常分布）的向量，生成器要輸出一個複雜的分布（稱為 $P_G$）。
    * **核心概念**：我們希望生成的 $P_G$ 與真實資料分布 $P_{data}$ 越接近越好。
    * **直觀理解**：教授以一維分布為例，說明 generator 透過輸入隨機資料點，經過 generator 轉換後，會形成一個新的分布，該分布的目標就是模仿真實資料的分布。

* **如何衡量 $P_G$ 與 $P_{data}$ 的距離 (Divergence)**
    * **Divergence 的意義**：可以視為兩個分布之間的「距離」。若 Divergence 很大，代表兩個分布差異很大；反之，則很接近。
    * **訓練 GAN 的目標**：就是要找到一組 generator 的參數，使得這兩個分布的 Divergence 最小化。

* **訓練困難點：Divergence 的計算**
    * **計算挑戰**：若要使用 KL 或 JS Divergence，在連續分布上會涉及極度複雜的積分運算，在實務上幾乎不可行。
    * **GAN 的突破**：由於無法直接計算 Divergence，GAN 透過判別器 (Discriminator) 輔助。

* **判別器 (Discriminator) 的角色**
    * **訓練目標**：給予真實圖片高分，給予生成圖片低分。這本質上是一個二元分類問題。
    * **與生成器的互動**：Discriminator 就像是一個訓練過後的「分類器」。它能幫助生成器找到一個方向，將生成分布推向真實分布。

* **WGAN (Wasserstein GAN) 的概念**
    * **為什麼要提出 WGAN？** 原本的 JS Divergence 在兩個分布沒有重疊時，梯度可能消失或變得不穩定。
    * **Wasserstein Distance (Earth Mover's Distance)**：這是一種更好的「距離」衡量標準。它計算將一個分布變成另一個分布所需的「最小搬運成本」。
    * **優勢**：即使兩個分布完全不重疊，Wasserstein Distance 仍然能提供平滑的梯度，讓訓練更穩定，解決了原始 GAN 訓練時容易出現的模式坍塌 (Mode Collapse) 或梯度消失問題。

* **訓練小撇步：Lipschitz 連續性與 Gradient Penalty**
    * **Lipschitz 限制**：在 WGAN 中，為了維持計算 Wasserstein Distance 的有效性，必須滿足 Lipschitz 連續條件。
    * **Gradient Penalty**：為了強制滿足 Lipschitz 條件，實務上會對判別器的梯度進行懲罰，這就是 WGAN-GP 的由來。這讓訓練更加健壯。

* **對「優化問題」的思考**
    * 教授特別強調，訓練 GAN 其實是在解決一個 Min-Max 的博弈問題：生成器試圖最小化 Divergence (或與 Discriminator 對抗)，而判別器試圖最大化其分類準確度。
