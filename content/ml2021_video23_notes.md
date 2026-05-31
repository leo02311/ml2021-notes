---
title: "生成式對抗網路 (Generative Adversarial Network, GAN) (三) – 生成器效能評估與條件式生成"
tags:
  - MachineLearning
  - ML2021
  - GenerativeAdversarialNetwork
  - GAN
---

# 第23堂課：生成式對抗網路 (Generative Adversarial Network, GAN) (三) – 生成器效能評估與條件式生成

本堂課主要介紹生成式模型（Generative Models），核心重點在於生成對抗網路（Generative Adversarial Network, GAN）的基本原理、運作機制、挑戰及其變體，並討論生成模型的評估方式。

## 一、 為什麼需要生成模型？

在機器學習任務中，我們常希望網絡能夠具備一定的「創造力」。當相同的輸入可能對應多種不同的合理輸出時（例如：根據前幾幀預測下一個畫面的 Video Prediction，或根據問題生成多種回答的 Chatbot），簡單的監督式學習往往無法應對。

此時，我們引入一個簡單的機率分佈（例如隨機雜訊向量 $z$），將其作為網絡的輸入。網絡（Generator）學習將這個簡單的分佈轉換為複雜的分佈 $P_G$，從而實現多樣化的生成。

## 二、 生成對抗網路 (GAN)

### 1. 核心概念
GAN 的核心在於「對抗」：
- **Generator (G)**：試圖生成足以亂真的資料來欺騙 Discriminator。
- **Discriminator (D)**：一個二元分類器，學習區分真實資料與 Generator 生成的偽造資料。

### 2. 演算法流程
GAN 的訓練過程是一個迭代的過程：
- **步驟 1**：固定 Generator $G$，更新 Discriminator $D$。目的是讓 $D$ 能正確判斷真實資料（給高分）與生成資料（給低分）。
- **步驟 2**：固定 Discriminator $D$，更新 Generator $G$。目的是讓 $G$ 生成能獲得 $D$ 高分的結果（即「欺騙」$D$）。

### 3. 目標函數
訓練目標為一個 Min-Max 問題：
$$\min_{G} \max_{D} V(D, G) = \mathbb{E}_{y \sim P_{\text{data}}}[\log D(y)] + \mathbb{E}_{y \sim P_{G}}[\log(1 - D(y))]$$

---

## 三、 知識圖譜

```mermaid
graph TD
    "生成式模型" --> "GAN"
    "GAN" --> "Generator"
    "GAN" --> "Discriminator"
    "GAN" --> "挑戰"
    "挑戰" --> "Mode Collapse"
    "挑戰" --> "Mode Dropping"
    "GAN" --> "評估指標"
    "評估指標" --> "Inception Score"
    "評估指標" --> "FID"
    "GAN" --> "變體"
    "變體" --> "Conditional GAN"
    "變體" --> "Cycle GAN"
    "變體" --> "WGAN"
```

---

## 四、 進階議題與挑戰

### 1. Wasserstein GAN (WGAN)
傳統 GAN 使用 JS 分佈（JS Divergence）容易導致訓練不穩定。WGAN 透過計算 Wasserstein 距離（Earth Mover's Distance）來衡量分佈間的差異，使得訓練過程更有意義。
- 為了滿足 Lipschitz 約束，WGAN 提出了 **Gradient Penalty (GP)** 與 **Spectral Normalization** 等技巧。

### 2. Conditional GAN
當我們需要依據特定條件（例如文字敘述 $x$）生成對應影像時，需將 $x$ 輸入到 Generator 與 Discriminator 中。為了避免 Generator 忽略條件，Discriminator 必須同時檢查「影像是否真實」以及「影像與條件是否匹配」。

### 3. Cycle GAN 與非配對資料
在沒有成對資料 (Unpaired Data) 的情況下（如人臉轉動漫臉），Cycle GAN 利用「循環一致性」（Cycle Consistency）機制，將影像 $X \to Y \to X$，並確保重建後的 $X$ 與原始影像盡量接近。

---

## 五、 隨堂測驗

**Q1：在 GAN 的訓練中，為什麼我們通常不希望 Generator 只是背下訓練資料（Memory GAN）？**
<details>
<summary>點擊查看解答</summary>
因為如果模型只是記憶（Memorize）訓練資料，它就失去了生成新樣式、具備創造力的能力，這也是我們訓練生成模型的初衷——從簡單分佈映射到複雜分佈，而非簡單的複寫資料。
</details>

**Q2：為什麼說 JS Divergence 在 GAN 訓練中可能是不適用的？**
<details>
<summary>點擊查看解答</summary>
因為在 GAN 訓練初期，生成的分佈 $P_G$ 與真實資料分佈 $P_{\text{data}}$ 通常幾乎沒有重疊。在兩個分佈不重疊的情況下，JS Divergence 會呈現常數值（$\log 2$），導致梯度消失，Discriminator 無法提供有效的反饋給 Generator。
</details>

**Q3：Conditional GAN 的 Discriminator 應該具備什麼功能？**
<details>
<summary>點擊查看解答</summary>
除了判斷生成的影像是否看起來像真實影像（Real vs Fake），還必須判斷影像內容是否與輸入的條件（Condition，如文字描述）一致。
</details>

## 來自課程原聲的重點摘要

## 來自課程原聲的重點摘要

* **GAN 的訓練困難與邏輯**
    * 教授指出，儘管現在有 WGAN 等更先進的訓練方式，但 GAN 依然難以訓練。
    * 關鍵在於 Discriminator (鑑別器) 和 Generator (生成器) 是「互為敵手」的關係。Discriminator 負責辨別真實與虛假圖像，而 Generator 負責生成假圖來欺騙 Discriminator。
    * 若訓練過程不穩定，導致其中一方「停止進步」，另一方也會隨之崩潰。因此，訓練 GAN 需要精細調整超參數 (Hyper-parameters)，且通常需要兩者「齊頭並進」，任何一方放棄都會導致整場比賽無法進行。

* **GAN 的實際應用挑戰**
    * 教授提醒學生，不要將尚未訓練好的 GAN 模型直接放入作業中。
    * 訓練 GAN 本質上並不容易，建議學生可以搜尋網路上關於 "GAN tips" 的資源，參考相關文獻與連結來進行實作。

* **條件式 GAN (Conditional GAN) 的運作與概念**
    * 當目標是生成特定內容（例如一段文字）時，Conditional GAN 會成為最困難的部分。
    * 教授解釋其運作邏輯：若要生成文字，會有一個 sequence-to-sequence 的模型作為 Generator，它負責根據輸入條件生成文字，而 Discriminator 則負責評估生成的文字是否真實。
    * 教授進一步說明，要訓練這樣的模型，通常需要透過「監督式學習」將真實資料與其對應的標籤 (Label) 一併餵給模型。這不僅包含輸入，還包括 Discriminator 需要「看過」正確的圖文配對，才能判斷生成的文字與圖片是否匹配。

* **模型評估的難點：Inception Score 與 FID**
    * 教授提到，評估 Generator 生成品質的好壞非常困難，通常依賴「人工評估」，但這極度耗時。
    * 為了客觀化，學者發展出 Inception Score 和 FID 等量化指標。
    * **Inception Score (IS)**：將生成的圖片丟入一個預先訓練好的分類器 (如 Inception)，若分類器能高度確認圖片類別（機率分佈集中），則分數較高。但該方法有其局限，並不一定反映圖片的多樣性。
    * **FID (Fréchet Inception Distance)**：則是目前較為常用的指標，透過計算真實圖片與生成圖片特徵分佈的距離來衡量好壞，數值越低代表品質越好。
