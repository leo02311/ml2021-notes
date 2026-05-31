---
title: "Generative Adversarial Network (GAN) (Part 1 of 4)"
tags:
  - MachineLearning
  - ML2021
  - GenerativeAdversarialNetworks
  - GAN
---

# 第21堂課：Generative Adversarial Network (GAN) (Part 1 of 4)

本堂課介紹了生成式模型（Generative Models）的核心概念，重點探討生成式對抗網路（GAN）的運作原理、訓練技巧以及在各種任務（如條件生成、非配對資料學習）中的應用。

## 1. 什麼是生成模型 (Generation)？

生成模型的基本概念是利用一個神經網路作為「產生器」(Generator)。
*   **輸入**：一個簡單的分佈（例如常態分佈）中的向量 $z$。
*   **輸出**：目標複雜分佈中的資料 $y$。
*   **目的**：學習複雜分佈的結構，使得產生出的資料 $y$ 在視覺或統計上與真實數據相似。

## 2. 生成式對抗網路 (GAN) 的基本架構

GAN 由兩個主要的網路組成，它們處於競爭關係：
*   **Generator (G)**：試圖生成足以以假亂真的資料。
*   **Discriminator (D)**：一個分類器，負責判斷輸入的資料是真實資料（Real）還是生成器產生的假資料（Fake）。

### 演算法流程
在每一輪訓練迭代中：
1.  **更新 Discriminator (D)**：固定 $G$，讓 $D$ 學習將真實資料評為高分（1），將 $G$ 生成的資料評為低分（0）。
2.  **更新 Generator (G)**：固定 $D$，讓 $G$ 學習如何產生資料以最大化 $D$ 的分數，即「欺騙」$D$。

## 3. 理論基礎：分佈間的距離

GAN 的目標是最小化生成分佈 $P_G$ 與真實數據分佈 $P_{data}$ 之間的差異（Divergence）：
$$G^* = \arg \min_G \text{Div}(P_G, P_{data})$$

### 為什麼 JS Divergence 有問題？
當 $P_G$ 與 $P_{data}$ 在高維空間中沒有重疊（Low-dim manifold）時，JS Divergence 恆等於 $\log 2$，這導致訓練過程中的梯度消失，使得判別器無法提供有效的反饋。

### Wasserstein Distance
為了改善此問題，引入了 Wasserstein Distance（又稱 Earth Mover's distance），計算將一個分佈轉換為另一個分佈所需的最小「搬運」成本。
*   **WGAN** 的核心在於確保判別器（Critic）滿足 **1-Lipschitz** 約束，常用的方法包括 Weight Clipping 或 Gradient Penalty。

## 4. 條件式生成 (Conditional Generation)

當我們希望生成的結果受特定條件（如文字描述、標籤）控制時，會使用 Conditional GAN (cGAN)。
*   **cGAN 架構**：Generator 和 Discriminator 皆接收條件 $x$ 作為額外輸入。
*   **pix2pix**：一種著名的條件式 GAN 應用，用於圖像翻譯任務，例如將標籤轉為地景、將素描轉為照片。

## 5. 無配對資料學習 (Learning from Unpaired Data)

在許多應用中，我們難以取得配對資料，因此發展出 **Cycle GAN**：
*   利用「循環一致性」（Cycle Consistency）機制：$G_{x\to y}(x)$ 轉換後的資料再經過 $G_{y\to x}$ 應該要回到原來的 $x$。
*   透過此機制，在沒有對應影像的情況下也能學習不同領域（Domain）之間的轉換。

## 6. 評估指標 (Evaluation)

評估生成模型品質非常困難：
*   **Inception Score (IS)**：衡量生成品質與多樣性。
*   **Fréchet Inception Distance (FID)**：計算真實資料與生成資料在特徵空間中分佈的距離，數值越小代表越好。

---

## 知識圖譜

```mermaid
graph TD
    "Generative Models" --> "GAN"
    "GAN" --> "Discriminator"
    "GAN" --> "Generator"
    "GAN" --> "Training Tips"
    "GAN" --> "Conditional GAN"
    "GAN" --> "Cycle GAN"
    "Conditional GAN" --> "pix2pix"
    "Evaluation" --> "Inception Score"
    "Evaluation" --> "FID"
    "Theory" --> "JS Divergence"
    "Theory" --> "Wasserstein Distance"
```

---

## 隨堂測驗

1. **為什麼在傳統 GAN 中使用 JS Divergence 在某些情況下效果不佳？**
   <details>
   <summary>點擊查看解答</summary>
   當生成分佈 $P_G$ 與真實數據分佈 $P_{data}$ 在高維空間中互不重疊時，JS Divergence 的值會保持常數（$\log 2$），這導致判別器無法提供有意義的梯度來更新產生器。
   </details>

2. **在 Cycle GAN 中，「循環一致性」(Cycle Consistency) 的作用是什麼？**
   <details>
   <summary>點擊查看解答</summary>
   Cycle Consistency 透過強制要求從一個領域轉換到另一個領域再轉回來後的資料，必須與原始輸入盡可能接近，從而解決在缺乏配對資料時，模型可能會忽略輸入內容（即將所有輸入映射到相同輸出）的問題。
   </details>

3. **FID (Fréchet Inception Distance) 評估指標中，「越小越好」代表什麼意義？**
   <details>
   <summary>點擊查看解答</summary>
   FID 衡量的是生成資料分佈與真實資料分佈在特徵空間中的距離（假設為高斯分佈），數值越小，代表生成的分佈越接近真實數據的分佈，即生成的品質越高。
   </details>

## 來自課程原聲的重點摘要

## 來自課程原聲的重點摘要

* **生成器 (Generator) 的核心觀念**
    * **與傳統網絡的差異**：過去我們學習的網絡，給定輸入 $x$ 就能得到固定輸出 $y$。但在生成器中，我們引入了一個「隨機變數 $z$」，它是從某個分佈中採樣出來的。
    * **為什麼要加 $z$？**：如果沒有 $z$，輸出將完全取決於固定的 $x$。加入 $z$ 後，輸出就變成「輸入 $x$」與「隨機變數 $z$」的組合。這使得每次輸出都不一樣，變成了「隨機的」或是「分佈式的」。
    * **實際應用舉例**：在影片預測中，如果給定過去的遊戲畫面，機器要預測下一幀。由於遊戲的未來可能有不同的變化，引入 $z$ 可以讓模型預測出多種可能，而不僅僅是預測單一結果。

* **生成器的運作與訓練邏輯**
    * **簡單分佈的要求**：生成器需要處理的隨機分佈必須是「簡單的」（例如高斯分佈或均勻分佈），因為我們要明確知道如何從中採樣。
    * **如何訓練生成器**：透過訓練，我們希望生成器的輸出能夠接近真實數據的期望。當我們輸入不同的 $z$，生成器會輸出不同的結果，這些結果的集合就形成了一種分佈，這個目標分佈就是我們要機器學習的對象。

* **生成對抗網絡 (GAN) 的關鍵概念**
    * **判別器 (Discriminator) 的角色**：生成器負責「產生」圖片，判別器負責「打分」。判別器其實就是一個分類器，當它看到真實圖片，目標輸出接近 1；看到生成器製造的假圖，目標輸出接近 0。
    * **攻防對決**：
        * 生成器與判別器構成了一個 adversarial（對抗）關係。
        * 訓練過程是一個交替進行的過程：先固定生成器，訓練判別器來更好地區分真假圖；接著固定判別器，訓練生成器來產生能欺騙判別器的圖片。
        * 透過這種方式，生成器會不斷進化，最終產生出極其逼真的圖片。

* **模型訓練的常見難點**
    * **模式崩潰 (Mode Collapse)**：這是訓練 GAN 最常見的錯誤之一。教授提到機器可能學會了「偷懶」，它發現某幾種特定的圖片特別能欺騙判別器，於是它就不再產生多樣化的圖片，而是一直複製那幾張「特強」圖片。這時生成的結果就會變得很單調，缺乏多樣性。
    * **訓練穩定性**：訓練 GAN 非常依賴技巧，如果參數沒調好，或是生成器與判別器的能力懸殊太大，訓練就無法收斂，結果可能是一堆無意義的雜訊。

* **老師的教學風格提醒**
    * 李宏毅教授特別喜歡運用動漫、電玩等流行文化進行解說。
    * 他強調「直覺」大於「複雜數學」。在解說訓練流程時，他常將複雜的數學梯度推導比喻為「警察與偽鈔製造者」或「動漫角色的爭鬥」，這對於建立對 GAN 運作機制的體感非常有幫助。
