---
title: "Generative Adversarial Network (GAN) (Part 3 of 4)"
tags:
  - MachineLearning
  - ML2021
  - GAN
  - DeepLearning
---

# 第24堂課：Generative Adversarial Network (GAN) (Part 3 of 4)

本堂課主要介紹生成式模型（Generative Models），特別是生成對抗網路（GAN）的基礎原理、運作機制及其變體。

## 一、 為何需要生成式模型？
在機器學習中，我們通常將網路視為一個函數。當任務需要「創造力」（Creativity）時，單一輸入對應單一輸出已不足夠。
- **範例**：影片預測（Video Prediction）、繪圖（Drawing）、聊天機器人（Chatbot）。
- **作法**：透過一個簡單的分佈（如常態分佈）採樣出向量 $z$，輸入到網路（Generator）中，生成複雜分佈的輸出 $y$。這使得同一個輸入可以產生多種不同的輸出。

## 二、 生成對抗網路 (GAN) 的基本概念
GAN 的核心概念是「寫作敵人，唸做朋友」。

### 1. 訓練演算法
GAN 包含兩個主要部件：**生成器 (Generator, G)** 與 **判別器 (Discriminator, D)**。訓練過程中，兩者互相博弈：
*   **Step 1：固定 Generator，更新 Discriminator**
    *   Discriminator 的任務是給真實資料高分，給生成資料低分。
    *   這本質上是一個二元分類問題。
*   **Step 2：固定 Discriminator，更新 Generator**
    *   Generator 的任務是生成能「欺騙」Discriminator 的資料，讓 Discriminator 給予高分。

### 2. 目標函數
我們的目標是讓 Generator 生成的機率分佈 $P_G$ 與真實資料分佈 $P_{data}$ 越接近越好：
$$G^* = \arg \min_G \text{Div}(P_G, P_{data})$$
Discriminator 的優化目標（Objective Function）則為：
$$V(G, D) = E_{y \sim P_{data}}[\log D(y)] + E_{y \sim P_G}[\log(1 - D(y))]$$

## 三、 生成模型的知識圖譜

```mermaid
graph TD
    "Generative Models" --> "GAN"
    "Generative Models" --> "VAE"
    "Generative Models" --> "Flow-based Model"
    "GAN" --> "Conditional GAN"
    "GAN" --> "Cycle GAN"
    "GAN" --> "WGAN"
    "Conditional GAN" --> "Text-to-image"
    "Conditional GAN" --> "Talking Head Generation"
    "Cycle GAN" --> "Unsupervised Translation"
    "GAN" --> "Evaluation"
    "Evaluation" --> "Inception Score"
    "Evaluation" --> "FID"
```

## 四、 進階與挑戰
*   **JS Divergence 的問題**：當真實分佈與生成分佈不重疊時，JS 散度無法提供有效的梯度，導致 Discriminator 訓練過快，Generator 無法學習。
*   **WGAN (Wasserstein GAN)**：引入 Wasserstein 距離（Earth Mover's Distance）解決分佈不重疊的問題。為了滿足 1-Lipschitz 限制，提出了 WGAN-GP (Gradient Penalty) 等方法。
*   **模式崩潰 (Mode Collapse)**：指 Generator 只學會生成極少數種類的樣本（缺乏多樣性）。
*   **評估指標**：
    *   **Inception Score (IS)**：評估生成圖片的品質與多樣性。
    *   **Fréchet Inception Distance (FID)**：比較真實資料與生成資料在特徵空間中的分佈距離，越小越好。

---

## 隨堂測驗

### 1. 在 GAN 的訓練過程中，Discriminator 的主要任務是什麼？
<details>
<summary>點擊查看解答</summary>
Discriminator 的任務是作為一個二元分類器，將真實資料標記為 1（高分），將生成器產出的虛假資料標記為 0（低分），從而協助生成器不斷改進。
</details>

### 2. 為什麼標準 GAN 在真實分佈與生成分佈不重疊時會遇到訓練困難？
<details>
<summary>點擊查看解答</summary>
因為標準 GAN 基於 JS 散度。當兩個分佈完全不重疊時，JS 散度會維持在常數（如 $\log 2$），這意味著梯度會消失，Discriminator 能輕易達到 100% 準確率，使得 Generator 無法獲得有效的更新訊號。
</details>

### 3. Cycle GAN 的核心特性是什麼？
<details>
<summary>點擊查看解答</summary>
Cycle GAN 主要用於非配對資料（Unpaired Data）的轉換。其核心在於「循環一致性（Cycle Consistency）」，即從 $X$ 轉換到 $Y$ 再轉回 $X$ 後，必須與原始輸入盡可能接近，藉此在沒有對應標籤的情況下學習領域之間的映射。
</details>

## 來自課程原聲的重點摘要

## 來自課程原聲的重點摘要

* **GAN 的訓練困難與機制**
  * 李宏毅教授比喻 GAN 中的生成器（Generator）與判別器（Discriminator）是一對「互相砥礪」的對手：生成器努力產出逼真的偽造品，而判別器則致力於區分真假。兩者會隨著訓練過程同步成長，但若其中一方停止訓練，另一方也會失去進步的動力，最終導致訓練失敗。
  * **關鍵觀念**：訓練判別器時，若無法達到良好的最適化（well-optimized），將導致判別器無法分辨真偽，進而讓生成器失去改善依據。兩者必須維持競爭平衡。

* **超參數調整（Hyper-parameter Tuning）的兩難**
  * 教授強調，在訓練過程中，我們無法隨時調整超參數，只能「祈禱」判別器的損失（loss）能隨著訓練步數穩定下降。若損失沒有下降，整個訓練過程就很難成功。

* **Token 的定義取決於任務**
  * 在討論生成文本時，Token 的定義是靈活的：
    * 中文可能是字（Character）。
    * 英文可能是字母（Letter）。
    * 英文也可能是單詞（Word，以空白區隔）。
  * 教授強調，Token 的最小單位由使用者根據任務需求自行定義。

* **為什麼梯度下降（Gradient Descent）可能無效？**
  * 學生常疑惑為何 Max Pooling 可能影響梯度下降，教授指出這與目標函數（Objective Function）的平滑度有關。當參數只有微小變動，卻造成輸出分佈（Output Distribution）大幅跳動時，梯度計算會失效。

* **關於 GAN 評估（Evaluation）的迷思**
  * 早期 GAN 研究常缺乏客觀指標，僅靠視覺評估（Human eyes）。教授點出這是不嚴謹的，並介紹了現代評估 GAN 的方式：使用預訓練好的影像分類模型，將生成圖像作為輸入，觀察輸出分佈。若分佈極度集中，代表生成的圖像多樣性不足；若分佈均勻，則代表模型產生了具有辨識度且多樣的類別。

* **模式崩潰（Mode Collapse）現象**
  * 當生成器只學會產生幾種特定類型的圖像（例如只有單一臉孔角度），而忽略了資料集的多樣性，即使判別器無法辨識其為假貨，這仍是訓練失敗的一種，稱為「模式崩潰」。教授建議透過調整網路結構或學習策略來改善。
