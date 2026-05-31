---
title: "Adversarial Attack (Part 2 of 2)"
tags:
  - MachineLearning
  - ML2021
  - AdversarialAttack
  - DeepLearning
---

# 第32堂課：Adversarial Attack (Part 2 of 2)

在機器學習模型部署到現實世界時，我們必須考慮其魯棒性（Robustness）。這堂課介紹了「對抗式攻擊」（Adversarial Attack），探討如何透過微小的擾動（perturbation）讓神經網路產生錯誤判斷。

---

## 一、 對抗式攻擊概論

### 1. 核心動機
訓練好的神經網路在實際應用中（如垃圾郵件過濾、惡意軟體偵測、網路入侵偵測等）可能面臨人為刻意設計的輸入，試圖「愚弄」模型，導致錯誤的分類。

### 2. 攻擊分類
*   **非目標式攻擊 (Non-targeted attack)**：只要讓模型分類結果不是正確答案即可（例如：只要不是「貓」都可以）。
*   **目標式攻擊 (Targeted attack)**：指定讓模型誤判為特定的類別（例如：指定判斷為「海星」）。

### 3. 攻擊數學形式
假設輸入為 $x^0$，我們尋找一個被修改後的輸入 $x$，使得：
*   **非目標式**：$x^* = \arg \min L(x)$，其中 $L(x) = -e(y, \hat{y})$，$e$ 為評估函數。
*   **目標式**：$L(x) = -e(y, \hat{y}) + e(y, y^{target})$
*   **限制條件**：$d(x^0, x) \le \varepsilon$，即人類無法察覺擾動的差異。

---

## 二、 擾動與距離度量

為了保證攻擊對人類是「不可見的」，我們需要限制修改幅度：
1.  **$L_2$-norm**：$d(x^0, x) = \|\Delta x\|_2 = \sqrt{(\Delta x_1)^2 + (\Delta x_2)^2 + \dots}$
2.  **$L_\infty$-norm**：$d(x^0, x) = \|\Delta x\|_\infty = \max(|\Delta x_1|, |\Delta x_2|, \dots)$

---

## 三、 攻擊演算法

### 1. 梯度下降法 (Gradient Descent)
不同於訓練模型時更新參數 $w, b$，攻擊時我們**固定參數，更新輸入 $x$**：
$$x^t \leftarrow x^{t-1} - \eta g$$
其中 $g = \left. \frac{\partial L}{\partial x} \right|_{x=x^{t-1}}$。若超出範圍 $\varepsilon$，則進行截斷（fix）。

### 2. Fast Gradient Sign Method (FGSM)
為了加速運算並滿足 $L_\infty$ 限制，只取梯度的符號（sign）：
$$g = \text{sign}\left( \left. \frac{\partial L}{\partial x} \right|_{x=x^{t-1}} \right)$$

### 3. 黑箱攻擊 (Black Box Attack)
若無法取得模型參數，可採取以下策略：
*   **代理模型 (Proxy Model)**：訓練一個功能類似的代理模型，對其進行攻擊，再將生成的擾動應用於目標模型。
*   **集成攻擊 (Ensemble Attack)**：同時考慮多個模型的弱點進行攻擊。

---

## 四、 知識圖譜

```mermaid
graph TD
    "Adversarial Attack" --> "Goal"
    "Adversarial Attack" --> "Method"
    "Adversarial Attack" --> "Defense"
    
    "Goal" --> "Non-targeted"
    "Goal" --> "Targeted"
    
    "Method" --> "White Box Attack"
    "Method" --> "Black Box Attack"
    
    "White Box Attack" --> "FGSM"
    "White Box Attack" --> "Iterative FGSM"
    
    "Defense" --> "Passive Defense"
    "Defense" --> "Proactive Defense"
    
    "Passive Defense" --> "Smoothing"
    "Passive Defense" --> "Compression"
    
    "Proactive Defense" --> "Adversarial Training"
```

---

## 五、 防禦機制

### 1. 被動防禦 (Passive Defense)
在輸入進入網路前進行處理，目的是去除攻擊訊號，但不影響原始分類能力：
*   **平滑化 (Smoothing)**
*   **圖像壓縮**
*   **隨機化 (Randomization)**：例如隨機 resize 或 padding。

### 2. 主動防禦 (Proactive Defense)
*   **對抗式訓練 (Adversarial Training)**：將對抗樣本加入訓練集，視為一種數據增強（Data Augmentation），讓模型學會對抗擾動。

---

## 六、 隨堂測驗

**Q1：在對抗式攻擊中，與一般的模型訓練相比，主要的區別是什麼？**
<details>
<summary>點擊查看解答</summary>
對抗式攻擊是「更新輸入 $x$」以最小化損失函數，而一般模型訓練是「更新模型參數 $w, b$」。
</details>

**Q2：FGSM（Fast Gradient Sign Method）演算法為什麼要取梯度的 sign（符號）？**
<details>
<summary>點擊查看解答</summary>
這是一種滿足 $L_\infty$-norm 限制的快速近似方法，透過只取梯度方向的符號，可以快速地讓每個像素在允許的 $\varepsilon$ 範圍內達到最大的擾動效果。
</details>

**Q3：什麼是「黑箱攻擊」中的「代理模型（Proxy Model）」策略？**
<details>
<summary>點擊查看解答</summary>
當攻擊者無法獲取目標模型的參數時，攻擊者可以自行訓練或利用一個已知架構的「代理模型」，在該模型上產生對抗擾動，並利用模型之間的遷移性（Transferability）來攻擊目標模型。
</details>

## 來自課程原聲的重點摘要

## 來自課程原聲的重點摘要

*   **關於「白箱攻擊 (White-box Attack)」與「黑箱攻擊 (Black-box Attack)」的口頭詮釋：**
    *   **白箱攻擊**：教授比喻為「白箱」(White Box)，是因為在這個情況下，我們對模型的參數是全然知悉的。在這種情況下，計算 Gradient 是可行的，因此可以直接進行攻擊。教授特別提到，如果不希望自己的模型被白箱攻擊，最簡單的方法就是「不要將模型公開在網路上」。
    *   **黑箱攻擊**：在我們對模型參數一無所知時，我們通常會採取「訓練一個代理模型 (Proxy Network)」的方式來模擬目標模型。如果這兩個模型在訓練資料上有一定的相似度，我們可以用這個代理模型來產生攻擊樣本，進而對目標模型達到攻擊效果。
*   **關於攻擊策略與防禦技巧的補充：**
    *   **攻擊成功的關鍵**：教授指出，在進行攻擊時，我們追求的是「準確率 (Accuracy) 的下降」。準確率越低，代表攻擊越成功。
    *   **如何提升攻擊成功率**：如果不希望模型被輕易攻擊，可以採用「強化的基礎防禦 (Strong Baseline)」。
    *   **關於「盲目攻擊」的誤區**：學生常以為必須使用訓練好的模型參數才能攻擊，但教授解釋，透過輸入大量的圖像並觀察其輸出（即使不知道確切參數），依然可以訓練出一個「能模仿目標模型行為」的代理模型，這就是實現黑箱攻擊的基礎。
    *   **對抗性訓練 (Adversarial Training)**：這是一種防禦機制。其核心觀念是將對抗性攻擊過程中產生的「異常樣本」也加入到訓練集中，作為正常的訓練資料進行學習。這樣模型在面對類似的惡意攻擊時，表現會更穩健。
*   **教授特別強調的觀念與誤區：**
    *   **不要執著於特定參數**：攻擊的核心不在於「破解」參數，而在於模型對特定輸入的「反應」。只要能掌握這種反應模式，即使是黑箱，也能進行有效攻擊。
    *   **關於計算資源的考量**：教授提醒，雖然「對抗性訓練」能提升模型防禦力，但其過程極度消耗計算資源。在處理海量資料時，如何優化這一過程是當前學界的重要課題。
