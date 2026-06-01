---
title: "第32堂課：Adversarial Attack (Video 32)"
tags:
  - MachineLearning
  - ML2021
  - AdversarialAttack
  - DeepLearning
---

# 第32堂課：Adversarial Attack (Video 32)

在實際應用中，我們將訓練好的深度神經網路（Neural Network）部署至真實世界（如垃圾郵件分類、惡意軟體偵測、人臉辨識或自動駕駛系統）。然而，這些看似強大的網路，在面對「蓄意欺騙它」的輸入時，是否足夠強韌（Robust）？

本堂課將深入探討**對抗攻擊（Adversarial Attack）**的原理、數學公式推導、常見攻擊演算法、實體世界攻擊案例，以及相應的防禦機制（Defense）。

---

## 1. 知識圖譜 (Knowledge Graph)

```mermaid
graph TD
    "Adversarial Attack" --> "Attack Goal"
    "Adversarial Attack" --> "Attack Constraints"
    "Adversarial Attack" --> "Attack Methods"
    "Adversarial Attack" --> "Defense Strategies"

    "Attack Goal" --> "Non-targeted Attack"
    "Attack Goal" --> "Targeted Attack"
    "Attack Goal" --> "Adversarial Reprogramming"

    "Attack Constraints" --> "L2 Norm"
    "Attack Constraints" --> "L-infinity Norm"
    "L-infinity Norm" --> "Better for Human Perception"

    "Attack Methods" --> "White Box Attack"
    "Attack Methods" --> "Black Box Attack"

    "White Box Attack" --> "FGSM"
    "White Box Attack" --> "Iterative FGSM / PGD"
    "White Box Attack" --> "One Pixel Attack"

    "Black Box Attack" --> "Proxy Model"
    "Black Box Attack" --> "Ensemble Attack"

    "Defense Strategies" --> "Passive Defense"
    "Defense Strategies" --> "Proactive Defense"

    "Passive Defense" --> "Image Smoothing"
    "Passive Defense" --> "Image Compression"
    "Passive Defense" --> "Randomization"

    "Proactive Defense" --> "Adversarial Training"
```

---

## 2. 什麼是對抗攻擊？

對抗攻擊的核心思想在於：在原本能被正確分類的乾淨圖片（Benign Image） $x^0$ 上，加入微小且人類無法察覺的擾動（Perturbation） $\Delta x$，形成對抗樣本（Adversarial Example） $x = x^0 + \Delta x$。這個對抗樣本會使模型給出完全錯誤、甚至被指定的預測結果。

### 2.1 攻擊範例
*   **原始圖片（$x^0$）**：一隻老虎貓（Tiger Cat），ResNet-50 判讀正確，信心度 0.64。
*   **對抗樣本（$x$）**：外觀上與原圖一模一樣的老虎貓，但 ResNet-50 卻以 1.00 的極高信心度將其判定為「海星（Star Fish）」，或者以 0.98 的信心度判定為「鍵盤（Keyboard）」。
*   若將兩張圖片相減並放大 50 倍，才會看見極其微弱、類似雜訊的對抗擾動 $\Delta x$。

### 2.2 對抗雜訊 vs. 隨機雜訊
如果我們在圖片中加入「隨機雜訊」（Random Noise），模型的預測結果通常只會受到些微干擾。例如：老虎貓預測成了「波斯貓」或「虎斑貓」，這在語意上仍是合理的。然而，**對抗雜訊**是專門針對神經網路的弱點進行最佳化設計的，能引導模型做出與常理完全不符的極端錯誤判斷（例如：將貓看成「火爐防護網（Fire Screen）」）。

---

## 3. 對抗攻擊的數學公式化

為了找出最有效的對抗樣本 $x$，我們需要將攻擊過程轉換為一個最佳化問題。

### 3.1 基礎架構
設原始輸入為 $x^0$，神經網路模型為 $f(\cdot)$，其參數 $\theta$ 在攻擊階段是**固定不變**的。
*   原始預測輸出：$y^0 = f(x^0)$
*   真實標籤（Ground Truth）：$\hat{y}$（以 One-hot 向量表示）
*   對抗樣本 $x$ 的預測輸出：$y = f(x)$

### 3.2 損失函數 $L(x)$ 的設計

根據攻擊者的目的，損失函數可分為兩種：

1.  **非目標攻擊（Non-targeted Attack）**
    攻擊者的目標只要模型預測錯誤即可，不管預測成什麼。
    $$L(x) = -e(y, \hat{y})$$
    其中 $e(y, \hat{y})$ 通常為交叉熵損失（Cross-Entropy Loss）。我們希望**最小化** $L(x)$，這等同於**最大化**預測值 $y$ 與真實標籤 $\hat{y}$ 之間的距離。

2.  **目標攻擊（Targeted Attack）**
    攻擊者希望模型將輸入錯誤分類為某個特定的目標類別 $y^{\text{target}}$（例如：將貓判定為「海星」）。
    $$L(x) = -e(y, \hat{y}) + e(y, y^{\text{target}})$$
    我們希望最小化此損失函數，即一方面拉大與真實標籤 $\hat{y}$ 的距離，另一方面縮短與目標標籤 $y^{\text{target}}$ 的距離。

### 3.3 人類感知限制（Constraints）
為了確保擾動不被人類肉眼察覺，我們必須限制原始影像 $x^0$ 與對抗影像 $x$ 之間的距離：
$$d(x^0, x) \le \epsilon$$

常用的距離度量方式（Norm）包括：

*   **$L_2$-norm**：
    $$d(x^0, x) = \|\Delta x\|_2 = \sqrt{\sum_{i} (\Delta x_i)^2}$$
    計算所有像素改動幅度的平方和再開根號。
*   **$L_\infty$-norm**：
    $$d(x^0, x) = \|\Delta x\|_\infty = \max_{i} |\Delta x_i|$$
    找出所有像素中，改動幅度最大的那一個像素的值。

#### 為什麼在對抗攻擊中 $L_\infty$ 優於 $L_2$？
人類的視覺系統對於「單一像素的劇烈變化」非常敏感，但對於「多數像素的極微小變化」卻毫無感覺。
*   若採用 $L_2$-norm 限制，最佳化過程可能會選擇「大幅度修改某幾個特定像素」，這在視覺上會產生顯眼的瑕疵斑點。
*   若採用 $L_\infty$-norm 限制，則確保了**沒有任何一個像素的修改量會超過 $\epsilon$**。即使每個像素都改變了一點點，整體視覺上依舊完全看不出差異。因此，在圖像對抗攻擊中，$L_\infty$-norm 是更符合人類視覺感知的約束條件。

---

## 4. 攻擊演算法與實現方式

對抗攻擊的最佳化與模型訓練非常相似，差別在於**訓練時更新模型參數 $\theta$，而攻擊時更新輸入 $x$**。

### 4.1 梯度下降法（Gradient Descent）
我們從原始影像 $x^0$ 出發，反覆沿著梯度的反方向（使損失函數減小的方向）更新輸入：
$$x^t \leftarrow x^{t-1} - \eta g$$
其中，梯度向量 $g$ 定義為損失函數對輸入 $x$ 的偏微分：
$$g = \nabla_x L(x) \big|_{x=x^{t-1}}$$

#### 邊界投影（Projection）
在每次更新後，若發現 $d(x^0, x^t) > \epsilon$，我們必須將其投影（Project/Fix）回限制邊界內：
$$x^t \leftarrow \text{fix}(x^t)$$
以 $L_\infty$ 為例，若某像素值超出 $[x_i^0 - \epsilon, x_i^0 + \epsilon]$ 的範圍，就直接將其截斷（Clip）至邊界值。

---

### 4.2 快速梯度標簽法 (FGSM)
由 Ian Goodfellow 等人於 2014 年提出。**FGSM (Fast Gradient Sign Method)** 是一種「一擊必殺」的單步攻擊法。它不進行多次反覆迭代，而是直接計算一次梯度，並沿著梯度方向跨出最大容許的一步：

$$x^1 \leftarrow x^0 - \epsilon \cdot \text{sign}(g)$$

其中 $\text{sign}(g)$ 取梯度的正負號（輸出為 $+1$ 或 $-1$）。這使得新生成的影像在每個維度上都精確地移動了 $\epsilon$ 的距離，直接推向 $L_\infty$ 限制邊界的角落。

---

### 4.3 迭代型 FGSM (Iterative FGSM / PGD)
FGSM 雖然速度快，但攻擊強度有限。**Iterative FGSM (I-FGSM)**（在許多文獻中與 PGD - Projected Gradient Descent 概念相同）透過多步更新並配合每次的邊界投影，來尋找更強的對抗樣本：

$$x^t \leftarrow \text{fix}\left( x^{t-1} - \eta \cdot \text{sign}(g) \right)$$

這種多步微調的攻擊方式，其成功率顯著高於單步的 FGSM。

---

## 5. 白箱攻擊 vs. 黑箱攻擊

根據攻擊者掌握的模型資訊多寡，對抗攻擊可分為兩大類：

| 比較維度 | 白箱攻擊 (White Box) | 黑箱攻擊 (Black Box) |
| :--- | :--- | :--- |
| **資訊掌握度** | 知道目標網路的所有參數 $\theta$ 與架構。 | 無法得知目標網路的參數與架構（例如線上 API）。 |
| **梯度獲取** | 可直接對目標網路計算 $\nabla_x L(x)$。 | 無法直接計算目標網路的梯度。 |
| **可行性** | 容易實現且成功率極高。 | 理論上較難，但實務上完全可行。 |

### 5.1 黑箱攻擊的實現：替代模型與轉移性 (Transferability)
當我們無法取得黑箱模型的參數時，該如何發動攻擊？這要歸功於對抗樣本的**轉移性（Transferability）**：在 A 模型上有效的對抗樣本，極大機率也能成功騙過 B 模型。

1.  **訓練替代模型（Proxy Model）**：利用與目標黑箱模型相似的訓練數據，自己訓練一個本機的替代模型（如 ResNet）。
2.  **生成對抗樣本**：在本機的替代模型上進行白箱攻擊，生成對抗樣本。
3.  **實施攻擊**：將此對抗樣本直接送入黑箱模型，便有很高的機率使其預測錯誤。

### 5.2 聯防攻擊 (Ensemble Attack)
為了進一步提升黑箱攻擊的成功率，攻擊者可以同時對多個不同的本機替代模型（例如：VGG-16, ResNet-50, GoogLeNet）進行白箱攻擊，尋找能**同時欺騙所有替代模型**的對抗樣本。實驗證實，這種方法產生的對抗樣本具有極強的普適性，幾乎能 100% 攻破未知的黑箱模型。

---

## 6. 各式各樣的對抗攻擊變體

對抗攻擊的研究領域非常廣泛，以下列舉幾種經典且有趣的變體：

### 6.1 單像素攻擊 (One Pixel Attack)
由 Su 等人於 2019 年提出。研究發現，在許多影像分類模型中，**僅僅修改一個像素點的值**，就能徹底顛覆模型的判斷。例如：將茶壺（Teapot）圖片中的一個點改成紅色，模型就會以極高信心度判定為「搖桿（Joystick）」。這種攻擊不依賴梯度下降（因為單像素不連續且梯度不可導），而是使用**差分進化演算法（Differential Evolution）**進行隨機搜尋。

### 6.2 通用對抗攻擊 (Universal Adversarial Attack)
一般的對抗擾動是「針對單一圖片」量身打造的。而通用對抗攻擊則旨在尋找一個**單一的擾動圖案 $\Delta x$**，不論將它疊加在任何一張圖片上，都能導致模型分類出錯。這代表該擾動代表了神經網路空間中某種系統性的幾何缺陷。

### 6.3 對抗重編程 (Adversarial Reprogramming)
這是一種極具創意的攻擊方式。攻擊者在圖像周圍加上特殊的對抗雜訊框，可以「重新編程」一個已經訓練好的模型。例如：將一個原本用來做 ImageNet 千類圖像分類的模型，強行改造為「計算圖片中白色方塊數量」的模型。

### 6.4 模型後門攻擊 (Backdoor / Trojan in Model)
這種攻擊發生在**訓練階段**。攻擊者在訓練資料中混入「帶有特殊印記（Trigger）」的樣本。例如：只要圖片右下角有一個黃色小方塊，其標籤就被改為「狗」。模型訓練完成後，在一般乾淨圖片上表現完全正常，但只要測試圖片中出現黃色小方塊，模型就會無條件將其歸類為「狗」。因此，在使用來源不明的開源資料集時必須格外小心。

---

## 7. 實體世界中的對抗攻擊

對抗攻擊不僅僅存在於數位虛擬世界中。當對抗影像被**列印出來**、透過相機鏡頭重新捕捉後，攻擊效果依然存在。

*   **對抗眼鏡（Adversarial Glasses）**：
    研究人員設計並列印出一副特殊花紋的眼鏡框架。當一般男性戴上這副眼鏡時，人臉辨識系統會將他誤認為知名女星（例如 Milla Jovovich）。
*   **交通號誌貼紙攻擊**：
    在「STOP」標誌上黏貼精心設計的小黑白貼紙（看似無意的塗鴉或髒污），自動駕駛車載相機辨識時，會將其誤判為「限速 45 英里」或「右轉標誌」，造成嚴重的安全隱患。
*   **速限牌修改**：
    僅在「Speed Limit 35」的 3 字中間貼上一小段黑膠帶，人眼看仍是 35，但 McAfee 實驗室測試發現，特斯拉的自動駕駛系統會將其辨識為 「85 哩」，因而自動加速。

---

## 8. 防禦機制 (Defense)

面對無孔不入的對抗攻擊，學術界與工業界發展出兩大防禦思維：被動防禦與主動防禦。

### 8.1 被動防禦 (Passive Defense)
被動防禦**不修改模型本身的參數**，而是在輸入影像送入網路前，先經過一層預處理（Filter），將可能存在的對抗擾動「過濾」或「破壞」掉。

1.  **影像平滑化（Smoothing/Filtering）**：
    使用高斯模糊（Gaussian Blur）或雙邊濾波（Bilateral Filter）。由於對抗擾動多屬於高頻雜訊，平滑化能有效破壞這些精密設計的微弱訊號。
    *   *副作用*：會導致正常乾淨圖片的分類信心度稍微下降。
2.  **影像壓縮（Image Compression）**：
    例如 JPEG 壓縮。在壓縮與解壓縮的過程中，會捨棄非視覺主導的高頻細節，從而順便消滅對抗干擾。
3.  **生成器重構（Generative Reconstruction）**：
    將圖片輸入 GAN 的 Generator（或 Autoencoder），重新生成一張對應的乾淨圖片，藉此過濾掉人肉眼看不見的像素微調。
4.  **隨機化機制（Randomization）**：
    在影像輸入前，對其進行隨機比例的縮放（Random Resizing）與隨機位置的填充（Random Padding）。由於對抗攻擊通常極度依賴特定的像素對齊，一旦影像位置發生隨機偏移，精心設計的攻擊訊號就會立刻失效。

### 8.2 主動防禦：對抗訓練 (Proactive Defense - Adversarial Training)
主動防禦的目標是**直接訓練出一個本質上就強韌（Robust）的模型**。其中最著名且有效的方法是**對抗訓練（Adversarial Training）**，這可以被看作是一種極限的數據增強（Data Augmentation）技術。

#### 對抗訓練演算法步驟：
對於每一個訓練回合（Epoch）：
1.  給定當前的訓練資料集 $\mathcal{X} = \{(x^1, \hat{y}^1), (x^2, \hat{y}^2), \dots\}$。
2.  針對每一個樣本 $x^n$，使用攻擊演算法（例如 PGD 攻擊 10 步）找出其對應的對抗樣本 $\tilde{x}^n$。
3.  將這些對抗樣本收集起來，組成新的對抗訓練集 $\mathcal{X}' = \{(\tilde{x}^1, \hat{y}^1), (\tilde{x}^2, \hat{y}^2), \dots\}$。
4.  將原始資料 $\mathcal{X}$ 與對抗資料 $\mathcal{X}'$ 混合，用來更新模型的參數 $\theta$。

#### 對抗訓練的痛點：
*   **運算資源消耗極大**：在訓練的每一步都要動態運行 PGD 攻擊，這相當於訓練時間翻了數倍（近幾年有「Adversarial Training for Free」等加速演算法被提出）。
*   **過擬合於特定攻擊**：用演算法 A 訓練出來的防禦模型，面對全新的攻擊演算法 B 時，防禦力可能會大幅下降。
*   **攻防持續演進**：這是一場永無止境的軍備競賽（Cat-and-Mouse Game），防禦與攻擊技術都在持續進化。

---

## 9. 隨堂測驗

### 測驗 1：關於 $L_2$-norm 與 $L_\infty$-norm 约束條件的比較，下列敘述何者正確？
A) $L_2$-norm 限制能完全保證每一個像素的修改量都小於 $\epsilon$，因此比 $L_\infty$-norm 更符合人類視覺特徵。  
B) $L_\infty$-norm 限制最大單一像素的修改量，能防止產生明顯的單點視覺瑕疵，因此比 $L_2$-norm 更適合作為圖像攻擊的約束。  
C) 兩種 Norm 在實務上對於人類感知的限制效果完全相同，沒有任何差別。  
D) $L_2$-norm 適用於黑箱攻擊，而 $L_\infty$-norm 僅適用於白箱攻擊。

<details>
<summary>點擊展開解答</summary>
**正確答案：B**  
解析：人類對單點大範圍的修改非常敏感。$L_\infty$-norm 限制了所有像素的最大修改上限，使得所有像素都只被微調一點點，從而達到「肉眼不可察覺」的效果；而 $L_2$-norm 雖然限制了整體的改動總和，但可能允許集中在某一個像素進行大改動，這會產生人眼容易察覺的突兀斑點。
</details>

---

### 測驗 2：攻擊者在完全不知道目標網路（如某線上辨識 API）的架構與參數時，最常用什麼方法成功實施攻擊？
A) 無限次隨機微調圖片像素，直到 API 輸出錯誤。  
B) 在本地端訓練一個或多個「替代模型（Proxy Model）」，在其上進行白箱攻擊產生對抗樣本，再利用「轉移性（Transferability）」攻破 API。  
C) 利用數學公式直接推導出該線上 API 的精確權重參數。  
D) 透過影像平滑化（Smoothing）直接將 API 的防禦系統關閉。

<details>
<summary>點幕展開解答</summary>
**正確答案：B**  
解析：這即是黑箱攻擊（Black Box Attack）的標準手法。利用對抗樣本在不同深度網路之間的轉移性，透過攻擊本地替代模型，就能高機率騙過雲端黑箱模型。
</details>

---

### 測驗 3：關於「被動防禦（Passive Defense）」中的「隨機化機制（Randomization）」，其能成功抵禦對抗攻擊的主要原因是什麼？
A) 隨機化會將圖片的解析度降到極低，讓攻擊雜訊直接消失。  
B) 對抗攻擊非常依賴「像素級別的精確對齊」，隨機的尺寸縮放與位置填充會破壞這種精細設計的幾何對齊。  
C) 隨機化會在圖片中加入更多隨機雜訊，以毒攻毒。  
D) 隨機化會自動將模型轉換為生成對抗網路（GAN）。

<details>
<summary>點擊展開解答</summary>
**正確答案：B**  
解析：對抗樣本的雜訊是針對特定神經網路層層傳遞後的特徵圖（Feature Map）精確計算出來的。一旦對輸入影像進行隨機縮放或位移，其梯度方向與特徵對齊便會錯位，使對抗雜訊失去其原本設計的破壞力。
</details>