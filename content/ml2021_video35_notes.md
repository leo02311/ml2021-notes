---
title: "[ML 2021 (English version)] Lecture 27: Domain Adaptation"
tags:
  - MachineLearning
  - ML2021
  - DomainAdaptation
  - AdversarialTraining
---

# 第35堂課：[ML 2021 (English version)] Lecture 27: Domain Adaptation

在機器學習的實際應用中，我們經常會遇到「訓練資料」與「測試資料」分佈不同的情況，這被稱為 **領域偏移 (Domain Shift)**。即使在某個領域訓練出準確率極高的模型，直接套用到另一個領域時，效能往往會大幅下降。

## 1. 領域偏移問題 (Domain Shift)

當訓練資料（Source Domain）與測試資料（Target Domain）的特徵分佈不一致時，即便模型在訓練集上有 99.5% 的準確率，在測試集上可能僅剩 57.5%。這種現象要求我們發展 **領域適應 (Domain Adaptation)** 技術，透過利用目標領域的資料（通常沒有標籤）來修正模型，使其適應新的分佈。

## 2. 領域適應的核心概念

我們希望模型能夠學到一種「與領域無關 (domain-invariant)」的特徵表示。

### 基本思路：忽略領域特徵
模型由兩部分組成：
1. **特徵提取器 (Feature Extractor)**：將輸入映射到特徵空間。
2. **標籤預測器 (Label Predictor)**：基於特徵進行分類。

我們的目標是讓特徵提取器學會忽略無關的差異（例如圖片顏色），使 Source 與 Target 的特徵分佈趨於一致。

### 領域對抗訓練 (Domain Adversarial Training)
為實現上述目標，引入了 **領域分類器 (Domain Classifier)**：
- **生成器 (Generator)**：即特徵提取器，目標是「騙過」領域分類器，使其無法分辨特徵來自 Source 還是 Target。
- **鑑別器 (Discriminator)**：即領域分類器，目標是準確判斷特徵屬於哪一個領域。

訓練過程是一個對抗博弈：
- 對於鑑別器：$\theta_{d}^* = \min_{\theta_d} L_d$
- 對於生成器：$\theta_{f}^* = \min_{\theta_f} (L - L_d)$（試圖最小化標籤損失 $L$ 同時最大化領域分類損失 $L_d$）。

---

## 3. 知識圖譜

```mermaid
graph TD
    "Domain Adaptation" --> "Domain Shift"
    "Domain Adaptation" --> "Domain Adversarial Training"
    "Domain Adversarial Training" --> "Feature Extractor"
    "Domain Adversarial Training" --> "Label Predictor"
    "Domain Adversarial Training" --> "Domain Classifier"
    "Feature Extractor" --> "Domain Invariant Features"
    "Domain Shift" --> "Unlabeled Target Data"
```

---

## 4. 進階議題與限制

### 決策邊界問題 (Decision Boundary)
僅僅對齊特徵分佈可能是不夠的，因為若決策邊界穿過了 Target 資料的密集區域，分類效果仍會很差。因此，需要額外的技術：
- **DIRT-T (Decision-boundary Iterative Refinement Training with a Teacher)**：透過細化決策邊界，確保目標資料遠離邊界。
- **Entropy Minimization**：鼓勵模型對目標資料輸出明確的類別（低熵），而非模糊的預測（高熵）。

### 範疇劃分
根據目標領域標籤的可用性與範圍，DA 可分為：
* **Closed Set DA**：類別完全相同。
* **Partial/Open Set DA**：類別集合部分重疊。
* **Universal DA**：更廣泛的類別適應場景。

### 其他相關技術
* **Testing Time Training (TTT)**：在測試階段利用少量的無標籤資料進行即時微調。
* **Domain Generalization**：目標是訓練一個在任何未見過的領域都能表現良好的模型，而非僅適應特定目標領域。

---

## 隨堂測驗

**Q1：在領域對抗訓練 (Domain Adversarial Training) 中，為什麼生成器 (Feature Extractor) 要最小化標籤損失並最大化領域分類損失？**
<details>
<summary>點擊查看解答</summary>
為了讓特徵提取器學會「與領域無關 (domain-invariant)」的特徵。最小化標籤損失是為了保持分類能力，而最大化領域分類損失是為了讓鑑別器無法分辨特徵來自哪個領域，進而達到領域適應的效果。
</details>

**Q2：如果只對齊了 Source 與 Target 的特徵分佈，為什麼依然可能無法達到良好的分類效果？**
<details>
<summary>點擊查看解答</summary>
因為如果學習到的「決策邊界」直接穿過了 Target 資料的密集區（高密度區域），則分類器對這些資料的判斷會產生錯誤。因此需要考慮決策邊界的細化 (如 DIRT-T 或熵最小化)。
</details>

**Q3：領域適應 (Domain Adaptation) 與領域泛化 (Domain Generalization) 的主要區別是什麼？**
<details>
<summary>點擊查看解答</summary>
領域適應是針對一個「特定的」目標領域進行調整；而領域泛化則是希望模型在訓練時學習到足夠穩健的特徵，使其不需要針對特定目標領域微調，也能在「未見過的」多個領域上表現良好。
</details>

## 來自課程原聲的重點摘要

## 來自課程原聲的重點摘要

*   **領域自適應（Domain Adaptation）的核心難點**
    *   教授用了一個生動的比喻：「訓練數據是黑白的，但測試數據是彩色的」。他解釋，雖然訓練集和測試集在「數字外型」上是一致的，但由於顏色分佈（Distribution）完全不同，即便模型在訓練集上能達到 99.5% 的高準確率，一旦應用到測試集上，效果將慘不忍睹（僅約 57%）。這凸顯了當數據分佈發生變化時，傳統模型會失效，這就是典型的「領域偏移（Domain Shift）」問題。

*   **領域自適應與遷移學習（Transfer Learning）的連結**
    *   教授指出，領域自適應可以看作遷移學習的一種。其核心邏輯是：在 Source Domain 學到的知識（對特定任務的處理能力），如何「遷移」並適應到另一個 Target Domain。他特別強調，這不僅僅是換個任務，而是要將訓練數據的特徵提取能力，與測試數據的特性進行對齊。

*   **處理領域偏移的技術細節**
    *   **資料標記問題**：在許多實際場景中，Source Domain 有標記，但 Target Domain 的資料卻是沒有標記的。
    *   **特徵提取器（Feature Extractor）與標記預測器（Label Predictor）**：教授推導了一個經典邏輯：我們需要一個模型，前半段負責提取特徵（學習「不管是什麼 Domain，貓或狗的核心特徵是一樣的」），後半段則進行分類。
    *   **領域對抗訓練（Domain Adversarial Training）**：這是課程的重點。教授將其比喻為「對抗」：Feature Extractor 的目標是提取出能騙過 Domain Classifier 的特徵（讓判別器分不出這是 Source 還是 Target），而 Domain Classifier 的目標則是盡量找出兩者差異。透過這種對抗，提取出的特徵將是領域無關（Domain-invariant）的。

*   **常見的認知誤區**
    *   教授特別警告：不要以為領域自適應很簡單，或者把它和單純的聚類（Clustering）混為一談。他強調 Domain Classifier 是一個二元分類器，而非聚類算法。
    *   學生常犯的錯是在「領域適應」時忽略了數據的標記分佈，或是沒有妥善調整模型結構。此外，教授特別提醒，若過度訓練領域自適應模型，可能會導致過擬合（Overfitting），即模型在某些特定數據上表現好，但整體泛化能力不佳。

*   **關於「測試時間訓練（Test-Time Training, TTT）」**
    *   如果面對的是極端情況（例如 Target Domain 只有極少量的未標記數據），可以使用 TTT 方法。這是一種在測試時動態微調的策略，透過「自我監督」來保持模型的穩定性。教授表示這雖然複雜，但在實際場景中非常有效。
