---
title: "[ML 2021 (English version)] Lecture 6: What to do when optimization fails? (3/4)"
tags:
  - MachineLearning
  - ML2021
  - Overfitting
  - ModelBias
  - Optimization
---

# 第07堂課：[ML 2021 (English version)] Lecture 6: What to do when optimization fails? (3/4)

本堂課由李宏毅教授講解機器學習模型在訓練與測試過程中常見的問題，以及系統化的診斷流程。當模型表現不如預期時，我們應如何判斷問題所在並進行調整。

## 1. 機器學習流程 (Framework of ML)

機器學習的核心流程可概括為三個步驟：
1. **定義模型 (Function with unknown)**：設定函數 $y = f_{\theta}(x)$，其中 $\theta$ 為待學習參數。
2. **定義損失函數 (Loss from training data)**：利用訓練資料定義損失 $L(\theta)$。
3. **優化 (Optimization)**：找到最佳參數 $\theta^* = \arg \min_{\theta} L$。

測試階段則利用學到的函數 $f_{\theta^*}(x)$ 對測試集進行預測。

---

## 2. 機器學習診斷圖譜 (General Guidance Tree)

當我們在訓練或測試集上遇到高損失 (large loss) 時，可以依據以下流程進行診斷：

```mermaid
graph TD
    "Loss on Training Data" -->|large| "Model Bias"
    "Loss on Training Data" -->|large| "Optimization Issue"
    "Loss on Training Data" -->|small| "Loss on Testing Data"
    "Loss on Testing Data" -->|small| "Great Success"
    "Loss on Testing Data" -->|large| "Overfitting"
    "Loss on Testing Data" -->|large| "Mismatch"
    "Overfitting" --> "More Data/Augmentation"
    "Overfitting" --> "Simpler Model"
    "Model Bias" --> "Complex Model"
```

---

## 3. 核心問題分析

### 3.1 模型偏差 (Model Bias)
*   **現象**：訓練資料的損失很大。
*   **成因**：模型過於簡單（限制了模型能搜尋的空間，找不到好的函數）。
*   **解法**：增加模型的靈活性 (flexibility)。
    *   增加特徵數量。
    *   使用深度學習（增加層數、神經元數量）。

### 3.2 優化問題 (Optimization Issue)
*   **現象**：訓練資料的損失很大（但模型可能已經夠複雜）。
*   **成因**：優化過程失敗，未能找到訓練資料上的最小值 $\theta^*$。
*   **診斷**：先從較淺的模型開始嘗試，若深度模型在訓練集表現反而更差，則確認為優化問題。
*   **解法**：使用更強大的優化技術（如更好的 Optimizer）。

### 3.3 過擬合 (Overfitting)
*   **現象**：訓練資料損失小，但測試資料損失大。
*   **成因**：模型在訓練集上捕捉到了雜訊，而非資料的分佈規律。
*   **解法**：
    *   增加訓練資料量。
    *   資料增強 (Data Augmentation)。
    *   模型約束 (Constrained model)：如減少參數、減少特徵、早停 (Early stopping)、正規化 (Regularization)、Dropout。

### 3.4 資料不匹配 (Mismatch)
*   **現象**：訓練與測試資料的分佈 (distribution) 不同。
*   **特徵**：增加訓練資料無效。
*   **說明**：這是真實世界常見問題，但在大多數課程作業中較少見（除 HW11 外）。

---

## 4. 模型選擇與交叉驗證 (Model Selection)

為了避免過度依賴公開測試集 (Public testing set)，造成模型在私有測試集 (Private testing set) 表現不佳，我們應：
1. 將訓練資料切分為 **Training set** 與 **Validation set**。
2. 使用 **N-fold Cross Validation**：將資料切成 $N$ 份，輪流驗證，取平均誤差，確保模型穩定性，避免單一切分帶來的偏差。

---

## 隨堂測驗

### Q1：若模型在訓練資料上的損失 (Training Loss) 很大，我們應該優先考慮什麼？
<details>
<summary>點擊查看解答</summary>
應檢查是否為「模型偏差 (Model Bias)」或是「優化問題 (Optimization Issue)」。通常可以透過嘗試更簡單的模型進行對比，若簡單模型表現更好，則可能是優化問題；若兩者表現都不好，則應增加模型複雜度。
</details>

### Q2：什麼是「過擬合 (Overfitting)」？該如何緩解？
<details>
<summary>點擊查看解答</summary>
過擬合是指模型在訓練資料上損失極小，但在測試資料上表現極差的現象。緩解方法包括：增加資料量、資料增強、限制模型複雜度（如正規化、Dropout、早停等）。
</details>

### Q3：為什麼在 Kaggle 競賽中，我們不建議直接根據 Public Leaderboard 選擇模型？
<details>
<summary>點擊查看解答</summary>
因為頻繁地根據 Public Leaderboard 調整模型，會導致模型對 Public 資料集產生過擬合，使得在 unseen 的 Private Leaderboard 上表現不佳。應透過交叉驗證 (Cross Validation) 來選擇模型。
</details>