---
title: "[ML 2021 (English version)] Lecture 8: Classification (Short Version)"
tags:
  - MachineLearning
  - ML2021
  - ModelBias
  - Overfitting
  - Optimization
---

# 第09堂課：[ML 2021 (English version)] Lecture 8: Classification (Short Version)

本堂課由李宏毅教授講解機器學習在開發流程中遇到錯誤時的診斷邏輯與解決方案。透過一套系統化的「一般指引 (General Guide)」，我們可以判斷模型的效能瓶頸源自於哪一個環節。

## 機器學習流程框架 (Framework of ML)

機器學習的核心步驟可以分為三階段：
1.  **步驟一 (Step 1)**：建立模型函數 $y = f_{\theta}(x)$，其中 $\theta$ 為未知參數。
2.  **步驟二 (Step 2)**：基於訓練資料定義損失函數 $L(\theta)$。
3.  **步驟三 (Step 3)**：進行優化，尋找最佳參數 $\theta^* = \arg \min_{\theta} L$。

訓練完成後，我們將模型應用於測試資料進行預測。

---

## 機器學習錯誤診斷圖

若模型在訓練資料或測試資料上表現不佳，可根據下方的邏輯樹進行診斷：

```mermaid
graph TD
    "Loss on Training Data" -->|large| "Model Bias"
    "Loss on Training Data" -->|large| "Optimization Issue"
    "Loss on Training Data" -->|small| "Loss on Testing Data"
    "Loss on Testing Data" -->|large| "Overfitting"
    "Loss on Testing Data" -->|large| "Mismatch"
    "Loss on Testing Data" -->|small| "Success"
```

### 1. 訓練損失過大 (Large Training Loss)
如果模型在訓練集上的表現就很差（損失很大），通常有兩種可能：

*   **模型偏差 (Model Bias)**：模型太簡單，無法擬合訓練資料。
    *   *解決方法*：增加模型複雜度（如：增加特徵、增加神經元或層數）。
*   **優化問題 (Optimization Issue)**：模型結構足夠，但優化技術不足，找不到最佳參數。
    *   *檢查方法*：比較不同深度的網路，若更深的網路在訓練集上損失反而更大，則為優化問題。
    *   *解決方法*：使用更強大的優化技術（將於後續課程討論）。

### 2. 測試損失過大 (Large Testing Loss)
如果模型在訓練集表現很好，但測試集表現很差，這代表模型泛化能力不足：

*   **過擬合 (Overfitting)**：模型在訓練集上「背誦」了資料，對未見過的資料泛化能力差。
    *   *解決方法*：增加訓練資料量、資料增強 (Data Augmentation)、縮小模型複雜度、使用正則化 (Regularization) 或 Dropout。
*   **不匹配 (Mismatch)**：訓練與測試資料的分佈 (Distribution) 不一致。
    *   *解決方法*：這通常無法透過單純增加訓練資料解決，需了解資料生成機制（例如課程作業 11 的案例）。

---

## 模型評估與驗證

為了做出正確的「模型選擇 (Model Selection)」，我們必須將訓練資料切分為 **訓練集 (Training Set)** 與 **驗證集 (Validation Set)**。

*   **避免只看公開排行榜 (Public Leaderboard)**：若根據公開測試集分數不斷調整參數，會導致模型針對該測試集過擬合，導致在私有測試集 (Private Test Set) 上表現不佳。
*   **交叉驗證 (Cross Validation)**：使用 N-fold Cross Validation 可以更穩定地評估模型效能，減少隨機抽樣導致的誤差。

---

## 隨堂測驗

<details>
<summary><b>問題 1：當模型在訓練集上的損失 (Training Loss) 很大時，可以採取什麼樣的策略來改善？</b></summary>
解答：這通常代表模型偏差 (Model Bias) 過大，建議增加模型的複雜度（例如：增加更多的特徵、更多的神經元或層數）或是檢查是否有優化問題 (Optimization Issue) 並採用更強大的優化演算法。
</details>

<details>
<summary><b>問題 2：什麼是資料增強 (Data Augmentation)？它主要用於解決什麼問題？</b></summary>
解答：資料增強是透過對訓練影像進行旋轉、翻轉或剪裁等操作，人為地增加訓練資料量。它主要用於解決「過擬合 (Overfitting)」問題，因為更多的訓練樣本有助於模型學習到更具泛化能力的特徵。
</details>

<details>
<summary><b>問題 3：為什麼在 Kaggle 等競賽中，僅僅追求公開排行榜 (Public Leaderboard) 的高分是危險的？</b></summary>
解答：這會造成模型對「公開測試集」的過擬合。若模型針對公開資料進行了過度調整，則在未知的「私有測試集」上表現往往會大幅下降。正確做法是使用驗證集 (Validation Set) 或交叉驗證來進行模型選擇。
</details>