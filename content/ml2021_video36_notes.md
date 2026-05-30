---
title: "[ML 2021 (English version)] Lecture 28: Introduction of Reinforcement Learning (RL) (1/5)"
tags:
  - MachineLearning
  - ML2021
  - ReinforcementLearning
---

# 第36堂課：[ML 2021 (English version)] Lecture 28: Introduction of Reinforcement Learning (RL) (1/5)

這堂課由李宏毅教授介紹深度強化學習（Deep Reinforcement Learning, RL）的基礎概念，並將強化學習拆解為機器學習的三個基本步驟：定義函式、定義損失函數（Loss Function）、與優化。

## 1. 什麼是強化學習？

強化學習與監督式學習（Supervised Learning）不同。在監督式學習中，機器根據輸入（如貓的圖片）學習對應的標籤（如 "Cat"）。然而，在某些任務中（例如下圍棋），獲得正確標籤非常困難，機器只能在事後得知結果是好是壞（Reward）。

### 機器學習的三大步驟：
1. **定義函式 (Function)**：這是一個 Actor，輸入為觀察（Observation），輸出為動作（Action）。
2. **定義損失函數 (Loss)**：在 RL 中，這表現為獎勵（Reward），目標是最大化累計獎勵（Total Reward / Return）。
3. **優化 (Optimization)**：找出最佳的 Actor，使其行為能最大化長期獎勵。

```mermaid
graph TD
    "Environment" -->|Observation| "Actor"
    "Actor" -->|Action| "Environment"
    "Environment" -->|Reward| "Actor"
    "Actor" -- "Find policy maximizing total reward" --> "Actor"
```

## 2. Policy Gradient（策略梯度）

Policy Gradient 的核心概念是將決策視為分類問題。
- **輸入**：環境的 Observation（向量或矩陣）。
- **輸出**：不同動作的機率值。
- **訓練**：透過調整參數 $\theta$ 來決定採取某個動作的機率。

### 訓練技巧：
*   **Version 0 (Short-sighted)**：直接使用當下得到的 $r_t$ 作為權重 $A_t$。這會導致模型只學會關注短期獎勵。
*   **Version 1 (Cumulative Reward)**：使用從時刻 $t$ 開始到結束的累計獎勵 $G_t = \sum_{n=t}^{N} r_n$ 作為權重。
*   **Version 2 (Discount Factor)**：引入折扣因子 $\gamma < 1$，計算 $G_t' = \sum_{n=t}^{N} \gamma^{n-t} r_n$。
*   **Version 3 (Baseline)**：減去一個 Baseline $b$ 來調整獎勵的相對性，使優勢函數 $A_t$ 有正有負。

## 3. Actor-Critic

Actor-Critic 方法結合了 Actor 與 Critic 兩個模型：
- **Actor**：負責決策（採取動作）。
- **Critic**：負責評估狀態價值 $V^\theta(s)$。給定 Actor $\theta$，當看到狀態 $s$ 時，預期未來能得到的累計獎勵。

### 估計價值函數 $V^\theta(s)$ 的兩種方法：
1. **Monte-Carlo (MC)**：讓 Actor 與環境互動到結束，直接用該次實際獲得的 $G_t'$ 來估計。
2. **Temporal-difference (TD)**：利用 $V^\theta(s_t) = \gamma V^\theta(s_{t+1}) + r_t$ 的關係進行更新。

**Advantage Actor-Critic (A3C/A2C)**：計算優勢函數 $A_t = r_t + V^\theta(s_{t+1}) - V^\theta(s_t)$，這能更準確地評估一個動作是否比平均表現更好。

## 4. 獎勵塑造 (Reward Shaping) 與模仿學習

*   **Reward Shaping**：在稀疏獎勵（Sparse Reward）任務中，由於大部分動作的獎勵為 0，開發者需要定義額外的輔助獎勵來引導模型。例如在 VizDoom 遊戲中，給予「移動」或「拾取道具」額外的獎勵。
*   **Imitation Learning (模仿學習)**：當 Reward 很難定義時，直接模仿專家的行為。
    *   **Behavior Cloning**：將人類專家的行為視為監督式學習的資料集。
    *   **Inverse Reinforcement Learning (IRL)**：透過專家的行為回推其隱含的「獎勵函數」，再利用該函數訓練 Actor。這與 GAN（生成對抗網路）的原理類似，Actor 扮演生成器，而 Reward Function 扮演判別器。

---

## 隨堂測驗

**Q1: 在強化學習中，為什麼要使用 Discount Factor $\gamma$？**
<details>
<summary>點擊查看解答</summary>
因為未來的獎勵比眼前的獎勵更有不確定性，或者為了讓模型更重視近期獎勵，通過 $\gamma < 1$ 的衰減，能有效處理無窮序列並收斂累計獎勵的計算。
</details>

**Q2: 什麼是 Actor-Critic 模型中的 Critic 角色？**
<details>
<summary>點擊查看解答</summary>
Critic 是一個評估器，輸入為當前狀態（Observation），輸出為一個數值（Scalar），代表在當前策略（Actor）下，預期未來能獲得的總獎勵（Value Function $V^\theta(s)$）。
</details>

**Q3: 為什麼在使用行為模仿（Behavior Cloning）時，單純學習人類行為有時會失敗？**
<details>
<summary>點擊查看解答</summary>
因為專家示範的觀察範圍有限（Limited Observation），若機器遇到未曾見過的狀態，便無法做出正確反應。此外，模型可能會盲目模仿無關的行為（Irrelevant actions）。
</details>

## 來自課程原聲的重點摘要

## 來自課程原聲的重點摘要

### 課程核心概念與設計初衷
* **深度強化學習（RL）的廣泛應用**：教授舉例 AlphaGo 是 RL 最知名的應用，並強調 RL 技術範疇廣大，單堂課程僅能提供基礎理解，旨在引發同學興趣與自主研究。
* **教學策略**：為了避免過度深奧的數學推導讓學生產生畏難情緒，課程設計著重於讓學生培養「對 RL 的直觀理解」，即能理解機器為何透過與環境互動來進行學習。

### 強化學習的關鍵機制
* **動作與環境的互動（Action & Environment）**：
    * 強化學習的目標是尋找一個能與環境互動的「函式（Actor）」。
    * 機器透過「觀察環境狀態（Observation）」執行「動作（Action）」，並根據環境的反饋獲取「獎勵（Reward）」。
    * **核心目標**：透過最大化累積獎勵，讓 Actor 學習到在正確的狀態下，採取最合適的行動。
* **生動比喻（Space Invader 遊戲）**：
    * 教授以《Space Invaders》為例，將 Actor 視為遊戲中的太空船，將其向左、向右、開火的決策視為 Action；獎勵則是擊中敵人的分數，懲罰則是太空船被摧毀。透過遊戲規則，讓機器自主探索如何獲得高分。

### 強化學習與機器學習的關係
* **三大步驟的一致性**：強化學習其實符合傳統機器學習的三步驟：
    1. 定義含未知參數的函式（Actor）。
    2. 定義損失函數（Loss Function）。
    3. 最佳化未知參數。
* **為何 RL 較為複雜？**：
    * **隨機性（Randomness）**：在訓練階段，隨機因素會影響結果；測試階段亦然。
    * **環境的不可測性（Black Box）**：環境通常是「黑盒子」，我們無法明確預知採取某個行動後，下一個狀態為何，這使得 RL 與一般的監督式學習（Supervised Learning）有所差異。
* **損失函數（Loss）的定義**：在 RL 中，將「負的累積獎勵」設為損失函數。訓練目標就是透過梯度下降法將此損失降至最低，即最大化累積獎勵。
