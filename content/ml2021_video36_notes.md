---
title: "第36堂課：Introduction of Deep Reinforcement Learning (RL)"
tags:
  - MachineLearning
  - ML2021
  - ReinforcementLearning
---

# 第36堂課：Introduction of Deep Reinforcement Learning (RL)

在機器學習中，監督式學習（Supervised Learning）扮演著極為重要的角色。然而，在許多實際應用場景中，要獲得精確的「人類標註數據（Human Labels）」是相當困難且成本高昂的。例如在圍棋棋局中，人類很難告訴機器在每一個特定盤勢下「最完美的一手」應該下在哪裡（即難以標註準確的座標）。

這時候，**增強學習（Reinforcement Learning, RL）** 便展現了其優勢。在增強學習中，我們不需要給予機器每一步的標準答案，只需要讓機器與環境互動，並在最後告訴它結果是「好」還是「壞」（即給予獎勵 Reward）。本堂課將系統性地介紹深層增強學習（Deep Reinforcement Learning）的核心概念與演算法。

---

## 增強學習與知識圖譜

下圖展示了本堂課所涵蓋的增強學習核心技術與其關聯性：

```mermaid
graph TD
    A['Reinforcement Learning'] --> B['Policy Gradient']
    A --> C['Actor-Critic']
    A --> D['Sparse Reward & Reward Shaping']
    A --> E['Imitation Learning']

    B --> B1['On-policy vs Off-policy']
    B1 --> B2['Proximal Policy Optimization (PPO)']
    B --> B3['Cumulative Reward']

    C --> C1['State Value Function V('s')']
    C1 --> C2['Monte-Carlo (MC)']
    C1 --> C3['Temporal-Difference (TD)']
    C --> C4['Advantage Actor-Critic (A2C)']

    D --> D1['Curiosity-driven Exploration']

    E --> E1['Behavior Cloning']
    E --> E2['Inverse Reinforcement Learning (IRL)']
    E2 --> E3['GAN-like Framework']
```

---

## 1. 什麼是增強學習（What is RL?）

機器學習的核心本質就是「尋找一個函數（Looking for a Function）」。在增強學習的框架下，我們尋找的函數稱為 **Actor（或稱為 Policy，策略）**。

$$\text{Action} = f(\text{Observation})$$

其互動流程如下：
1. **環境（Environment）** 提供一個 **觀測值（Observation / State, $s$）** 給 Actor。
2. **Actor** 根據此觀測值輸出一個 **動作（Action, $a$）**。
3. **環境** 接收到動作後，狀態發生改變，並回傳一個 **獎勵（Reward, $r$）** 給 Actor。
4. 我們的目標是：**尋找一個能夠最大化累積預期總獎勵（Expected Total Reward）的 Actor。**

### 經典範例 1：小蜜蜂遊戲（Space Invaders）
* **Observation**: 遊戲畫面的像素像素值（Pixels）。
* **Action**: 往左移動、往右移動、開火（Fire）。
* **Reward**: 擊中外星人獲得分數（例如 $+5$），沒擊中則為 $0$。當太空船被摧毀時，遊戲終止（Termination）。

### 經典範例 2：圍棋（Learning to play Go）
* **Observation**: 19x19 的棋盤盤勢。
* **Action**: 下子在棋盤上的某個座標。
* **Reward**: 在對局結束前，大部分步驟的 Reward 皆為 $0$（極度稀疏）。若最後贏棋，Reward 為 $+1$；輸棋則為 $-1$。

---

## 2. 機器學習三步驟在增強學習中的體現

### 步驟 1：定義帶有未知參數的函數（Function with Unknown Parameters）
在深層增強學習中，Actor 通常由一個神經網路（稱為 **Policy Network**）來表示。
* **輸入**：機器的觀測值（表示為向量或矩陣，例如遊戲影像的 Pixels）。
* **輸出**：每一個 Action 的分數（經過 Softmax 後，代表採取各個動作的機率值）。採取隨機採樣（Sampling）的方式來決定最終動作。
* **本質**：這在結構上非常類似一個**分類任務（Classification Task）**。

```
Pixels (Input) ──> [ Neural Network ] ──> Action Scores (e.g., Left: 0.7, Right: 0.2, Fire: 0.1)
```

### 步驟 2：定義損失函數（Define Loss）
在一次遊戲過程（稱為一個 **Episode**）中，會產生一個軌跡（Trajectory）$\tau$：

$$\tau = \{s_1, a_1, s_2, a_2, \dots, s_T, a_T\}$$

我們定義該 Episode 的 **總回報（Total Reward / Return, $R(\tau)$）** 為：

$$R(\tau) = \sum_{t=1}^{T} r_t$$

我們的損失函數（或目標函數）即為最大化此總回報的期望值。

### 步驟 3：優化（Optimization）
增強學習與生成對抗網路（GAN）類似，其優化的最大難點在於：**Environment 與 Reward Function 對於我們而言皆是黑盒子（Black Box）**，且內部含有隨機性。我們無法直接對其進行倒傳遞（Backpropagation）求導，因此必須採用特殊的策略梯度法。

---

## 3. 策略梯度（Policy Gradient）

### 如何控制你的 Actor？
要讓 Actor 在給定觀測值 $s$ 下採取特定的動作 $\hat{a}$，我們可以定義交叉熵損失（Cross-entropy Loss） $e$。
* 若要**鼓勵**採取 $\hat{a}$：則最小化 $L = e$。
* 若要**避免**採取 $\hat{a}$：則最小化 $L = -e$。

如果有整組訓練數據，其權重為 $A_n$，我們可以定義損失函數為：

$$L = \sum_{n=1}^{N} A_n e_n \quad \Rightarrow \quad \theta^* = \arg\min_{\theta} L$$

其中 $A_n$ 代表該動作的好壞程度（Evaluation）。

### 權重 $A_n$ 的演進歷史（從 Version 0 到 Version 3）

#### Version 0：Short-sighted Version（短視版）
令 $A_n = r_n$（即當下的立即獎勵）。
* **缺點**：忽略了「延遲獎勵（Reward Delay）」。在小蜜蜂遊戲中，唯有「開火」能獲得立即分數，但「左右移動」是躲避子彈、存活下去並在未來獲得高分的關鍵。若使用 Version 0，機器將只學會瘋狂開火而不移動。

#### Version 1：Cumulated Reward（累積回報）
令 $A_t = G_t$，其中 $G_t$ 為自時間點 $t$ 開始往後累積的所有獎勵之和：

$$G_t = \sum_{n=t}^{N} r_n$$

* **邏輯**：動作 $a_t$ 的好壞，應由採取該動作後產生的所有後續 Reward 來決定。

#### Version 2：Discounted Cumulated Reward（折扣累積回報）
引入折扣因子（Discount Factor） $\gamma < 1$：

$$G'_t = \sum_{n=t}^{N} \gamma^{n-t} r_n$$

* **邏輯**：時間距離越遙遠的獎勵，與當前動作 $a_t$ 的關聯性越低（信用分配問題 Credit Assignment）。

#### Version 3：引入基準（Baseline）
如果所有的 $r_n$ 皆為正數（例如 $r_n \ge 10$），在採樣不夠充分的情況下，沒被採樣到的好動作其發生機率會相對被壓低。為了解決這個問題，我們引入一個基準 $b$：

$$A_t = G'_t - b$$

* **效果**：使 $G'_t$ 與平均期望值比較。高於平均者為正（鼓勵），低於平均者為負（懲罰）。

---

### 策略梯度的訓練演算法流程

```
1. 初始化 Actor 參數 θ^0
2. For 訓練迭代 i = 1 to T:
   a. 使用目前的 Actor 參數 θ^{i-1} 與環境進行互動
   b. 收集軌跡數據 {s_1, a_1}, {s_2, a_2}, ..., {s_N, a_N}
   c. 計算每個步驟的 Advantage 值 A_1, A_2, ..., A_N
   d. 計算損失函數 L = \sum_n A_n e_n
   e. 更新參數 θ^i \leftarrow θ^{i-1} - \eta \nabla L
```

> **重要觀念：同策略（On-policy） vs 異策略（Off-policy）**
> * **同策略（On-policy）**：收集數據的 Actor 與正在被訓練更新的 Actor 是同一個。這意味著**每一次參數更新後，先前收集的所有數據都必須丟棄，重新與環境互動收集新數據**（極度耗時）。
> * **異策略（Off-policy）**：收集數據的 Actor（通常固定一段時間）與正在訓練的 Actor 不同。透過重要性採樣（Importance Sampling），我們能重用歷史數據。
> * **PPO (Proximal Policy Optimization)**：是一種極為成功的異策略演算法，它在更新 Actor 時會加入限制，避免新舊 Policy 差異過大。

---

## 4. 執行者-評論者演算法（Actor-Critic）

在策略梯度（Policy Gradient）中，$G'_t$ 是透過蒙地卡羅法（Monte-Carlo, MC）採樣得到的。由於環境與策略具有高度隨機性，$G'_t$ 的變異數（Variance）通常極大，導致訓練過程不穩定。為了解決此問題，我們引入 **Critic（評論者）**。

### 什麼是 Critic？
Critic 的工作不是產生動作，而是評估一個給定 Actor $\theta$ 在處於狀態 $s$ 時的好壞程度。這由 **狀態價值函數（State Value Function, $V^\theta(s)$）** 來表示：
* **$V^\theta(s)$**：當使用 Actor $\theta$ 時，在看到狀態 $s$ 後，預期能獲得的折現累積獎勵期望值。

#### 如何估測 $V^\theta(s)$？
1. **蒙地卡羅法（MC-based Approach）**：讓 Actor 完整玩完遊戲，記錄狀態 $s_a$ 後，實際得到的累積獎勵 $G'_a$。接著將其視為回歸問題（Regression），強迫 $V^\theta(s_a) \approx G'_a$。
2. **時序差分法（Temporal-Difference, TD Approach）**：我們不需要等遊戲結束。根據馬可夫性質，前後相鄰的兩個狀態價值函數滿足以下關係：

$$V^\theta(s_t) = \gamma V^\theta(s_{t+1}) + r_t \quad \Rightarrow \quad V^\theta(s_t) - \gamma V^\theta(s_{t+1}) \approx r_t$$

* **MC vs TD 比較**：
  * **MC**：變異數（Variance）大值，但無偏誤（Unbiased）。
  * **TD**：變異數小，但可能因 $V^\theta$ 估測不準而引入偏誤（Biased）。

---

### 優勢執行者-評論者（Advantage Actor-Critic, A2C）

在 Version 3.5 中，我們使用價值函數 $V^\theta(s_t)$ 作為動態基準：

$$A_t = G'_t - V^\theta(s_t)$$

為了進一步降低蒙地卡羅採樣 $G'_t$ 帶來的巨大變異數，在 **A2C** 中，我們直接將 $G'_t$ 的期望值用 $r_t + \gamma V^\theta(s_{t+1})$ 進行替換，得到 **Advantage 值（優勢值）**：

$$A_t = r_t + \gamma V^\theta(s_{t+1}) - V^\theta(s_t)$$

這樣，我們在計算 $A_t$ 時只需要當下的立即獎勵 $r_t$ 以及 Critic 對當前與下一狀態的評估值，從而大幅降低了變異數。

```
                    ┌──────────────┐
                    │  State (s)   │
                    └──────┬───────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
     ┌──────────────┐            ┌──────────────┐
     │  Actor Net   │            │  Critic Net  │
     └──────┬───────┘            └──────┬───────┘
            ▼                           ▼
      Action Probabilities         Value Scalar
```
*(提示：在實際網路設計中，Actor 與 Critic 可以共享前幾層處理 State 的神經網路參數，以提升特徵提取效率。)*

---

## 5. 稀疏獎勵與獎勵塑形（Sparse Reward & Reward Shaping）

在許多複雜任務中，獎勵是非常稀疏的（例如機械手臂鎖螺絲，只有最後完全鎖進去才得到 $+1$，其餘數萬步皆為 $0$）。若沒有引導，機器隨機嘗試幾乎不可能成功。

### 獎勵塑形（Reward Shaping）
這時開發者需要主動在環境中設計「額外獎勵（Extra Rewards）」來引導 Agent。例如在 3D 射擊遊戲中，我們除了最後的勝負外，還可以設計以下引導式獎勵：
* **扣分項**：活著的每一幀微幅扣分（鼓勵快速通關）、血量減少、彈藥減少。
* **加分項**：撿起醫藥包、撿起彈藥、朝終點前進。

### 好奇心機制（Curiosity-driven Exploration）
讓 Agent 擁有「好奇心」，當它看到新穎、無法預測（但有意義）的狀態時，主動給予內部獎勵（Intrinsic Reward），促使它自發探索未知的環境地圖。

---

## 6. 無獎勵：仿冒學習（Imitation Learning / Learning from Demonstration）

在某些極端任務中，我們甚至很難去定義和寫出 Reward Function。例如，要如何用數學式寫出「安全、舒適駕駛車輛」的 Reward？這時，我們可以採用**仿冒學習（Imitation Learning）**，讓機器直接模仿人類專家的操作。

### 方法 1：行為複製（Behavior Cloning, BC）
* **做法**：收集專家駕駛的狀態-動作對群組 $\{\hat{s}_i, \hat{a}_i\}$，利用監督式學習（Supervised Learning）強迫機器在看見 $\hat{s}_i$ 時輸出 $\hat{a}_i$。
* **致命缺點（Covariate Shift）**：人類專家的採樣狀態分布是有限的。一旦機器做出些微偏差的動作，導致車輛偏離常軌，進入專家從未見過的狀態時，機器將完全不知道該如何應對，從而導致災難性的錯誤。此外，機器也會盲目複製人類多餘無用的習慣動作。

### 方法 2：逆向增強學習（Inverse Reinforcement Learning, IRL）
不同於一般 RL 先有 Reward 再找 Policy，IRL 是**先有專家的示範軌跡（Demonstrations），從中推導出一個 Reward Function，再利用該 Reward Function 去訓練我們的 Actor**。

* **核心假設**：教師（專家）的行為永遠是最完美的。
* **IRL 演算法架構**：

```
                    ┌─────────────────────────┐
                    │  Expert Demonstrations  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │   Reward Function R   │ ◄────┐
                     └───────────┬───────────┘      │
                                 │                  │ 反覆迭代
                                 ▼                  │ 調整 R
                     ┌───────────────────────┐      │
                     │  RL finds optimal π   │ ─────┘
                     └───────────────────────┘
```

* **IRL 與 GAN 的數學對偶性（Duality）**：

| 元件 \ 框架 | 生成對抗網路 (GAN) | 逆向增強學習 (IRL) |
| :--- | :--- | :--- |
| **生成器 (Generator)** | $G$ (生成假圖像) | **Actor** $\pi$ (生成遊戲軌跡) |
| **鑑別器 (Discriminator)** | $D$ (真實圖像給高分，生成圖像給低分) | **Reward Function** $R$ (專家軌跡給高分，Actor 軌跡給低分) |

---

## 隨堂測驗

### 問題 1
在策略梯度（Policy Gradient）演算法中，若所有的立即獎勵 $r_t$ 皆為正數，為什麼仍必須引入基準 $b$（Baseline）？
<details>
<summary>點擊展開解答</summary>
若沒有 Baseline，所有被採樣到的動作之權重皆為正值，其機率都會被調高。然而，未被採樣到的動作（即便它是極佳的動作）其相對機率就會被強行壓低。引入基準 $b$ 後，可以讓低於平均的動作之權重變為負值（受懲罰），進而確保動作機率的更新方向是健康且具比較性的。
</details>

---

### 問題 2
請簡述蒙地卡羅法（MC）與時序差分法（TD）在估測價值函數 $V^\theta(s)$ 時的優缺點對比。
<details>
<summary>點擊展開解答</summary>
* **蒙地卡羅法 (MC)**：
  * **優點**：無偏誤（Unbiased），因為它是根據實際遊戲結束後的真實累積 Reward 進行回歸。
  * **缺點**：必須玩完整場遊戲才能更新，且因累積了整場遊戲的隨機性，估測值的變異數（Variance）非常大。
* **時序差分法 (TD)**：
  * **優點**：不需要等待遊戲結束，每走一步即可利用 $V^\theta(s_t) = \gamma V^\theta(s_{t+1}) + r_t$ 進行更新，變異數非常小。
  * **缺點**：容易受到神經網路 $V^\theta(s_{t+1})$ 當前估計不精確的影響，因而具有偏誤（Biased）。
</details>

---

### 問題 3
行為複製（Behavior Cloning）與逆向增強學習（Inverse RL）皆屬於模仿學習。請問哪一個方法在面對「未曾見過的異常狀態（Out-of-Distribution States）」時表現更具強健性（Robustness）？為什麼？
<details>
<summary>點擊展開解答</summary>
**逆向增強學習 (IRL)** 表現更具強健性。
因為行為複製（BC）僅僅是在複製專家的動作表面，一旦偏離軌道進入陌生狀態，就會因 Covariate Shift 導致失控。而 IRL 是在學習專家行為背後的「意圖」（即 Reward Function）。只要學到了核心的 Reward Function（例如：碰撞障礙物會扣一萬分），就算機器偏離了常軌，它依然能利用強化學習演算法規劃出安全回到常軌的路徑。
</details>