---
title: "第06堂課：【機器學習2021】類神經網路訓練不起來怎麼辦 (三)：自動調整學習速率 (Learning Rate)"
tags:
  - MachineLearning
  - ML2021
  - Optimizer
  - AdaptiveLearningRate
  - Adagrad
  - RMSProp
  - Adam
  - LearningRateScheduling
  - WarmUp
  - LearningRateDecay
---

# 【機器學習 2021】第06堂課：【機器學習2021】類神經網路訓練不起來怎麼辦 (三)：自動調整學習速率 (Learning Rate)

## 課程引言：梯度下降 (Gradient Descent) 的挑戰

上週我們討論了 critical point 的問題。本堂課將深入探討，在訓練類神經網路時，critical point 其實不一定是最大的障礙，而更常見的問題是如何有效地運用梯度下降。

### Critical Point 並非唯一魔王

*   **常見現象：** 訓練過程中，Loss 曲線初期快速下降，隨後停滯不前，不再下降。多數人會猜測這是因為到達了 critical point (梯度為零)，導致參數無法更新。
*   **真實情況：** 其實，當 Loss 停滯時，梯度 (gradient) 的大小（norm）往往並沒有變得非常小。
    *   **原因分析：** 參數可能在 Error Surface 的峽谷兩側（坡度陡峭）來回震盪，導致 Loss 無法進一步下降，但梯度值依然很大。
    *   **誤區：** 不要輕易斷定卡在 Local Minima 或 Saddle Point。在作業 2-2 中，我們也會要求同學確認梯度 norm，以釐清停滯原因。
*   **實驗證明：** 實際上，要讓訓練過程走到真正接近 critical point 的位置，使用一般的梯度下降是困難的。許多研究顯示，在梯度仍然很大的情況下，訓練就已經停止了。這表示在優化過程中，我們面對的「魔王」往往不是 critical point，而是其他因素。

### 傳統梯度下降 (Vanilla Gradient Descent) 的困境

*   **簡單範例：** 即使是一個簡單的凸 (convex) Error Surface，其等高線呈橢圓形，在某一方向上坡度平坦（梯度小），另一方向上坡度陡峭（梯度大）。
*   **固定學習率 (Fixed Learning Rate) 的問題：**
    *   **學習率過大：** 參數會在峽谷兩側劇烈震盪，無法進入谷底，Loss 不再下降。
    *   **學習率過小：** 雖然能避免震盪，但在陡峭方向進展緩慢，在平坦方向幾乎停滯，導致訓練時間過長，無法到達終點。
*   **結論：** 傳統的梯度下降在處理這種簡單的 Error Surface 時都力不從心，更遑論複雜的深度網路。因此，我們需要更進階的梯度下降方法。

## 自適應學習率 (Adaptive Learning Rate)

為了克服固定學習率的限制，我們需要為每一個參數客製化其學習率。

### 核心思想

*   **客製化原則：**
    *   在**平坦方向**（梯度值小）上，希望給予**較大的學習率**，以加速前進。
    *   在**陡峭方向**（梯度值大）上，希望給予**較小的學習率**，以避免震盪或越過谷底。
*   **更新公式：** 原始梯度下降更新公式為 $\theta_i^{t+1} = $\theta_i$^t - $\eta$ \cdot g_i^t$。
    *   自適應學習率將其修改為：$\theta_i^{t+1} = $\theta_i$^t - \frac{$\eta$}{$\sigma_i^t$} \cdot g_i^t$。
    *   其中，$\sigma_i^t$ 是一個與參數 $i$ 和時間 $t$ 相關的項，用來自動調整學習率。

### Adagrad

Adagrad 是一種常見的自適應學習率方法，其 $\sigma_i^t$ 的計算方式是基於過去所有梯度平方的 Root Mean Square。

*   **$\sigma_i^t$ 計算方式：**
    $$ \sigma_i^t = \sqrt{\frac{1}{t+1} \sum_{k=0}^{t} (g_i^k)^2} $$
    其中，$g_i^k$ 是在第 $k$ 次迭代時參數 $\theta_i$ 對 Loss 的梯度。
*   **優點：**
    *   若某方向梯度值持續較小，$\sigma_i^t$ 會累積較小的值，導致學習率 $\eta / $\sigma_i$^t$ 變大，加速在平坦方向的前進。
    *   若某方向梯度值持續較大，$\sigma_i^t$ 會累積較大的值，導致學習率 $\eta / $\sigma_i$^t$ 變小，減緩在陡峭方向的步長。
*   **缺點：**
    *   由於 $\sum (g_i^k)^2$ 是一個累加值，$\sigma_i^t$ 會隨著時間 $t$ 不斷增大。
    *   這導致學習率 $\eta / $\sigma_i$^t$ 最終會趨近於零，可能使得訓練過早停止。
    *   在峽谷地形中，可能因為某軸向在初期梯度很大，累積了很大的 $\sigma$，之後即便梯度變小，也難以再有大步長。反之，若某一軸向初期梯度很小，累積了很小的 $\sigma$，後期一旦遇到稍大的梯度，就可能因為學習率過大而「噴」出去。

### RMSProp

![Momentum vs RMSProp Optimization Trajectory](../assets/vid06_opt.gif)

RMSProp 是一種解決 Adagrad 缺點的改良方法，其傳奇之處在於它最早由 Hinton 在 Coursera 課程中提出，卻沒有正式的論文，因此常常需要引用課程影片。

*   **$\sigma_i^t$ 計算方式：** RMSProp 採用指數加權移動平均 (Exponentially Weighted Moving Average) 的方式來計算平方梯度的平均值，使其更關注近期的梯度。
    $$ (\sigma_i^t)^2 = \alpha \cdot (\sigma_i^{t-1})^2 + (1-\alpha) \cdot (g_i^t)^2 $$
    $$ \sigma_i^t = \sqrt{\alpha \cdot (\sigma_i^{t-1})^2 + (1-\alpha) \cdot (g_i^t)^2} $$
    *   其中，$\alpha$ 是一個超參數 (hyperparameter)，用於控制新梯度 ($g_i^t$) 和舊梯度累積值 ($($\sigma_i$^{t-1})^2$) 的權重。
        *   $\alpha$ 趨近於 0：更看重當前梯度 ($g_i^t$)。
        *   $\alpha$ 趨近於 1：更看重過去累積的梯度。
*   **優點：**
    *   透過 $\alpha$ 的調整，RMSProp 能夠動態地調整 $\sigma_i^t$，使其不會無限累積。
    *   這使得學習率能夠更靈活地響應 Error Surface 的變化（例如：從平坦變陡峭，或從陡峭變平坦），避免 Adagrad 的學習率過早趨零或後期因累積問題而「噴」出去。

### Adam (Adaptive Moment Estimation)

Adam 是目前最廣泛使用的優化器之一，它結合了 RMSProp 和 Momentum 的優點。

*   **組成：** Adam = RMSProp + Momentum。
    *   **Momentum (動量)：** 考慮過去所有梯度的加總方向來更新參數，包含梯度的方向 and 正負號，使得更新方向更穩定。
    *   **RMSProp：** 考慮過去所有**平方梯度**的指數加權移動平均，調整學習率的大小，只考慮梯度的大小，不考慮方向。
*   **應用：** Adam 的演算法細節較為複雜，但 PyTorch 等深度學習框架都已內建，並提供良好的預設超參數。通常直接使用預設值就能獲得不錯的結果。
*   **區別：** Momentum 和 RMSProp 的 $\sigma$ 項雖然都考慮了「過去的梯度」，但作用機制不同：
    *   Momentum 直接加總梯度，保留了**方向性**。
    *   RMSProp 對平方梯度進行平均，僅關注梯度的**大小**。因此，兩者不會互相抵銷。

## 學習率排程 (Learning Rate Scheduling)

除了自動調整每個參數的學習率外，整體學習率 $\eta$ 也可以隨著訓練過程動態調整。

### 學習率衰減 (Learning Rate Decay)

*   **概念：** 隨著訓練迭代次數的增加，整體學習率 $\eta$ 逐漸減小。
*   **原理：**
    *   訓練初期，參數距離最佳點較遠，可以採用較大的學習率快速探索。
    *   訓練後期，參數接近最佳點，需要較小的學習率來微調，避免在最佳點附近震盪，有助於更穩定地收斂。
*   **應用：** 結合 Adagrad 時，學習率衰減可以有效解決 Adagrad 後期因 $\sigma$ 過大導致訓練停滯或「噴」出去的問題，讓參數最終能平順地走到終點。

### 熱身 (Warm Up)

*   **概念：** 一種反直覺的學習率排程策略，學習率 $\eta$ 會先緩慢增大，達到峰值後再開始衰減。
*   **應用案例：**
    *   許多知名深度學習模型，如 Residual Network (ResNet, 2015) 和 Transformer，都採用了 Warm Up 策略，並強調其對訓練成功的關鍵作用，即便沒有給出明確的理論解釋，論文中也常將其作為「黑科技」記錄。
*   **可能解釋 (仍是研究問題)：**
    *   在訓練初期，Adagrad、RMSProp 或 Adam 中的 $\sigma$ 值是基於很少的梯度數據計算的，因此並不精準。
    *   Warm Up 期間，較小的學習率允許模型在初始點附近進行「探索」，收集足夠的梯度信息，使得 $\sigma$ 的估計變得更為精準。
    *   一旦 $\sigma$ 估計穩定，學習率再逐漸增大，以加速訓練過程。

## 優化器 (Optimizer) 總結與展望

### 梯度下降的演進

從最原始的梯度下降，我們逐步進化出更完善的優化策略：
1.  **Momentum：** 納入過去梯度的累積方向，加速在一致方向上的前進，並減少震盪。
2.  **Adaptive Learning Rate (例如 RMSProp 或 Adam)：** 根據每個參數的梯度大小自動調整其學習率。
    *   Momentum 考慮梯度**方向**，而自適應學習率考慮梯度**大小**，兩者互補。
3.  **Learning Rate Scheduling (例如 Learning Rate Decay 或 Warm Up)：** 根據訓練階段動態調整整體學習率。

今天的最佳實踐通常會結合這三種技術，例如使用 Adam 優化器 (已內含 Momentum 和 RMSProp 的思想)，並搭配 Learning Rate Scheduling (如 Warm Up 後再 Decay)。

### 未來方向：平坦化 Error Surface

本課程至今討論的優化技巧，都是在 Error Surface 崎嶇不平的前提下，嘗試找到更好的「路徑」來訓練模型（「山不轉路轉」）。然而，是否有可能直接「移平」或「剷平」這個 Error Surface，使其變得更容易訓練？

*   下週的課程將探討如何透過**改變類神經網路的架構、激活函數或其他內部機制**，來直接影響 Error Surface 的形狀，使其更加平坦，從而簡化優化過程（「神羅天征，把山炸平」）。

---

## 隨堂測驗

### 測驗一

當訓練深度學習模型時，Loss 曲線不再下降但梯度（Gradient Norm）仍然很大，最可能的原因是什麼？
A. 模型已達到 Local Minima
B. 模型已達到 Saddle Point
C. 學習率過大，導致參數在陡峭的峽谷中震盪
D. 學習率過小，導致模型無法前進

<details>
  <summary>點擊展開解答</summary>
  <p><b>正確答案：C</b></p>
  <p><b>解釋：</b> 當 Loss 不再下降但梯度仍然很大時，通常表示參數在 Error Surface 的陡峭峽谷中來回震盪，無法穩定收斂。這通常是學習率過大導致的。Local Minima 和 Saddle Point 都意味著梯度會變得很小甚至接近零。</p>
</details>

### 測驗二

關於 Adagrad、RMSProp 和 Adam 優化器，下列敘述何者錯誤？
A. Adagrad 的學習率會隨著時間不斷減小，可能導致訓練過早停止。
B. RMSProp 通過指數加權移動平均，能更好地響應近期的梯度變化。
C. Adam 結合了 RMSProp 的自適應學習率和 Momentum 的梯度方向累積。
D. Adagrad 的 $\sigma_i^t$ 和 Momentum 的累計梯度在計算上是完全相同的，會互相抵銷。

<details>
  <summary>點擊展開解答</summary>
  <p><b>正確答案：D</b></p>
  <p><b>解釋：</b></p>
  <ul>
    <li>Adagrad 的 $\sigma_i^t$ 是基於過去<b>平方梯度</b>的 Root Mean Square，只關注梯度<b>大小</b>。</li>
    <li>Momentum 則是直接加總過去的<b>梯度向量</b>，同時考慮梯度<b>方向</b>和大小。</li>
  </ul>
  <p>兩者在計算方式和所關注的梯度屬性上均不同，因此不會互相抵銷，而是相互配合來優化訓練過程。</p>
</details>

### 測驗三

Warm Up 學習率排程策略的特點是什麼？其可能的解釋為何？
A. 學習率在訓練初期較大，隨後逐漸減小，以快速收斂。
B. 學習率在訓練初期較小，隨後逐漸增大，達到峰值後再減小。
C. 學習率保持固定不變，以便於模型的穩定訓練。
D. 學習率與模型性能掛鉤，性能下降時增大學習率。

<details>
  <summary>點擊展開解答</summary>
  <p><b>正確答案：B</b></p>
  <p><b>解釋：</b> Warm Up 的特點是學習率在訓練初期先逐漸增大，達到一定峰值後再開始遞減。其可能的解釋是，在訓練初期，梯度估計（如 Adagrad 或 RMSProp 中的 $\sigma$）可能不夠精準，較小的學習率可以讓模型進行探索並收集更多梯度信息，從而使估計更準確。一旦梯度估計變得可靠，學習率再逐漸增大以加速訓練，隨後再減小以穩定收斂。</p>
</details>