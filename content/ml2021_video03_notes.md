---
title: "第03堂課：機器學習任務攻略 (Overfitting & Optimization)"
tags:
  - MachineLearning
  - DeepLearning
  - Overfitting
  - Optimization
---

# 第03堂課：機器學習任務攻略 (Overfitting & Optimization)

> [!NOTE]
> 講師：李宏毅 (Hung-yi Lee)
> 本筆記由 AI 助手根據課程逐字稿自動整理生成。

當我們把模型設定好，卻發現在 Kaggle 或實際場域上的測試結果不如預期時，我們不應該盲目猜測，而是要遵循一套標準的「攻略」來找出問題。

## 1. 檢查訓練資料的 Loss (Training Loss)

當結果不好時，第一步永遠是先檢查你的 **Training Loss**。
許多新手只關注 Testing Loss，這是錯誤的。如果你的模型連在訓練資料上都考不好，根本不用管測試資料的結果。

如果 Training Loss 很大，有兩種可能：

### 可能性 A：Model Bias (模型偏差)
* **原因**：你的模型太過簡單，彈性（Flexibility）不夠大。就像「大海撈針，但針根本不在海裡」。
* **解法**：重新設計模型，賦予它更大的彈性。
  * 增加更多的特徵（Features），例如從看過去 1 天的資料改成看過去 56 天的資料。
  * 使用更深、更廣的神經網路（Deep Learning）。

### 可能性 B：Optimization Issue (最佳化失敗)
* **原因**：模型的彈性其實已經夠大了（針在海裡），但是你的 Gradient Descent 演算法不給力，卡在 Local Minima，沒辦法把那根針撈出來。
* **如何判斷是 A 還是 B？**
  * 比較「淺層模型」與「深層模型」的 Training Loss。
  * 理論上，深層模型（例如 56 層）的彈性絕對大於淺層模型（例如 20 層）。如果深層的 Training Loss 反而比淺層還要**高**，那就絕對不是 Model Bias（因為 56 層大可複製 20 層的參數，剩下 36 層什麼都不做），而是你的 Optimization 壞掉了！

---

## 2. 檢查測試資料的 Loss (Testing Loss)

如果經過努力，你的 Training Loss 已經變得很小了，但一放到測試資料上，**Testing Loss 卻很大**，這個時候才叫做 **Overfitting (過度擬合)**。

### 為什麼會 Overfitting？
當模型的彈性（自由度）非常大，而你的訓練資料（資料點）又太少時，模型為了讓 Training Loss 降到最低，會產生非常崎嶇扭曲的曲線來穿過所有訓練資料點。但在沒有資料限制的空白區域，模型就會開始「Freestyle」，導致預測未知資料時產生極大的誤差。

### 解決 Overfitting 的兩大方向：

#### 方向一：增加資料量 (最有效的方法)
* **蒐集更多真實資料**：這是王道，當資料點夠密，模型就無法隨便 Freestyle。
* **Data Augmentation (資料擴增)**：不花額外力氣蒐集，而是用現有資料「變出」新資料。例如將圖片左右翻轉、裁切。但必須注意擴增要有合理性（例如人臉可以左右翻，但不能上下顛倒）。

#### 方向二：給模型加上限制 (Constrain)
不要讓模型有太大的自由度，迫使它只能選出合理的函數。
* **減少參數數量**：使用較少的神經元或較淺的網路。
* **共用參數 (Weight Sharing)**：例如 CNN (卷積神經網路) 就是針對影像特性，給予極大限制的特化模型，因此在影像辨識上特別強大。
* **Early Stopping (提早停止訓練)**。
* **Regularization (正規化)**。
* **Dropout**。

> [!WARNING]
> **不能給太多限制**
> 如果給模型的限制過大（例如強迫一定要是一條直線），你會重新掉回 **Model Bias** 的陷阱中！機器學習就是在「模型彈性」與「資料量」之間尋求完美的平衡。

---

## 🧠 知識圖譜 (Knowledge Graph)

```mermaid
graph TD
    Start['結果不理想'] --> CheckTrain['步驟一：檢查 Training Loss']
    
    CheckTrain -->|Loss 很大| TrainFail['訓練失敗']
    TrainFail --> Bias['Model Bias (模型太弱)']
    TrainFail --> Opt['Optimization 失敗 (找不到最佳解)']
    Bias -.->|解法| ModelUp['增加特徵、加深網路']
    
    CheckTrain -->|Loss 很小| CheckTest['步驟二：檢查 Testing Loss']
    CheckTest -->|Loss 很小| Success['完美收工']
    CheckTest -->|Loss 很大| Overfit['Overfitting (過度擬合)']
    
    Overfit --> Fix1['方向一：增加資料']
    Fix1 --> Aug['Data Augmentation']
    
    Overfit --> Fix2['方向二：限制模型']
    Fix2 --> Constrain['CNN, Early Stopping, Dropout, 正規化']
    
    Constrain -.->|限制過度| Bias
```

---

## 📝 課後測驗

**Q1：訓練一個 50 層的神經網路，發現其 Training Loss 比 5 層的網路還要高，最可能的原因是什麼？**
<details>
<summary>點擊查看答案</summary>
這通常是 Optimization 失敗（例如卡在 Local Minima 或梯度消失），因為 50 層的網路理論上包含了 5 層網路的解，彈性一定夠大，所以不是 Model Bias。
</details>

**Q2：下列哪一種方法「不適合」用來解決 Overfitting？**
(A) 增加訓練資料
(B) 增加神經網路的層數與神經元數量
(C) 使用 Data Augmentation
(D) 加入 Dropout 機制
<details>
<summary>點擊查看答案</summary>
(B)。增加網路層數與神經元會讓模型變得更複雜、彈性更大，這反而會加劇 Overfitting 的情況。
</details>
