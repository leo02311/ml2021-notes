# 李宏毅 2021 機器學習課程筆記 (ML 2021 Digital Garden)

這是一個基於 [Quartz](https://quartz.jzhao.xyz/) 建立的「數位花園」網站，致力於將李宏毅老師的 [2021 機器學習課程 (Machine Learning 2021)](https://speech.ee.ntu.edu.tw/~hylee/ml/2021-spring.php) 轉化為最容易閱讀、最具互動性的終極圖文版筆記。

🔗 **線上瀏覽網址**: [https://ml2021-notes.pages.dev/](https://ml2021-notes.pages.dev/)

## ✨ 核心特色 (Features)

這個專案不僅僅是文字的堆疊，而是透過多種自動化技術與 AI 輔助，打造出極致的學習體驗：

*   🧠 **知識圖譜梳理 (Mermaid Knowledge Graphs)**
    每堂課的筆記末尾皆附帶自動生成的 Mermaid 知識圖譜，完美相容 Quartz 的深色/淺色模式，幫助您一眼看懂概念間的邏輯關聯。
*   🧮 **完美的數學排版 (LaTeX)**
    所有的微積分推導、矩陣運算與損失函數皆使用嚴謹的 LaTeX 語法渲染，告別難以閱讀的純文字公式。
*   📊 **程式化動態圖表 (Dynamic Programmatic Visualizations)**
    我們特別針對難以想像的數學概念（例如：Sigmoid 疊加曲線、Gradient Descent 的收斂軌跡、3D 鞍點/Saddle Point 的誤差曲面），使用 Python (`matplotlib`) 生成了高畫質的靜態圖表與 **GIF 動畫**。
    *   *支援深淺主題切換：透過 HTML `<picture>` 標籤，圖表會根據您的作業系統自動切換為深色或淺色風格。*
*   📱 **響應式與超連結 (Responsive & Networked)**
    支援 Obsidian 的雙向連結 (Backlinks)，並具備極速的全文搜尋體驗與標籤分類。

## 🚀 本地執行與開發 (Local Development)

若您想要在本地端運行或修改此網站，請確保您已安裝 Node.js (v18.14 或以上版本)。

1.  **安裝依賴套件**:
    ```bash
    npm install
    ```

2.  **啟動本地伺服器**:
    ```bash
    npx quartz build --serve
    ```
    伺服器啟動後，請在瀏覽器開啟 `http://localhost:8080` 預覽網站。

3.  **編譯並部署**:
    網站已設定透過 GitHub Actions 自動同步至 Cloudflare Pages。只要推送到 `main` 分支，即會自動觸發更新。

## 📁 資料夾結構說明

*   `content/`：存放所有的 Markdown 筆記。
*   `content/assets/`：存放所有自動生成的圖表、GIF 動畫與資源檔。
*   `visualizations/` (根目錄外)：存放用於生成 `assets/` 內 3D 圖表與視覺化的 Python 腳本 (`generate_charts.py`)。

---
> *本專案筆記內容來源為李宏毅老師 YouTube 授課影片之逐字稿，並由 AI 輔助整理與排版。*
