# 機器學習 2021 (ML 2021) 自動化課程筆記

這是一個基於 [Quartz](https://quartz.jzhao.xyz/) 建立的靜態筆記網站，內容來自於 **國立台灣大學 李宏毅 (Hung-yi Lee) 教授** 的 [Machine Learning (2021)](https://speech.ee.ntu.edu.tw/~hylee/ml/2021-spring.php) 課程。

所有的筆記內容、逐字稿擷取、重點整理以及 Mermaid 知識圖譜，皆是由 AI 自動化生成的學習輔助資料。

## 🌐 網站連結

本專案目前已部署於 Cloudflare Pages：
**[👉 點此瀏覽 ML2021 筆記網站 (https://ml2021-notes.pages.dev)](https://ml2021-notes.pages.dev)**

## 🚀 是否支援 GitHub Pages (github.io)?

**完全支援！** Quartz 原生就非常適合部署在 GitHub Pages 上。
如果您希望將這個網站部署到 `您的帳號.github.io`，您可以透過以下兩種方式輕鬆達成：

1. **使用 GitHub Actions (自動部署)**：
   Quartz 內建了 GitHub Actions 的工作流程（在 `.github/workflows` 資料夾中）。您只需要在 GitHub 儲存庫的 `Settings -> Pages` 中，將來源 (Source) 設定為 `GitHub Actions`，每次 push 到 `main` 分支時，GitHub 就會自動幫您打包並發布到 GitHub Pages！

2. **使用 Quartz Sync (指令發布)**：
   如果您在本地端執行，可以先在本地設定好 GitHub repository，然後直接執行：
   ```bash
   npx quartz sync
   ```
   它會自動將最新的網頁打包並推送到您 GitHub repo 的分支中進行部署。

> **註**：Cloudflare Pages 與 GitHub Pages 兩者皆是極佳的靜態網站託管選擇，Cloudflare 通常在圖片載入與快取速度上更有優勢，但功能上是完全一樣的！

## 🛠️ 本地端執行開發

如果您想在自己的電腦上預覽或修改這些筆記，請按照以下步驟執行：

1. 安裝 [Node.js](https://nodejs.org/) (建議 v18 以上)
2. 複製此專案並進入資料夾：
   ```bash
   git clone https://github.com/leo02311/ml2021-notes.git
   cd ml2021-notes
   ```
3. 安裝相依套件：
   ```bash
   npm install
   ```
4. 啟動本地開發伺服器：
   ```bash
   npx quartz build --serve
   ```
5. 打開瀏覽器瀏覽 `http://localhost:8080` 即可預覽網站！

---

## ⚖️ 版權聲明 (Copyright Disclaimer)
本專案為**非商業性之個人開源學習專案**，所有課程核心內容（包含逐字稿、簡報影像、課程知識脈絡）之智慧財產權與著作權，皆完全屬於 **國立台灣大學 李宏毅教授** 所有。
本網站的排版與筆記摘要係由 AI 自動輔助生成，僅供社群交流與學術推廣使用。如原作者認為有任何侵權或不妥之處，請與我們聯繫下架。
