# 機器學習 2021 (ML 2021) 自動化課程筆記

這是一個基於 [Quartz](https://quartz.jzhao.xyz/) 建立的靜態筆記網站，內容來自於 **國立台灣大學 李宏毅 (Hung-yi Lee) 教授** 的 [Machine Learning (2021)](https://speech.ee.ntu.edu.tw/~hylee/ml/2021-spring.php) 課程。

所有的筆記內容、逐字稿擷取、重點整理以及 Mermaid 知識圖譜，皆是由 AI 自動化生成的學習輔助資料。

## 🌐 部署與網站連結 (Deployment)

本專案目前已部署於 Cloudflare Pages：
**[👉 點此瀏覽 ML2021 筆記網站 (https://ml2021-notes.pages.dev)](https://ml2021-notes.pages.dev)**

## 🛠️ 本地端執行開發

如果您想在自己的電腦上預覽或修改這些筆記，請按照以下步驟執行：

1. 安裝 [Node.js](https://nodejs.org/) (建議 v22 以上)
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

## 🤝 貢獻與版權聲明 (Contributors & Copyright)

* **原課程版權**：本專案所有課程核心內容（包含逐字稿、簡報影像、知識脈絡）之智慧財產權與著作權，皆完全屬於 **國立台灣大學 李宏毅教授** 所有。
* **貢獻者 (Contributors)**：本專案架構與內容均由 AI 輔助建立，並由 [leo02311](https://github.com/leo02311) 及開源社群共同推動與維護。歡迎任何有興趣的朋友提出 Pull Request 協助修正與擴充內容！
* **免責聲明**：本專案為非商業性之開源學習專案，僅供交流與學術推廣。如原作者認為有任何侵權或不妥之處，請與我們聯繫下架。
