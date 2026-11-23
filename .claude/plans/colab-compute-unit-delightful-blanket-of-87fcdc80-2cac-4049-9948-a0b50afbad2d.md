# 讓 Algorithm/ 每個實驗都能在 Colab 跑起來 + compute unit 估算

## Context 背景

想把 repo 裡的實驗丟到 Colab（透過 `colab-remote-runner` skill 的 `colab_job.py`）當遠端算力，
並知道每個實驗該用哪種 runtime、要花多少 compute unit（CU）。

盤點結果（已確認）：
- `Algorithm/` 下約 30 個可執行 `.py`，全部是小型 CPU 工作；資料都在 `Datasets/`（共約 130 MB，最大 `MNIST.csv` 77 MB），腳本以 **repo root 為 cwd** 讀 `Datasets/...`。
- **沒有任何實驗真正需要 GPU**（唯一的 torch 腳本 `LinearRegression_CSM_pytorch.py` 是 CPU、1000 epoch 的小線性模型）。
- Colab = Python 3.13 + pandas 2 / 新版 sklearn → 目前多數腳本會壞：
  - 14 處 `df.drop([...], 1)`（pandas 2 移除 positional axis）
  - `Boosting_..._LightGBM.py`: `sklearn.cross_validation`
  - `SVM_Iris_Multiclass.py`: `sys.path.append("../SVM_MNIST")` 依賴 cwd + 需要 `mlxtend`
  - 可能還有 sklearn 參數移除（如 AdaBoost `algorithm='SAMME.R'`、LogisticRegression `multi_class`）— 在本機相容性測試中逐一發現
  - `plt.show()` headless → 設 `MPLBACKEND=Agg`
- 本機 `colab` CLI 可用，帳戶餘額 **571.15 CU**，目前無 active session。Keyring 鎖住 → 一律用 `colab_job.py --no-keyring`（或 `colab --auth=adc`）。
- 決策（使用者已選）：**直接修原始碼**；範圍 **只做 `Algorithm/`**，`Project/KaggleElo`、`Project/DCIC2019`、`CLiMF_Epinions_TensorFlow.py`（TF1 `contrib`）列為 follow-up，只在 manifest 標記需求與估算。

## Runtime 建議與 CU 估算

費率為 Colab 公開的近似值（會隨方案變動），Phase 3 smoke 時用 `colab usage` 的 `rate_per_hr` 實測並回填。

| Tier | 實驗 | 建議 runtime | 預估 Colab 時間 | 近似費率 | 預估 CU |
|---|---|---|---|---|---|
| A 秒級 | sklearn 版全部、LinearRegression×3、LogisticRegression×2、NaiveBayes×2、PCA×2、Post-Operative×2、AdaBoost/Adaboost×4、LightGBM、SVM_Iris | CPU 標準 | 每個 < 1 min，合計 ~5 min | ~0.07–0.1 CU/h | < 0.02 |
| B 分鐘級 (純 Python 迴圈) | kNN_FromScratch、DecisionTree_FromScratch、KMeans×2、FP-Growth(retail)、SVD_MSWeb、VSM/LSI Gensim、SVM_MNIST_Binary | CPU 標準 | 合計 ~20–40 min | 同上 | ~0.05 |
| C 可能較久 | SVM_MNIST_Multiclass_FromScratch（10× SMO）、CLiMF_FromScratch | CPU 標準（VSM 若 RAM 吃緊改 `--high-mem`） | 各 10–60 min | 同上 / high-mem ~0.13 CU/h | ~0.1 |
| D 選配 | HMM `CompareJiebaHanLP.py`（`pyhanlp` 需 JVM + 首次下載資料） | CPU | ~5 min | — | ~0.01 |
| Follow-up | KaggleElo（~3 GB CSV） | CPU `--high-mem`（~51 GB） | 1–2 h | ~0.13 CU/h | ~0.3 |
| Follow-up | DCIC2019（xgboost K-fold） | CPU 或 T4 | < 1 h | CPU ~0.1 / T4 ~1.5–1.8 CU/h | 0.1 / ~2 |

**結論**：全部 Algorithm 實驗在單一 CPU session 跑完 ≈ 1–1.5 h，**< 0.5 CU**；就算全改 T4 也只 ~2–3 CU。GPU 對這些實驗沒有加速效果，不建議開。

## Plan 實作步驟

### Phase 1 — 相容性修正（本機，不花 CU）
1. 用接近 Colab 的環境逐一跑腳本找錯：
   `uv run --python 3.13 --with pandas,numpy,scikit-learn,scipy,matplotlib,lightgbm,torch,tqdm,gensim,cvxopt,mlxtend python <script>`（cwd = repo root，`MPLBACKEND=Agg`，每個加 `timeout`）。
2. 最小修改原始碼：
   - `drop([...], 1)` → `drop(columns=[...])`（DecisionTree×2、KMeans×2、LinearRegression×3、NaiveBayes、SVM_MNIST×2、kNN×2）
   - `sklearn.cross_validation` → `sklearn.model_selection`
   - `SVM_Iris_Multiclass.py`: path 改成 `os.path.join(os.path.dirname(__file__), "../SVM_MNIST")`
   - 其餘在步驟 1 出現的已移除 API 逐一修
3. 不改演算法與超參數；只修讓它能跑的部分。

### Phase 2 — Project wrapper（依 skill 的 thin-wrapper pattern）
新增 `tools/colab/`：
- `experiments.toml` — 每個實驗一筆：`name`、`script`、`data`（需要的 `Datasets/` 檔案）、`pip`（額外套件，如 `cvxopt`、`mlxtend`、`gensim`、`jieba pyhanlp`）、`tier`、`timeout_min`、`est_min`、`status`（`ready` / `optional` / `followup`，follow-up 項目寫明缺什麼）。
- `vm_run_all.py` — 在 VM 上執行：讀 manifest、依序以 `timeout` 跑選定的實驗（cwd=repo root、`MPLBACKEND=Agg`），每個實驗 stdout/stderr 存 `out/<name>.log`，並寫 `out/summary.json`（exit code、wall 秒數、peak RSS）。單一實驗失敗不影響其他。
- `run_experiments.py`（PEP 723，本機執行）：
  - `--only/--tier/--status` 篩選；`--estimate` 只印上表的預估（不連 Colab）；`--dry-run` 轉傳給 `colab_job.py`。
  - 打包 `Algorithm/` + 所需 `Datasets/` 檔 + `tools/colab/` 成一個 tgz（MNIST 壓縮後很小）→ 一次 `colab_job.py --no-keyring run -s ml-practice --upload repo.tgz:/content/job/repo.tgz --pip ... --cmd "tar xzf ... && python tools/colab/vm_run_all.py ..." --tag compute=colab --out runs/colab`。
  - 一個 session 跑完所有選定實驗（攤平開機、pip install、上傳成本）；結束自動 stop（除非 `--keep`）。
  - 下載後把 `_colab_meta.json` 的 `timing_s`、`cost.units_estimated`、`vm` 與 `summary.json` 合併成 `runs/colab/<timestamp>/report.md`（每個實驗：狀態、秒數、估算 CU 分攤）。
- 重用：`.claude/skills/colab-remote-runner/scripts/colab_job.py`（session/upload/marker/timeout/download/stop 全部交給它，不重寫）。
- `.gitignore` 加 `runs/`。

### Phase 3 — Colab 驗證與實測
1. `run_experiments.py --dry-run --tier A` → 檢查計畫指令。
2. Smoke：`--tier A`（CPU），確認 result dir、`_colab_meta.json` tags、`rate_per_hr` 實際值。
3. 正式：`--tier A,B,C`（同一 CPU session），選配 D 另跑一次。
4. `colab sessions` 確認沒有殘留 session。
5. 用實測數字更新 `experiments.toml` 的 `est_min`，並在 `tools/colab/README.md`（英文，長期文件）記錄：每個實驗的建議 runtime、實測時間、CU、本機 vs Colab 差異、follow-up 清單（KaggleElo 需 Kaggle token/Drive + high-mem；DCIC 需資料；CLiMF TF 需改寫成 TF2）。

## Critical files 關鍵檔案
- 修改：`Algorithm/**/*.py`（約 15 個，見 Phase 1）、`.gitignore`
- 新增：`tools/colab/experiments.toml`、`tools/colab/run_experiments.py`、`tools/colab/vm_run_all.py`、`tools/colab/README.md`
- 重用：`.claude/skills/colab-remote-runner/scripts/colab_job.py`

## Verification 驗證
- 本機：Phase 1 的 uv 3.13 環境下所有 `ready` 實驗 exit 0。
- Colab：`report.md` 中所有 `ready` 實驗 exit 0；`cost.units_estimated` 與 `colab usage` 餘額差異合理（預期 < 0.5 CU）。
- `colab sessions` 為空。
- Commit 前讓使用者確認 diff（修正 + 新工具分開兩個 commit）。
