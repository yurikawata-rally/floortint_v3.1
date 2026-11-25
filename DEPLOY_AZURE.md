# Azure App Service デプロイ手順書 — floortint v2

本書は、この資材一式を **Azure App Service（Web App for Containers, Linux）**
で稼働させるための手順です。Azure サブスクリプションの操作権限を持つ
エンジニア / PM を対象としています。

## 全体像

```
[この資材] --docker build--> [コンテナイメージ]
    --push--> [Azure Container Registry (ACR)]
    --pull--> [App Service (Web App for Containers)]
```

- アプリはコンテナ内でポート **8000** で待ち受けます（`WEBSITES_PORT=8000` を設定）。
- ヘルスチェックは `/_stcore/health`（HTTP 200 で正常）。
- 外部 API キーは不要です。イメージに資格情報は含まれません。
- 日本語フォント（IPA P ゴシック）はイメージのビルド時に apt で入ります。

## 前提

- Azure CLI (`az`) ログイン済み、対象サブスクリプション選択済み
- Docker が使えるビルド環境（ローカル or CI）。
  ※ Docker がない場合も `az acr build` でクラウドビルド可能（手順 2-B）

以下、リソース名は例です。組織の命名規則に合わせて読み替えてください。

```bash
RG=rg-floortint                 # リソースグループ
ACR=acrfloortint                # ACR 名（グローバル一意・英数字のみ）
PLAN=plan-floortint             # App Service プラン
APP=app-floortint               # Web App 名（グローバル一意）
LOC=japaneast
IMG=floortint:v2
```

## 手順

### 1. リソースグループと ACR

```bash
az group create -n $RG -l $LOC
az acr create -n $ACR -g $RG --sku Basic
```

### 2. イメージのビルドとプッシュ

**2-A. ローカル Docker でビルドする場合**（この資材のルートで実行）

```bash
az acr login -n $ACR
docker build -t $ACR.azurecr.io/$IMG .
docker push $ACR.azurecr.io/$IMG
```

※ Apple Silicon 等 ARM マシンでは `docker build --platform linux/amd64` を付けること。

**2-B. Docker なしで ACR にクラウドビルドさせる場合**

```bash
az acr build -r $ACR -t $IMG .
```

### 3. App Service プランと Web App の作成

```bash
az appservice plan create -n $PLAN -g $RG --is-linux --sku P1v3
az webapp create -n $APP -g $RG -p $PLAN \
  --container-image-name $ACR.azurecr.io/$IMG \
  --assign-identity [system]
# マネージド ID に ACR pull 権限を付与（admin キー不要の推奨構成）
az webapp config set -n $APP -g $RG --generic-configurations '{"acrUseManagedIdentityCreds": true}'
PRINCIPAL=$(az webapp identity show -n $APP -g $RG --query principalId -o tsv)
ACR_ID=$(az acr show -n $ACR -g $RG --query id -o tsv)
az role assignment create --assignee $PRINCIPAL --scope $ACR_ID --role AcrPull
```

SKU の目安: 300dpi の A3 出力（約 17M 画素）で 1.5GB 程度のメモリを使い、複数人が同時に使うと
その人数分かかります。**P1v3（8GB）を推奨**します。B1（1.75GB）では 1 人でも図面帳の
取込でメモリ不足になることがあります。

### 4. アプリ設定（環境変数）

```bash
az webapp config appsettings set -n $APP -g $RG --settings \
  WEBSITES_PORT=8000 \
  WEBSITES_CONTAINER_START_TIME_LIMIT=300 \
  MAX_UPLOAD_MB=200
```

**床材ライブラリと物件設定の保存先**（v2）。登録した写真と保存した設定はコンテナの `/app/library` に置かれ、再起動で消えます。
残すには永続ストレージを当てます。方法は 2 つ。

```bash
# A. App Service の永続ストレージ（/home 配下）を使う
az webapp config appsettings set -n $APP -g $RG --settings \
  WEBSITES_ENABLE_APP_SERVICE_STORAGE=true \
  FLOORTINT_LIBRARY=/home/data/floortint_library
# B. Azure Files を /home/data にマウントする（複数インスタンスや他ツールと共有するとき）
az storage account create -n <storage> -g $RG -l $LOC --sku Standard_LRS
az storage share create --account-name <storage> -n floortint
az webapp config storage-account add -n $APP -g $RG --custom-id floortint --storage-type AzureFiles \
  --account-name <storage> --share-name floortint --access-key "<key>" --mount-path /home/data
```

コンテナは非 root（appuser）で動くので、マウント先に書き込めることを起動ログ `[entrypoint] library:` で確認してください
（書けないときは `WARNING: library dir not writable` が出ます）。

サンプル案件を画面に出したい場合だけ、永続ストレージに PDF を置いて場所を指定します
（図面は社内限のため、閉域または認証つきの環境に限ること）:

```bash
az webapp config appsettings set -n $APP -g $RG --settings \
  FLOORTINT_SAMPLES=/home/data/floortint_samples \
  FLOORTINT_SAMPLES2=/home/data/floortint_samples2   # 指示書モードの案件
```

### 5. ヘルスチェックと WebSocket

```bash
az webapp config set -n $APP -g $RG --web-sockets-enabled true
az webapp config set -n $APP -g $RG --generic-configurations '{"healthCheckPath": "/_stcore/health"}'
```

- **WebSocket は必須**です（Streamlit はブラウザと WebSocket で通信します）。
- ARR affinity（既定で ON）は **ON のまま**にしてください。
  Streamlit はサーバ側セッション状態を持つため、スケールアウト時に
  同一インスタンスへ振り分ける必要があります。

### 6. 動作確認

```bash
az webapp browse -n $APP -g $RG   # または https://$APP.azurewebsites.net
```

1. ページが表示される（初回はコンテナ pull のため 1〜2 分かかることあり）
2. 左ペイン「着色の根拠」で「床仕上図から」を選び、「PDF をアップロード」で平面図と床仕上図の PDF を入れる → 数秒〜十数秒で着色結果が出る
3. 上段の 6 項目に縮尺・図面の整理（レイヤー名）・位置合わせ（通り芯）・材種数・領域数・家具数が出る
4. 「🛠 処理の内訳」タブで段階がすべて ✅ で、警告（⚠️）が無い
5. 「指示書から」に切り替え、平面図と指示書（PDF か画像）を入れる → 「📋 材種と割り当て」で色 → 材種を選ぶと塗り分けが変わる
6. 「🗂 床材ライブラリ」で写真を 1 枚登録し、再起動後も残っている（永続ストレージが効いている）
7. 「💾 書き出し」で PNG / PDF / レイヤー分け ZIP がダウンロードできる

うまく動かない場合のログ確認:

```bash
az webapp log tail -n $APP -g $RG
```

### 7. 更新（再デプロイ）

コード修正後はイメージを再ビルド・プッシュして再起動するだけです:

```bash
az acr build -r $ACR -t floortint:v2 .
az webapp config container set -n $APP -g $RG \
  --container-image-name $ACR.azurecr.io/floortint:v2
```

## トラブルシューティング

| 症状 | 原因と対処 |
| --- | --- |
| 起動直後に「Application Error」 | `WEBSITES_PORT=8000` の設定漏れ。設定後に再起動 |
| ページは出るが「Please wait...」のまま | WebSocket が無効。手順 5 を実施 |
| PDF アップロードで 413 エラー | `MAX_UPLOAD_MB` を超えている。値を上げて再起動 |
| 着色結果の室名が □（豆腐）になる | フォントが入っていない。Dockerfile の `fonts-ipafont-gothic` が入ったイメージか確認 |
| 「凡例に無い色の線がある」の警告 | 床仕上図に凡例へ載っていない色のハッチがある。凡例を直すか、その範囲は塗られないことを了承する |
| 「位置合わせは画像の相関」の警告 | 通り芯ラベルが両図に無い。結果がずれていれば「抽出の詳細」の手動補正で x/y を合わせる |
| 領域が 0、または着色されない | 床仕上図が色別ハッチ線でなく塗りつぶしやスキャン画像。本アプリの対象外 |
| 図面帳の取込中にコンテナが再起動 | メモリ不足。プランを P1v3 以上へ。または出力解像度を 150dpi に下げる |
| 材種の数が凡例より多い | 中止行の打ち消しハッチなど。「📋 材種と割り当て」で該当行の領域を無効にする |
| 指示書の色が平面図とずれる | 画像の指示書の位置合わせが弱い（段階 10 に警告）。「抽出の詳細」の手動補正（倍率・x・y）で合わせる |
| 部屋が 0、または開放エリアが塗られない | 壁や間仕切りの線に切れ目がある。「線の切れ目を埋める幅」を上げる。部屋の最小面積も確認 |
| 登録した写真や保存した設定が消えた | 永続ストレージが当たっていない。手順 4 の `FLOORTINT_LIBRARY` と起動ログを確認 |

## 運用上の注意

- **認証**: アプリ自体にログイン機能はありません。扱う図面は顧客物件の社内限資料なので、
  App Service の「認証（Easy Auth / Entra ID）」を有効化するか、
  VNet 統合＋プライベートエンドポイントでの閉域構成を推奨します。
- **データの保存**: アップロードした PDF はメモリ上でだけ処理し、ディスクに保存しません。
  抽出結果のキャッシュはプロセス内（Streamlit のキャッシュ、最大 3 件）で、再起動で消えます。床材の写真と物件設定（図面は含まない）だけが `FLOORTINT_LIBRARY` に残ります。
- **サンプル案件**: この資材にはサンプル図面を同梱していません。`app/samples/` は空です。
  同梱する場合は `app/run_cases.py` の `CASES` にファイル名とページ番号を合わせ、
  取り扱いを社内限としてください。
- **ログ**: アプリは図面の内容をログに出力しません。


## v3 での追加事項（2026-10-16）

- イメージに LibreOffice（`libreoffice-impress`）を含めた（PPTX の指示書を PDF に変換するため。イメージが約 400MB 増える）。PPTX を使わない運用なら Dockerfile の該当行を外してよい。
- 依存に `scipy` を追加した。
- ライブラリ（`FLOORTINT_LIBRARY`）には指示書から自動登録した品番・製品写真・リンクが溜まる。永続ストレージ（Azure Files など）を当てないと再起動で消える点は v2 と同じ。
