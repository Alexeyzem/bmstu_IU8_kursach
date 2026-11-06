# Gold validation PE set: источники, безопасность, leakage-control

Практически файлы берутся так:

1. **Malicious 2,000–2,500:** MalwareBazaar через `scripts/download_gold_pe_samples.py`. Нужен бесплатный Auth-Key с <https://auth.abuse.ch/>.
2. **Benign 2,000–2,500:** сначала сгенерировать большой manifest из WinGet:

```bash
python3 scripts/build_winget_benign_manifest.py \
  --output benign_pe_sources.json \
  --limit 2500
```

Потом скачать benign PE из этого manifest:

```bash
python3 scripts/download_gold_pe_samples.py \
  --malware-limit 0 \
  --benign-limit 2500 \
  --benign-manifest benign_pe_sources.json
```

После получения MalwareBazaar key скачать malicious часть:

```bash
export MALWAREBAZAAR_AUTH_KEY="YOUR_AUTH_KEY"
python3 scripts/download_gold_pe_samples.py \
  --malware-limit 2500 \
  --benign-limit 0 \
  --i-understand-malware-handling
```

Итоговая папка `datasets/gold_pe/` должна содержать `metadata.csv`, `benign/*.txt`, `malicious/*.zip.txt`.

## Современные PE-угрозы, которые нужно покрыть

| Категория | Примеры | Почему включать |
|---|---|---|
| Stealers / MaaS | LummaC2, Rhadamanthys, RedLine, Vidar, StealC, Raccoon-like | Активный post-2020 класс Windows PE; часто packed/obfuscated; важен для проверки drift относительно SOREL |
| Loaders / droppers / initial access | HijackLoader, MintsLoader, SocGholish, DarkGate, Amadey-like | Часто являются precursor для ransomware; статические признаки могут отличаться от старых семейств |
| Ransomware / precursor activity | Akira, Qilin, Play, LockBit/Black Basta/RansomHub-related | Высокорисковый класс; важно оценивать TPR при низком FPR |
| RAT / backdoor / botnet | AgentTesla, Remcos, AsyncRAT, Quasar, njRAT, FormBook/XLoader-like | Хорошо представлен в PE; имеет характерные imports/strings/packer artifacts |
| Packed / obfuscated PE | Разные families | Проверяет устойчивость EMBER-v2 признаков: entropy, sections, imports, strings |

## Источники malicious PE

| Источник | URL | Роль | Ограничения |
|---|---|---|---|
| MalwareBazaar / abuse.ch | <https://bazaar.abuse.ch/api/> | Основной источник свежих malware PE; API отдаёт metadata и password-protected ZIP samples | Auth-Key обязателен; fair use; samples в ZIP с паролем `infected`; не выполнять и не распаковывать вне isolated lab |
| MalShare | <https://malshare.com/doc.php> | Резервный источник для добора и cross-check | API key; labels менее curated; нужна независимая проверка |
| Malpedia | <https://malpedia.caad.fkie.fraunhofer.de/usage/api> | Качественные curated family anchors | Регистрация/API token; учитывать TLP/licensing |
| VirusShare | <https://virusshare.com/> | Резервный malware corpus / hash lookup | Login; malware only; не curated ML benchmark |
| VirusTotal / ReversingLabs | <https://github.com/VirusTotal/vt-cli>, <https://docs.reversinglabs.com/> | Проверка reputation/metadata, если есть легальный доступ | Search/download часто premium; не загружать proprietary benign без разрешения |

## Источники benign PE

| Источник | URL | Роль | Ограничения |
|---|---|---|---|
| WinGet manifests | <https://github.com/microsoft/winget-pkgs> | Основной индекс официальных installer URLs и hashes | Manifests открыты, но сами installers имеют отдельные лицензии; не публиковать proprietary binaries |
| Microsoft Download Center | <https://www.microsoft.com/en-us/download> | High-confidence benign Microsoft redistributables/tools | Соблюдать лицензии |
| Sysinternals | <https://learn.microsoft.com/en-us/sysinternals/license-terms> | Небольшой подписанный benign набор | Лицензия ограничивает redistribution; лучше хранить hashes/URLs |
| Scoop Main bucket | <https://github.com/ScoopInstaller/Main> | Добор portable CLI/utility software | Использовать official URLs и hashes |
| Chocolatey Community | <https://community.chocolatey.org/> | Индекс популярных Windows packages | Не считать label абсолютным; исключать bundleware/adware; проверять official URL/signature |

## Leakage-control

1. Exact hash dedup: исключить SHA-256/SHA-1/MD5, найденные в SOREL `meta.db`, EMBER metadata, BODMAS metadata и уже собранной выборке.
2. Temporal cutoff: брать 2021+; SOREL покрывает 2017-01-01—2019-04-10, EMBER — до 2017/2018.
3. Near-duplicate caps: ограничивать TLSH/ssdeep/imphash clusters; не больше 5–10 samples на near-duplicate cluster.
4. Family/source caps: не больше 10–15% на одну family/signature, не больше 20–25% из одного календарного квартала.
5. Не включать filename/path/source в feature vector, чтобы не было label leakage.

## Безопасное хранение

- MalwareBazaar samples хранить как password-protected ZIP, переименованные в `.zip.txt`; не распаковывать автоматически.
- Benign raw PE хранить как `.exe.txt`, `.dll.txt`, `.pe.txt`; не запускать.
- Каталог `datasets/gold_pe/` содержит `README_SAFETY.txt`, `metadata.csv`, `benign/`, `malicious/`.
- Работать на noexec/offline/encrypted volume, если доступны политики ОС.
- Для сдачи/публикации лучше отдавать scripts + metadata + hashes + URLs, а не сами malware/benign binaries.

## Скрипт

`scripts/download_gold_pe_samples.py`:

- `--dry-run` показывает план без скачивания;
- `--malwarebazaar-auth-key` или `MALWAREBAZAAR_AUTH_KEY` нужен для MalwareBazaar;
- `--i-understand-malware-handling` обязателен для фактической загрузки malware;
- benign URLs берутся из `benign_pe_sources.json`;
- `--exclude-hashes` принимает файл с SHA-256 для исключения SOREL/EMBER/BODMAS overlaps;
- metadata пишется в `datasets/gold_pe/metadata.csv`.

`scripts/build_winget_benign_manifest.py`:

- читает public `microsoft/winget-pkgs` tree;
- выбирает latest `.installer.yaml` на package;
- берёт HTTPS `.exe` installer URLs с `InstallerSha256`;
- пишет `benign_pe_sources.json` для `download_gold_pe_samples.py`;
- бинарники не скачивает, только строит список источников.
