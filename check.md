# Проверка SQL для распределения тегов SOREL-20M

## SQL 1 — доля тегов среди всех файлов `meta`

Этот запрос отвечает на вопрос: **какой процент всех PE-файлов в SOREL имеет данный tag > 0**.

```sql
.mode column
.headers on
WITH totals AS (
  SELECT COUNT(*) AS n FROM meta
)
SELECT 'adware' AS tag, SUM(adware > 0) AS files, ROUND(100.0 * SUM(adware > 0) / n, 4) AS percent FROM meta, totals
UNION ALL SELECT 'flooder', SUM(flooder > 0), ROUND(100.0 * SUM(flooder > 0) / n, 4) FROM meta, totals
UNION ALL SELECT 'ransomware', SUM(ransomware > 0), ROUND(100.0 * SUM(ransomware > 0) / n, 4) FROM meta, totals
UNION ALL SELECT 'dropper', SUM(dropper > 0), ROUND(100.0 * SUM(dropper > 0) / n, 4) FROM meta, totals
UNION ALL SELECT 'spyware', SUM(spyware > 0), ROUND(100.0 * SUM(spyware > 0) / n, 4) FROM meta, totals
UNION ALL SELECT 'packed', SUM(packed > 0), ROUND(100.0 * SUM(packed > 0) / n, 4) FROM meta, totals
UNION ALL SELECT 'crypto_miner', SUM(crypto_miner > 0), ROUND(100.0 * SUM(crypto_miner > 0) / n, 4) FROM meta, totals
UNION ALL SELECT 'file_infector', SUM(file_infector > 0), ROUND(100.0 * SUM(file_infector > 0) / n, 4) FROM meta, totals
UNION ALL SELECT 'installer', SUM(installer > 0), ROUND(100.0 * SUM(installer > 0) / n, 4) FROM meta, totals
UNION ALL SELECT 'worm', SUM(worm > 0), ROUND(100.0 * SUM(worm > 0) / n, 4) FROM meta, totals
UNION ALL SELECT 'downloader', SUM(downloader > 0), ROUND(100.0 * SUM(downloader > 0) / n, 4) FROM meta, totals
ORDER BY percent DESC;
```

### Ожидаемый результат SQL 1, примерно

| Tag | Да, файлов | Нет, файлов | Примерный % от всех 19,724,997 файлов |
|---|---:|---:|---:|
| `spyware` | 4,550,007 | 15,174,990 | 23.07% |
| `packed` | 3,726,059 | 15,998,938 | 18.89% |
| `dropper` | 3,577,111 | 16,147,886 | 18.13% |
| `worm` | 3,414,132 | 16,310,865 | 17.31% |
| `file_infector` | 3,317,569 | 16,407,428 | 16.82% |
| `downloader` | 2,565,838 | 17,159,159 | 13.01% |
| `adware` | 2,411,262 | 17,313,735 | 12.22% |
| `ransomware` | 1,152,354 | 18,572,643 | 5.84% |
| `installer` | 1,018,880 | 18,706,117 | 5.17% |
| `crypto_miner` | 339,565 | 19,385,432 | 1.72% |
| `flooder` | 101,595 | 19,623,402 | 0.52% |

Дополнительная sanity-check строка для всей таблицы:

| Label | Да, файлов | Нет, файлов | Примерный % от всех файлов |
|---|---:|---:|---:|
| `is_malware` | 9,762,177 | 9,962,820 | 49.49% |

---

## SQL 2 — доля тегов среди malware-файлов

Этот запрос отвечает на вопрос: **какой процент вредоносных PE имеет данный tag > 0**.

```sql
.mode column
.headers on
WITH totals AS (
  SELECT COUNT(*) AS n FROM meta WHERE is_malware = 1
)
SELECT 'adware' AS tag, SUM(adware > 0) AS files, ROUND(100.0 * SUM(adware > 0) / n, 4) AS percent_of_malware FROM meta, totals WHERE is_malware = 1
UNION ALL SELECT 'flooder', SUM(flooder > 0), ROUND(100.0 * SUM(flooder > 0) / n, 4) FROM meta, totals WHERE is_malware = 1
UNION ALL SELECT 'ransomware', SUM(ransomware > 0), ROUND(100.0 * SUM(ransomware > 0) / n, 4) FROM meta, totals WHERE is_malware = 1
UNION ALL SELECT 'dropper', SUM(dropper > 0), ROUND(100.0 * SUM(dropper > 0) / n, 4) FROM meta, totals WHERE is_malware = 1
UNION ALL SELECT 'spyware', SUM(spyware > 0), ROUND(100.0 * SUM(spyware > 0) / n, 4) FROM meta, totals WHERE is_malware = 1
UNION ALL SELECT 'packed', SUM(packed > 0), ROUND(100.0 * SUM(packed > 0) / n, 4) FROM meta, totals WHERE is_malware = 1
UNION ALL SELECT 'crypto_miner', SUM(crypto_miner > 0), ROUND(100.0 * SUM(crypto_miner > 0) / n, 4) FROM meta, totals WHERE is_malware = 1
UNION ALL SELECT 'file_infector', SUM(file_infector > 0), ROUND(100.0 * SUM(file_infector > 0) / n, 4) FROM meta, totals WHERE is_malware = 1
UNION ALL SELECT 'installer', SUM(installer > 0), ROUND(100.0 * SUM(installer > 0) / n, 4) FROM meta, totals WHERE is_malware = 1
UNION ALL SELECT 'worm', SUM(worm > 0), ROUND(100.0 * SUM(worm > 0) / n, 4) FROM meta, totals WHERE is_malware = 1
UNION ALL SELECT 'downloader', SUM(downloader > 0), ROUND(100.0 * SUM(downloader > 0) / n, 4) FROM meta, totals WHERE is_malware = 1
ORDER BY percent_of_malware DESC;
```

### Ожидаемый результат SQL 2, примерно

Эти проценты посчитаны как `tag_yes / 9,762,177 malware`. Это ожидаемая оценка, потому что опубликованная Sophos table даёт общие tag counts, а не отдельную crosstab `tag AND is_malware`. Если в `meta.db` найдутся tagged benign rows, SQL 2 может немного отличаться.

| Tag | Да, файлов | Примерный % от malware-файлов |
|---|---:|---:|
| `spyware` | 4,550,007 | 46.61% |
| `packed` | 3,726,059 | 38.17% |
| `dropper` | 3,577,111 | 36.64% |
| `worm` | 3,414,132 | 34.97% |
| `file_infector` | 3,317,569 | 33.98% |
| `downloader` | 2,565,838 | 26.28% |
| `adware` | 2,411,262 | 24.70% |
| `ransomware` | 1,152,354 | 11.80% |
| `installer` | 1,018,880 | 10.44% |
| `crypto_miner` | 339,565 | 3.48% |
| `flooder` | 101,595 | 1.04% |

---

## Запуск

```bash
sqlite3 datasets/sorel20m/meta.db < check_tags.sql
```

Или интерактивно:

```bash
sqlite3 datasets/sorel20m/meta.db
```

- `spyware ≈ 23.07% от всех файлов` означает: примерно 23% всех PE в `meta` имеют `spyware > 0`.
- `spyware ≈ 46.61% от malware` означает: если считать denominator только malware, почти половина malicious PE имеет spyware-related tokens.
- Сумма процентов по тегам больше 100%, потому что malware может одновременно быть `packed`, `dropper`, `downloader`, `spyware` и т.д.
- Это не CVE-распределение. SOREL tags описывают malware behavior labels из detection names, а не уязвимости продуктов.
