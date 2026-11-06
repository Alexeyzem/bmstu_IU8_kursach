.mode column
.headers on

.print 'Доли tags во всём SOREL-20M'
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

.print ''
.print 'Доли tags только среди malware'
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
