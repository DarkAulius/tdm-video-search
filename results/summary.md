# Risultati

Test: 1000 clip di MSR-VTT 1k-A, di cui 662 con parlato rilevato. Configurazione scelta sulla validazione (300 clip): strategia `uniform`, aggregazione `max_segments`, peso del parlato w = 0.15.

## Recupero testo -> video (test)

| Strategia | Aggregazione | w | Visivo R@1 | R@5 | R@10 | MdR | Fusione R@1 | R@5 | R@10 | MdR |
|---|---|---|---|---|---|---|---|---|---|---|
| uniform | mean_frames | 0.25 | 31.1 | 52.4 | 62.6 | 5 | 33.1 | 56.1 | 66.4 | 4 |
| uniform | mean_segments | 0.10 | 31.1 | 52.8 | 62.4 | 4 | 32.5 | 55.4 | 65.1 | 4 |
| uniform **(scelta)** | max_segments | 0.15 | 28.8 | 53.6 | 63.5 | 4 | 30.9 | 55.9 | 66.2 | 4 |
| iframes | mean_frames | 0.20 | 27.9 | 50.9 | 60.7 | 5 | 29.9 | 53.8 | 63.0 | 5 |
| iframes | mean_segments | 0.10 | 26.9 | 49.1 | 60.9 | 6 | 28.7 | 51.3 | 61.3 | 5 |
| iframes | max_segments | 0.10 | 26.3 | 48.2 | 59.5 | 6 | 28.6 | 51.9 | 61.6 | 5 |
| iframes_hsv | mean_frames | 0.30 | 25.9 | 49.6 | 60.5 | 6 | 27.6 | 50.0 | 62.0 | 6 |
| iframes_hsv | mean_segments | 0.30 | 25.9 | 49.6 | 60.5 | 6 | 27.6 | 50.0 | 62.0 | 6 |
| iframes_hsv | max_segments | 0.15 | 24.5 | 46.8 | 58.4 | 6 | 27.9 | 50.1 | 61.7 | 5 |

Solo parlato (tutte le query): R@1 10.9, R@5 17.4, R@10 21.5, MdR 246.
Solo parlato (query con clip parlata): R@1 16.5, R@5 26.3, R@10 32.5.

## Costo delle strategie di selezione dei fotogrammi (test)

| Strategia | Fotogrammi decodificati/clip | Fotogrammi codificati/clip | Segmenti/clip | Clip con integrazione uniforme | Tempo selezione (s) | Tempo CLIP (s) |
|---|---|---|---|---|---|---|
| uniform | 387.5 | 8.0 | 3.32 | 0 | 27.2 | 40.2 |
| iframes | 92.6 | 4.45 | 3.23 | 309 | 15.1 | 23.5 |
| iframes_hsv | 92.6 | 3.23 | 3.23 | 309 | 15.1 | 18.8 |
