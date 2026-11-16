# Risultati

Test: 1000 clip di MSR-VTT 1k-A, di cui 0 con parlato rilevato. Configurazione scelta sulla validazione (300 clip): strategia `uniform`, aggregazione `max_segments`, peso del parlato w = 0.00.

## Recupero testo -> video (test)

| Strategia | Aggregazione | w | Visivo R@1 | R@5 | R@10 | MdR | Fusione R@1 | R@5 | R@10 | MdR |
|---|---|---|---|---|---|---|---|---|---|---|
| uniform | mean_frames | 0.00 | 30.2 | 51.8 | 62.1 | 5 | 30.2 | 51.8 | 62.1 | 5 |
| uniform | mean_segments | 0.00 | 29.7 | 52.8 | 61.8 | 4 | 29.7 | 52.8 | 61.8 | 4 |
| uniform **(scelta)** | max_segments | 0.00 | 29.8 | 53.7 | 62.5 | 4 | 29.8 | 53.7 | 62.5 | 4 |
| iframes | mean_frames | 0.00 | 28.1 | 51.1 | 62.2 | 5 | 28.1 | 51.1 | 62.2 | 5 |
| iframes | mean_segments | 0.00 | 28.4 | 51.3 | 61.6 | 5 | 28.4 | 51.3 | 61.6 | 5 |
| iframes | max_segments | 0.00 | 27.4 | 50.9 | 62.2 | 5 | 27.4 | 50.9 | 62.2 | 5 |
| iframes_hsv | mean_frames | 0.00 | 27.3 | 50.9 | 60.2 | 5 | 27.3 | 50.9 | 60.2 | 5 |
| iframes_hsv | mean_segments | 0.00 | 27.3 | 50.9 | 60.2 | 5 | 27.3 | 50.9 | 60.2 | 5 |
| iframes_hsv | max_segments | 0.00 | 25.8 | 50.9 | 60.8 | 5 | 25.8 | 50.9 | 60.8 | 5 |

Solo parlato (tutte le query): R@1 0.0, R@5 0.0, R@10 0.0, MdR 1000.

## Costo delle strategie di selezione dei fotogrammi (test)

| Strategia | Fotogrammi decodificati/clip | Fotogrammi codificati/clip | Segmenti/clip | Clip con integrazione uniforme | Tempo selezione (s) | Tempo CLIP (s) |
|---|---|---|---|---|---|---|
| uniform | 45.6 | 8.0 | 3.48 | 0 | 13.0 | 37.8 |
| iframes | 21.8 | 4.64 | 3.38 | 434 | 9.7 | 23.0 |
| iframes_hsv | 21.8 | 3.38 | 3.38 | 434 | 9.5 | 17.7 |
