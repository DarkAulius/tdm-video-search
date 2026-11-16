# Ricerca testuale in una collezione video con fusione di visivo e parlato

Elaborato di **Trattamento Dati Multimediali**, Università Telematica San Raffaele Roma.
Studente: Vincenzo Chianese (matricola 0322600081).

Data una query in linguaggio naturale (es. *"a person is cooking on stage"*), il sistema
restituisce le clip più pertinenti di una collezione video e, per ciascuna, il segmento
temporale più rilevante come URI W3C Media Fragments (`.../video7362#t=4.20,9.80`).
La ricerca combina il contenuto visivo dei fotogrammi (CLIP) con il parlato estratto dalla
traccia audio (Whisper), e può essere filtrata con interrogazioni SPARQL su un knowledge
graph RDF della collezione.

## Pipeline

```
video .mp4 ──► selezione fotogrammi ──► segmentazione in scene ──► CLIP (immagini) ──► aggregazione per segmento ─┐
   │           (uniforme / fotogrammi I /    (istogrammi HSV,                                                   │
   │            fotogrammi I + filtro HSV)    distanza di Bhattacharyya)                                         ├─► fusione tardiva ─► ranking
   │                                                                                                             │   (z-score, peso w
   └─► audio 16 kHz mono ──► controllo energia ──► Whisper (traduzione in inglese) ──► filtro segmenti ──► MiniLM ─┘    tarato su validazione)
                                                                                                                          │
metadati MSR-VTT + scene + trascrizioni ──► knowledge graph RDF (ma: + Media Fragments) ──► filtri SPARQL ───────────────┘
```

| Fase | Modulo | Cosa fa |
|---|---|---|
| Dati | `tdm/dataset.py` | MSR-VTT: metadati da Hugging Face, video originali con audio dall'archivio di Frozen in Time; test 1k-A (1000 clip), validazione (300 clip disgiunte dal test) |
| Fotogrammi | `tdm/keyframes.py` | tre strategie di selezione, istogrammi HSV, segmentazione in scene |
| Audio | `tdm/audio.py` | estrazione con PyAV/FFmpeg, ricampionamento a 16 kHz mono, livello RMS in dBFS |
| Parlato | `tdm/asr.py` | Whisper con filtro dei segmenti senza parlato o allucinati |
| Embedding | `tdm/encoders.py` | CLIP ViT-B/32 per immagini e query, MiniLM per le trascrizioni |
| Indici | `tdm/index.py` | indice visivo (fotogrammi, segmenti, clip) e indice del parlato; tre modi di aggregazione |
| Fusione | `tdm/fusion.py` | standardizzazione dei punteggi e fusione tardiva con peso tarato |
| Metriche | `tdm/metrics.py` | Recall@1/5/10, Median Rank, Mean Rank |
| Esperimenti | `tdm/experiments.py` | taratura sulla validazione, ablazioni sul test, tabelle |
| Knowledge graph | `tdm/kg.py` | grafo RDF con W3C Ontology for Media Resources e Media Fragments, query SPARQL |
| Ricerca | `tdm/search.py` | filtri SPARQL + ranking semantico + localizzazione del segmento |

## Contributo originale rispetto ai modelli pre-addestrati

CLIP, Whisper e MiniLM sono usati come estrattori di embedding senza riaddestramento.
Il lavoro dell'elaborato è tutto ciò che sta intorno:

1. **Selezione dei fotogrammi guidata dalla codifica video.** Si decodificano solo i fotogrammi I
   del GOP (`skip_frame = NONKEY`): non dipendono da altri fotogrammi, quindi il decoder salta
   tutti i P e B. Un rilevatore di cambi di scena sugli istogrammi HSV tiene un solo fotogramma
   per scena. Il confronto con il campionamento uniforme misura recall, fotogrammi decodificati,
   fotogrammi codificati e tempi.
2. **Aggregazione per segmento.** I fotogrammi sono raggruppati in scene; si confrontano media
   dei fotogrammi, media dei segmenti e segmento migliore. Il segmento migliore localizza anche
   il risultato nel tempo.
3. **Fusione e taratura.** I punteggi di CLIP e MiniLM hanno scale diverse: vengono standardizzati
   per query (z-score) e combinati come `z_visivo + w · z_parlato`, con le clip senza parlato a
   valore neutro. Il peso `w`, la strategia e l'aggregazione sono scelti sulla validazione.
4. **Knowledge graph.** Video, scene, fotogrammi chiave e trascrizioni sono descritti in RDF con la
   W3C Ontology for Media Resources; ogni segmento ha un URI Media Fragments. Le query SPARQL
   filtrano le clip per categoria, durata e presenza di parlato prima del ranking.

## Tecniche del corso applicate

| Tecnica | Dove | Perché |
|---|---|---|
| Struttura GOP, fotogrammi I/P/B | `keyframes.extract_iframes` | decodificare solo i fotogrammi I riduce il costo; l'encoder li inserisce ai cambi di scena |
| Spazi colore (HSV) | `keyframes.hsv_histogram` | separa tinta e saturazione dalla luminosità: il rilevatore reagisce ai cambi di contenuto, poco ai flash |
| Campionamento temporale | `keyframes.sample_uniform` | baseline di confronto con K fotogrammi equidistanti |
| Campionamento e quantizzazione audio | `audio.load_audio`, `audio.rms_dbfs` | 16 kHz (Nyquist a 8 kHz copre il parlato), mono, float32; soglia di silenzio in dBFS |
| Rappresentazione dei metadati | `kg.py` | RDF, W3C Ontology for Media Resources, W3C Media Fragments URI, SPARQL |

## Installazione

Serve Python 3.10–3.12. FFmpeg non va installato a parte: PyAV include le sue librerie.

```bash
cd tdm-video-search
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

La prima esecuzione scarica i metadati di MSR-VTT da Hugging Face, i video originali dall'archivio
distribuito dagli autori di Frozen in Time (circa 6 GB: se il server lo consente si leggono solo
i video necessari) e i pesi di CLIP, MiniLM e Whisper.

## Esecuzione

Su macOS basta un doppio clic su `avvia_elaborato.command`: crea l'ambiente, installa le
dipendenze, esegue la prova veloce e poi quella completa, salvando l'output in `logs/`.
In alternativa, i comandi a mano.

Prova veloce su poche clip, per verificare che tutto funzioni (pochi minuti):

```bash
python main.py all --n-test 50 --n-val 30
```

Esecuzione completa (1000 clip di test, 300 di validazione):

```bash
python main.py all
```

`all` esegue in ordine i passi seguenti, che si possono lanciare anche uno alla volta:

```bash
python main.py prepare     # metadati e video in data/
python main.py visual      # selezione fotogrammi + CLIP per le 3 strategie -> cache/
python main.py speech      # audio + Whisper + MiniLM -> cache/ (riprende se interrotto)
python main.py evaluate    # taratura sulla validazione, valutazione sul test -> results/
python main.py kg          # knowledge graph RDF del test -> results/kg_test.ttl
```

Gli indici già calcolati vengono riutilizzati; `--force` li ricalcola. Con `--whisper-model small`
le trascrizioni sono più accurate ma circa tre volte più lente.

### Ricerca

```bash
python main.py search "a person is cooking on stage"
python main.py search "a man talks about cars" --category vehicles --speech yes
python main.py search "people dancing" --max-duration 15 --top-k 10
```

Formato dell'output, una riga per risultato:

```
<posizione>. <video_id>  punteggio <z-score fuso>  segmento <inizio>-<fine> s
   <URI Media Fragments del segmento>
   parlato: "<trascrizione del segmento, se presente>"
```

### Query SPARQL sul grafo

```bash
python main.py sparql queries/clip_per_categoria.rq
python main.py sparql queries/parlato_con_parola.rq
python main.py sparql queries/scene_per_clip.rq
```

## Risultati

Esecuzione completa su MSR-VTT 1k-A (1000 clip di test, 662 con parlato rilevato; 300 clip di
validazione). Tabelle complete in `results/summary.md`.

| Variante (test) | R@1 | R@5 | R@10 | MdR |
|---|---|---|---|---|
| Solo visivo, uniforme, media dei fotogrammi | 31,1 | 52,4 | 62,6 | 5 |
| Solo visivo, fotogrammi I + HSV, media | 25,9 | 49,6 | 60,5 | 6 |
| Solo parlato | 10,9 | 17,4 | 21,5 | 246 |
| Fusione, configurazione scelta in validazione (uniforme, segmento migliore, w = 0,15) | 30,9 | 55,9 | 66,2 | 4 |
| Fusione, uniforme, media dei fotogrammi (w = 0,25) | 33,1 | 56,1 | 66,4 | 4 |
| Solo clip con parlato: visivo → fusione (uniforme, media) | 28,1 → 35,8 | 49,1 → 57,3 | 60,0 → 67,8 | 6 → 3,5 |

| Selezione dei fotogrammi | Decodificati/clip | Codificati/clip | Tempo (1000 clip) |
|---|---|---|---|
| Uniforme (K = 8) | 387,5 | 8,0 | 67,4 s |
| Fotogrammi I | 92,6 | 4,45 | 38,6 s |
| Fotogrammi I + filtro HSV | 92,6 | 3,23 | 33,8 s |

`python main.py evaluate` scrive:

- `results/evaluation.json`: tutte le metriche, i pesi provati e le curve di taratura;
- `results/summary.md`: tabelle di recupero (solo visivo, solo parlato, fusione) e costi.

Il protocollo è quello standard di MSR-VTT 1k-A: ogni didascalia è una query, la clip corretta è
quella da cui proviene la didascalia, la ricerca avviene su tutte le 1000 clip. I parametri sono
scelti solo sulla validazione.

## Riproducibilità

- seme fisso (`tdm/config.py`, `SEED = 42`) per il campione di validazione, la scelta della
  didascalia e torch;
- decodifica greedy di Whisper (`temperature = 0`), quindi trascrizioni deterministiche;
- tutti i parametri sono in `tdm/config.py`, con il motivo di ogni scelta.

## Test

```bash
pytest -q
```

I test generano video sintetici con PyAV (scene di colori diversi, GOP noto, tono audio) e usano
encoder finti, quindi non scaricano modelli. Verificano fotogrammi I, campionamento uniforme,
segmentazione HSV, ricampionamento audio, filtro di Whisper, metriche, fusione, grafo RDF,
ricerca e collegamento dei comandi di `main.py`.

## Struttura

```
main.py              comandi: prepare, visual, speech, evaluate, kg, search, sparql, all
tdm/                 moduli della pipeline (vedi tabella sopra)
queries/             esempi di query SPARQL
tests/               test automatici su dati sintetici
results/             metriche, tabelle e grafo RDF prodotti dalla valutazione
```

## Limiti noti

- I video di MSR-VTT pubblicati su Hugging Face (`friedrichor/MSR-VTT`) sono ricodificati a 3 fps
  e senza audio: per questo si usano i file originali. I risultati sul solo visivo con quella
  versione sono conservati in `results_3fps_senza_audio/` come confronto.

- Molte clip di MSR-VTT contengono solo musica: il parlato aiuta solo sulla parte di collezione
  in cui è presente, per questo le metriche sono riportate anche sul sottoinsieme con parlato.
- Le categorie di MSR-VTT derivano dalla categoria YouTube di origine e sono rumorose.
- La localizzazione del segmento è dimostrativa: MSR-VTT non ha annotazioni temporali, quindi la
  valutazione quantitativa è a livello di clip.
