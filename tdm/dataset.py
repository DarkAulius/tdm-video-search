"""Caricamento di MSR-VTT da Hugging Face: metadati (didascalie, categorie) e file video.

- test: lo split standard "1k-A" (1000 clip, una didascalia per clip usata come query).
- val: un campione di clip del training, disgiunto dal test, usato solo per tarare i parametri.
"""

from __future__ import annotations

import json
import random
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import SEED, Config

# Categorie di MSR-VTT nell'ordine del file category.txt del dataset (id 0-19).
# Derivano dalla categoria YouTube scelta da chi ha caricato il video, quindi sono rumorose.
CATEGORIES = [
    "music", "people", "gaming", "sports/actions", "news/events/politics",
    "education", "tv shows", "movie/comedy", "animation", "vehicles/autos",
    "howto", "travel", "science/technology", "animals/pets", "kids/family",
    "documentary", "food/drink", "cooking", "beauty/fashion", "advertisement",
]


@dataclass
class Clip:
    video_id: str
    caption: str      # didascalia usata come query testuale
    category: int     # indice in CATEGORIES
    url: str          # video YouTube di origine
    start: float      # inizio della clip nel video di origine (s)
    end: float        # fine della clip nel video di origine (s)

    @property
    def category_name(self) -> str:
        return CATEGORIES[self.category] if 0 <= self.category < len(CATEGORIES) else "unknown"


def _find_config(repo: str, prefix: str) -> str:
    """Restituisce il nome della configurazione HF che inizia con `prefix` (es. 'test', 'train')."""
    from datasets import get_dataset_config_names

    names = sorted(get_dataset_config_names(repo))
    matches = [n for n in names if n.startswith(prefix)]
    if not matches:
        raise RuntimeError(f"Nessuna configurazione '{prefix}*' in {repo}: {names}")
    # Tra più configurazioni di training si preferisce la più grande (es. train_9k).
    return matches[-1]


def _numeric_id(video_id: str) -> int:
    """'video7021' -> 7021."""
    digits = "".join(ch for ch in video_id if ch.isdigit())
    return int(digits) if digits else -1


def _row_to_clip(row: dict, caption: str) -> Clip:
    return Clip(
        video_id=row["video_id"],
        caption=caption,
        category=int(row["category"]),
        url=row.get("url", ""),
        start=float(row.get("start time", 0.0)),
        end=float(row.get("end time", 0.0)),
    )


def _load_pool(cfg: Config, split: str) -> list[Clip]:
    """Elenco completo delle clip candidate per lo split, salvato in JSON dopo il primo download.

    - test: le 1000 clip di 1k-A nell'ordine del dataset;
    - val: le clip del training non presenti nel test e con id >= cfg.val_min_id, mescolate
      con seme fisso, ciascuna con una didascalia scelta a caso tra le sue (circa 20).
    """
    name = f"{split}_pool.json" if split == "test" else f"val_pool_min{cfg.val_min_id}.json"
    local = cfg.data_dir / name
    if local.exists():
        return [Clip(**c) for c in json.loads(local.read_text())]

    from datasets import load_dataset

    if split == "test":
        ds = load_dataset(cfg.hf_repo, _find_config(cfg.hf_repo, "test"), split="test")
        pool = []
        for row in ds:
            caption = row["caption"]
            if isinstance(caption, list):  # per sicurezza: in 1k-A c'è una sola didascalia
                caption = caption[0]
            pool.append(_row_to_clip(row, caption))
    elif split == "val":
        test_ids = {c.video_id for c in _load_pool(cfg, "test")}
        ds = load_dataset(cfg.hf_repo, _find_config(cfg.hf_repo, "train"), split="train")
        # Raggruppa le didascalie per video: il dataset può avere una riga per didascalia
        # oppure una riga per video con la lista delle didascalie.
        by_video: dict[str, tuple[dict, list[str]]] = {}
        for row in ds:
            if row["video_id"] in test_ids or _numeric_id(row["video_id"]) < cfg.val_min_id:
                continue
            caps = row["caption"] if isinstance(row["caption"], list) else [row["caption"]]
            by_video.setdefault(row["video_id"], (row, []))[1].extend(caps)
        video_ids = sorted(by_video)
        random.Random(SEED).shuffle(video_ids)
        pool = []
        for vid in video_ids:
            row, caps = by_video[vid]
            caption = random.Random(f"{SEED}-{vid}").choice(caps)  # scelta riproducibile per video
            pool.append(_row_to_clip(row, caption))
    else:
        raise ValueError(f"Split sconosciuto: {split}")

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    local.write_text(json.dumps([asdict(c) for c in pool], indent=1))
    return pool


def load_split(cfg: Config, split: str) -> list[Clip]:
    """Le prime n clip dello split (n = cfg.n_test o cfg.n_val)."""
    n = cfg.n_test if split == "test" else cfg.n_val
    return _load_pool(cfg, split)[:n]


def video_path(cfg: Config, clip: Clip) -> Path:
    return cfg.video_dir / f"{clip.video_id}.mp4"


def ensure_videos(cfg: Config, clips: list[Clip]) -> None:
    """Scarica i video mancanti dalla sorgente configurata (archivio .zip via URL o
    repository Hugging Face). Dagli archivi si estraggono solo i video che servono."""
    needed = {f"{c.video_id}.mp4" for c in clips if not video_path(cfg, c).exists()}
    for source in cfg.video_sources:
        if not needed:
            break
        try:
            if source.startswith("http"):
                _from_remote_zip(cfg, source, needed)
            else:
                _from_hf_repo(cfg, source, needed)
        except Exception as exc:  # sorgente non raggiungibile o non adatta: si prova la successiva
            print(f"Sorgente {source} scartata: {type(exc).__name__}: {exc}")

    missing = [c.video_id for c in clips if not video_path(cfg, c).exists()]
    if missing:
        print(f"Attenzione: {len(missing)} video non trovati (es. {missing[:3]}); verranno saltati.")


def _extract_members(cfg: Config, zf: zipfile.ZipFile, needed: set[str]) -> None:
    """Estrae dallo zip i video richiesti, nell'ordine in cui stanno nell'archivio
    (letture in avanti: con un archivio remoto si evitano salti inutili)."""
    members = [i for i in zf.infolist() if Path(i.filename).name in needed]
    members.sort(key=lambda i: i.header_offset)
    for k, info in enumerate(members, start=1):
        name = Path(info.filename).name   # i file "._video..." di macOS non coincidono
        if name not in needed:            # eventuali duplicati nell'archivio
            continue
        (cfg.video_dir / name).write_bytes(zf.read(info))
        needed.discard(name)
        if k % 100 == 0 or k == len(members):
            print(f"  estratti {k}/{len(members)} video")


def _fetch_member(session, url: str, info: zipfile.ZipInfo) -> bytes:
    """Scarica un solo file dell'archivio remoto con una richiesta HTTP Range.

    Ogni file nello zip è preceduto da un'intestazione locale di 30 byte seguita dal nome e
    da un campo "extra" di lunghezza variabile: si chiede un margine di 1 KB per coprirli,
    si leggono le due lunghezze dall'intestazione e si ritaglia il contenuto.
    """
    import struct
    import zlib

    name_len = len(info.filename.encode("utf-8"))
    start = info.header_offset
    end = start + 30 + name_len + 1024 + info.compress_size
    r = session.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
    r.raise_for_status()
    if r.status_code != 206:
        raise RuntimeError("il server non rispetta le richieste Range")
    blob = r.content
    if blob[:4] != b"PK\x03\x04":
        raise RuntimeError(f"intestazione locale non valida per {info.filename}")
    n, m = struct.unpack("<HH", blob[26:30])   # lunghezza del nome e del campo extra
    data = blob[30 + n + m : 30 + n + m + info.compress_size]
    if info.compress_type == zipfile.ZIP_STORED:
        return data
    if info.compress_type == zipfile.ZIP_DEFLATED:
        return zlib.decompress(data, -15)        # deflate "grezzo", senza intestazione zlib
    raise RuntimeError(f"compressione {info.compress_type} non supportata")


def _from_remote_zip(cfg: Config, url: str, needed: set[str], workers: int = 24) -> None:
    """Estrae i video da un archivio remoto senza scaricarlo tutto.

    1. si legge solo l'indice (directory centrale) dello zip, che sta in fondo al file;
    2. ogni video si scarica con una richiesta HTTP Range, con `workers` richieste in
       parallelo: il server limita la banda per connessione, quindi più connessioni
       aumentano la velocità complessiva.
    Se il server non supporta le richieste Range si scarica l'archivio completo.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import fsspec
    import requests

    try:
        print(f"Leggo l'indice di {url} ...")
        with fsspec.open(url, "rb", block_size=2**20) as remote, zipfile.ZipFile(remote) as zf:
            by_name = {}
            for info in zf.infolist():
                name = Path(info.filename).name   # i file "._video..." di macOS non coincidono
                if name in needed and name not in by_name:
                    by_name[name] = info
        print(f"  {len(by_name)} video da scaricare con {workers} connessioni in parallelo")
        lock = threading.Lock()
        local = threading.local()
        done = [0]

        def work(item):
            name, info = item
            if not hasattr(local, "session"):
                local.session = requests.Session()
            data = _fetch_member(local.session, url, info)
            (cfg.video_dir / name).write_bytes(data)
            with lock:
                needed.discard(name)
                done[0] += 1
                if done[0] % 100 == 0 or done[0] == len(by_name):
                    print(f"  scaricati {done[0]}/{len(by_name)} video")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in [pool.submit(work, item) for item in by_name.items()]:
                try:
                    future.result()
                except Exception as exc:   # un video non scaricato non ferma gli altri
                    print(f"  errore: {type(exc).__name__}: {exc}")
        return
    except Exception as exc:
        print(f"Lettura parziale non riuscita ({type(exc).__name__}: {exc}); scarico l'archivio completo.")

    local = cfg.data_dir / Path(url).name
    if not local.exists():
        _download(url, local)
    with zipfile.ZipFile(local) as zf:
        _extract_members(cfg, zf, needed)


def _download(url: str, dest: Path) -> None:
    import requests

    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        done = last = 0
        with open(tmp, "wb") as out:
            for chunk in r.iter_content(chunk_size=8 * 2**20):
                out.write(chunk)
                done += len(chunk)
                if done - last >= 500 * 2**20:
                    print(f"  scaricati {done / 2**30:.1f} / {total / 2**30:.1f} GB")
                    last = done
    tmp.rename(dest)


def _from_hf_repo(cfg: Config, repo: str, needed: set[str]) -> None:
    """Repository Hugging Face con file .mp4 sciolti oppure archivi .zip: i nomi dei file
    vengono letti dal repository invece di essere assunti."""
    from huggingface_hub import HfApi, hf_hub_download

    files = HfApi().list_repo_files(repo, repo_type="dataset")
    mp4_files = {Path(f).name: f for f in files if f.lower().endswith(".mp4")}
    tars = [f for f in files if f.lower().endswith(".tar")]

    if tars and not mp4_files:
        for tar_name in tars:
            if needed:
                _from_hf_tar(cfg, repo, tar_name, needed)
    elif mp4_files:
        for name in sorted(needed):
            if name in mp4_files:
                src = hf_hub_download(repo, mp4_files[name], repo_type="dataset")
                (cfg.video_dir / name).write_bytes(Path(src).read_bytes())
    else:
        zips = [f for f in files if f.lower().endswith(".zip")]
        # Prima gli archivi con "test" nel nome (contengono le clip di 1k-A), poi gli altri.
        zips.sort(key=lambda f: ("test" not in f.lower(), "video" not in f.lower(), f))
        for zip_name in zips:
            if not needed:
                break
            print(f"Scarico {repo}/{zip_name} (la prima volta può richiedere diversi minuti)...")
            archive = hf_hub_download(repo, zip_name, repo_type="dataset")
            with zipfile.ZipFile(archive) as zf:
                _extract_members(cfg, zf, needed)


def _has_audio(path: Path) -> bool:
    import av

    try:
        with av.open(str(path)) as container:
            return bool(container.streams.audio)
    except Exception:
        return False


def _from_hf_tar(cfg: Config, repo: str, tar_name: str, needed: set[str]) -> None:
    """Legge un archivio .tar di Hugging Face in streaming (dall'inizio alla fine, senza
    salvarlo su disco) ed estrae i video richiesti. Dopo i primi 5 video controlla che
    abbiano la traccia audio: se nessuno ce l'ha la sorgente viene scartata subito."""
    import tarfile

    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    path = f"datasets/{repo}/{tar_name}"
    total = fs.size(path)
    print(f"Leggo in streaming {repo}/{tar_name} ({total / 2**30:.1f} GB)...")
    extracted: list[Path] = []
    with fs.open(path, "rb", block_size=16 * 2**20) as remote, tarfile.open(fileobj=remote, mode="r|") as tar:
        for member in tar:
            name = Path(member.name).name
            if not member.isfile() or name not in needed:
                continue
            dest = cfg.video_dir / name
            dest.write_bytes(tar.extractfile(member).read())
            needed.discard(name)
            extracted.append(dest)
            if len(extracted) == 5 and not any(_has_audio(p) for p in extracted):
                for p in extracted:   # non mescolare video senza audio con quelli buoni
                    p.unlink()
                    needed.add(p.name)
                raise RuntimeError("i video di questa sorgente non hanno traccia audio")
            if len(extracted) % 100 == 0:
                print(f"  estratti {len(extracted)} video ({remote.tell() / total:.0%} dell'archivio letto)")
            if not needed:
                break
    print(f"  estratti {len(extracted)} video da {repo}")


def media_summary(cfg: Config, clips: list[Clip], sample: int = 100) -> dict:
    """Controlla su un campione di video la presenza della traccia audio e il formato video."""
    import av

    paths = [video_path(cfg, c) for c in clips if video_path(cfg, c).exists()][:sample]
    with_audio, fps = 0, []
    for p in paths:
        try:
            with av.open(str(p)) as container:
                with_audio += bool(container.streams.audio)
                fps.append(float(container.streams.video[0].average_rate or 0))
        except Exception:
            continue
    return {"checked": len(paths), "with_audio": with_audio,
            "median_fps": float(sorted(fps)[len(fps) // 2]) if fps else 0.0}


def available(cfg: Config, clips: list[Clip]) -> list[Clip]:
    """Tiene solo le clip il cui file video è presente su disco."""
    return [c for c in clips if video_path(cfg, c).exists()]
