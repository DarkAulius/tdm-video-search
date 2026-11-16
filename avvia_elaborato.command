#!/bin/bash
# Esegue tutta la pipeline dell'elaborato con un doppio clic dal Finder.
# 1. crea l'ambiente virtuale e installa le dipendenze
# 2. prova veloce su 50 clip di test e 30 di validazione
# 3. se la prova va a buon fine, esecuzione completa (1000 + 300 clip)
# L'output viene salvato anche in logs/. Se un passo fallisce, lo script resta in attesa
# (fino a 3 ore) che venga creato il file logs/riprova dopo una correzione, poi riprova
# lo stesso passo. Per interrompere basta chiudere la finestra.

set -o pipefail
cd "$(dirname "$0")" || exit 1
mkdir -p logs

# Si preferisce Python 3.12, poi 3.11, 3.10, 3.13; in ultima istanza python3.
PY=""
for candidate in python3.12 python3.11 python3.10 python3.13 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then PY="$candidate"; break; fi
done
if [ -z "$PY" ]; then
    echo "Python 3 non trovato. Installalo (es. brew install python@3.12) e riprova." | tee logs/errore.log
    read -r -p "Premi Invio per chiudere"
    exit 1
fi
echo "Uso $PY: $($PY --version)" | tee logs/ambiente.log
[ -d .venv ] || "$PY" -m venv .venv || { echo "Creazione dell'ambiente virtuale fallita" | tee -a logs/ambiente.log; exit 1; }
source .venv/bin/activate

install_deps() {
    echo "== Installazione dipendenze ==" | tee -a logs/ambiente.log
    python -m pip install --upgrade pip 2>&1 | tail -1 | tee -a logs/ambiente.log
    python -m pip install -r requirements.txt 2>&1 | tee -a logs/ambiente.log
}

wait_for_fix() {
    echo "In attesa ($1) ($(date +%H:%M))" | tee -a logs/stato.txt
    rm -f logs/riprova
    for _ in $(seq 1 1080); do          # 1080 x 10 s = 3 ore
        if [ -f logs/riprova ]; then
            rm -f logs/riprova
            echo "Riprovo '$1' ($(date +%H:%M))" | tee -a logs/stato.txt
            return 0
        fi
        sleep 10
    done
    echo "Nessuna correzione entro 3 ore: chiudo." | tee -a logs/stato.txt
    exit 1
}

# Esegue un passo e, se fallisce, aspetta la correzione e lo ripete.
run_step() {
    local name="$1"; shift
    while true; do
        echo "== $name ($(date +%H:%M)) ==" | tee -a logs/stato.txt
        if "$@" 2>&1 | tee "logs/$name.log"; then return 0; fi
        echo "ERRORE in '$name'" | tee -a logs/stato.txt
        wait_for_fix "correzione di $name"
        install_deps >/dev/null   # nel caso la correzione aggiunga dipendenze
    done
}

run_all() {
    run_step prova_veloce python -u main.py all --n-test 50 --n-val 30 --results-dir results_prova
    run_step completa python -u main.py all
    echo "COMPLETATO ($(date +%H:%M))" | tee -a logs/stato.txt
}

# Tutto il programma principale sta in una funzione: bash la legge per intero prima di
# eseguirla, quindi modificare questo file mentre gira non crea problemi.
main() {
    run_step dipendenze install_deps
    run_all
    # Dopo la fine resta in ascolto: un nuovo logs/riprova rilancia la pipeline (es. dopo una
    # modifica al codice). Senza segnali, si chiude da solo dopo 3 ore.
    while true; do
        wait_for_fix "nuova esecuzione"
        install_deps >/dev/null
        run_all
    done
}

main; exit
