# Ottimizzazione Spazi Espositivi MVP

Questa applicazione dimostra come il machine learning può essere utilizzato per ottimizzare l'allocazione degli spazi espositivi al fine di massimizzare i ricavi. Fornisce un'interfaccia interattiva per esplorare la modellazione della domanda degli espositori e l'ottimizzazione degli spazi.

## Funzionalità

- **Generazione/Caricamento Dati**: Utilizza la generazione di dati campione integrata o carica i tuoi dati storici
- **Stima Parametri di Domanda**: Calcolo automatico dei parametri di domanda per ogni tipo di espositore ed evento
- **Ottimizzazione Spazi**: Ottimizzazione basata su ML per massimizzare i ricavi rispettando i vincoli di spazio
- **Visualizzazione Interattiva**: Esplorazione visiva dei risultati di ottimizzazione e dei dati storici
- **Regolazione dei Parametri**: Regola i parametri di ottimizzazione per esplorare diversi scenari

## Per Iniziare

### Prerequisiti

- Python 3.7 o superiore
- Pacchetti richiesti come elencati in `requirements.txt`

### Installazione

1. Clona questo repository
2. Installa i pacchetti richiesti:

```bash
pip install -r requirements.txt
```

### Esecuzione dell'Applicazione

Per avviare l'app Streamlit:

```bash
streamlit run app/app.py
```

## Come Utilizzare

1. **Genera Dati**: Utilizza la barra laterale per generare dati espositivi di esempio o carica il tuo file CSV
2. **Seleziona un Evento**: Scegli un evento da ottimizzare dal menu a tendina
3. **Configura i Parametri**: Regola i parametri di ottimizzazione nella barra laterale
4. **Esegui l'Ottimizzazione**: Fai clic su "Ottimizza Allocazione Spazi" per calcolare l'allocazione ottimale
5. **Esplora i Risultati**: Visualizza i risultati nella scheda "Risultati Ottimizzazione"
6. **Analizza i Dati**: Utilizza la scheda "Esploratore Dati" per esaminare i dati storici e i parametri di domanda

## Comprensione del Modello

L'applicazione utilizza:

1. **Modellazione della Domanda**: Stima come il prezzo è correlato alla quantità e alla dimensione dello stand per ogni tipo di espositore
2. **Ottimizzazione con Vincoli**: Trova l'allocazione ottimale di stand che massimizza i ricavi rispettando i vincoli
3. **Stima dei Parametri**: Calcola i parametri di domanda (α, β₁, β₂) dai dati storici

La funzione di domanda inversa utilizzata è:

```
Prezzo = α + β₁ * Quantità + β₂ * DimensioneStand
```

Dove:
- α è il prezzo base
- β₁ è l'effetto quantità (come cambia il prezzo con più espositori)
- β₂ è l'effetto dimensione (come cambia il prezzo con la dimensione dello stand)

## Formato dei Dati

Se vuoi caricare i tuoi dati, dovrebbero essere in un CSV con le seguenti colonne:
- Event_ID: Identificatore unico per l'evento
- Event_Name: Nome dell'evento
- Year: Anno in cui si è svolto l'evento
- Location: Luogo dell'evento
- Fee_CHF: La tariffa pagata dall'espositore (in CHF)
- Exhibitor_Type: Tipo di espositore (es. Small_Business, Premium_Brand)
- Stand_Size: Categoria di dimensione (Small, Medium, Large)
- Stand_m2: Dimensione in metri quadrati
- Total_Event_Space_m2: Spazio totale disponibile per l'evento 