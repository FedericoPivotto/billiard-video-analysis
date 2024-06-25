# Goal
Identificare e tracciare oggetti di interesse per generare informazioni semantiche:
 - Monitoraggio del campo di gioco
 - Punteggio automatizzato (punteggio per palle piene e mezze)
 - Ricostruzione traiettorie palle dopo un colpo (un video registra un colpo)
 - Monitoraggio posizione palle tramite 2D top-view minimap per ogni frame del video in ingresso

Per ogni frame il sistema di computer vision deve:
 1. BOUNDING BOXES: Riconoscere e localizzare palle nel campo di gioco dal punto vista 2D distinguendole in base alla categoria (1:white "cue ball", 2:black "8-ball", 3:balls solid colors, 4:balls stripes): creazione delle bounding boxex
 2. BORDER IDENTIFICATION: Identificare bordi del campo di gioco (solo campo verde, accettabile anche se bordo tavolo)
 3. SEGMENTATION: segmentazione nelle categorie (1:white "cue ball", 2:black "8-ball", 3:balls solid colors, 4:balls stripes, 5:playing field) + (FORSE anche segmentazione della categoria 0:background ossia esterno del tavolo)
 4. 2D TOP-VIEW VISUALIZATION MAP: aggiornare posizioni palline e tutte le loro traiettorie di moviemnto nella visualizzazione 2D, colorando le palline nei colori delle due categorie (solid e stripes)

Robustezza sistema:
 - Deve ignorare qualsiasi persona intorno al tavolo di gioco (occlusions)
 - Robusto a vari colori del tavolo di gioco
 - Diverso numero di palline (stripes e solid)
 - Robusto e diversi punti di vista della camera che riprende

Semplificazioni possibili:
 - Per il tavolo di gioco meglio il bordo interno, ma va bene anche il bordo del tavolo di legno
 - Assumi che in un video la posizione della camera non cambia
 - Per tracciare le traiettorie delle palline dopo un colpo si puo usare tecniche di tracciamento di OpenCV (Tracking API: https://docs.opencv.org/3.4/d9/df8/group__tracking.html) tra frame consecutivi

Test della robustezza del nostro sitema usa il dataset con annotazioni: https://drive.google.com/drive/folders/1dzNrhDpc2DXRqmQgbO5l2WMjzfhMdxVn?usp=sharing

La cartella di ciascun video nela dataset ha:
 - masks: maschera di segmentazione del campo di gioco escluso tutto il resto (esterno del campo di gioco, persone e palline) per il primo e ultimo frame; per la loro costruzione si associa alle categorie il seguenti colori RGB nella nostra segmentazione (0: (128,128,128) 1: (255, 255, 255) 2: (0,0,0) 3: (0,0,255) 4: (255,0,0,) 5: (0,255,0)), il cui risultato andra poi convertito in scala di grigi (aka grayscale)
 - frames: primo e ultime frame del video (easy)
 - bounding_boxes: c'è un file di testo per il primo e ultimo frame del video; ciascun file di testo contiene una riga per pallina in cui sono riportati 5 parametri [x, y, width, height, ball category ID], dove (x,y) è la posizione dell'angolo in alto a sinistra della bounding box della pallina (il sistema degli assi ha origine nell'angolo in alto a sinistra dell'immagine, con asse x orizzontale da sinsitra a destra e asse y verticale dall'alto al basso); height è l'altezza della boundign box; width è la larghezza della boundign box; e l'ID identifica la categoria palla (solid o stripe); in questi file di testo ci sono solo categorie di palline (1, 2, 3, 4)

La cartella di ciascun video ha nomi standard per sotto-cartelle e file (vedi README dataset).

# Metriche
Calcolare metriche per valutare il nostro sistema:
 - Per localizzazione palle, utilizzare mean Average Precision (mAP) calcolata alla IoU threshold 0.5: qui dobbiamo utilizzare i file di testo nella cartella bounding_boxes (https://learnopencv.com/mean-average-precision-map-object-detection-model-evaluation-metric/)
 - Per segmentare palle e campo di gioco dobbiamo calcolare la mean Intersection over Union (mIoU) metric che è la media della IoU calcolata per ogni classe (0:background, 1:white "cue ball", 2:black "8-ball", 3:balls solid colors, 4:balls stripes, 5:playing field) (https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2)

# Eight Ball
15 balls (1-7 solid color, 8 solid black, 9-15 stripes).

Il gioco finisce quando tutte le palline (solid o stripes) sono finite in buca, e la 8 per ultima.