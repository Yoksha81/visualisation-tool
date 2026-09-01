# VizTool

VizTool je alat za interaktivnu vizuelizaciju arhitekture PyTorch modela.

Model se prikazuje hijerarhijski: na početku se vide veći moduli, a njihovim otvaranjem moguće je doći do manjih PyTorch modula i elementarnih računskih operacija.

Alat koristi `torch.fx.symbolic_trace` kao primarni način analize modela. Ako symbolic tracing nije dovoljan, moguće je proslediti reprezentativne ulaze i koristiti `torch.export` kao rezervni način analize.

**Autor:** Jovan Boksan, EE156/2022

## Mogućnosti

- prikaz hijerarhije `nn.Module` objekata;
- prikaz stvarnih dataflow veza između operacija;
- prikaz grananja i residual/skip veza;
- prikaz funkcijskih operacija kao što su `Add`, `MatMul`, `Cat`, `Transpose` i druge;
- interaktivno otvaranje i zatvaranje modula;
- podrška za modele sa jednim ili više ulaza;
- `torch.export` fallback za modele koje `torch.fx` ne može da obradi;
- composite prikaz za složene modele kao što je SAM2;
- high-level SAM2 pregled i detaljna analiza njegovih glavnih komponenti.

## Instalacija

Preporučeno je korišćenje posebnog Python okruženja.

```bash
git clone https://github.com/Yoksha81/visualisation-tool.git
cd visualisation-tool
pip install -r requirements.txt
```

Za osnovni rad alata dovoljne su biblioteke navedene u `requirements.txt`.

## Brzi primer

Za standardne `torchvision` modele nije potrebno ručno praviti ulaz ako `torch.fx.symbolic_trace` može uspešno da analizira model.

```python
from torchvision.models import resnet18, ResNet18_Weights
from model_visualizer import visualize

model = resnet18(
    weights=ResNet18_Weights.DEFAULT
)

visualize(model)
```

Gotov primer nalazi se u:

```text
eksperimenti/torchvision_demo.py
```

Pokreće se iz root foldera projekta:

```bash
python -m eksperimenti.torchvision_demo
```

Primer koristi prethodno obučeni `ResNet18` iz `torchvision.models`.

## Model kome je potreban example input

Ako `torch.fx.symbolic_trace` ne može da obradi model, korisnik može da prosledi konkretan reprezentativni ulaz. Tada alat može da pokuša analizu pomoću `torch.export`.

Za jedan ulaz:

```python
visualize(
    model,
    example_inputs=x
)
```

Za više pozicionih argumenata:

```python
visualize(
    model,
    example_inputs=(x1, x2)
)
```

Za keyword argumente:

```python
visualize(
    model,
    example_inputs=(x,),
    example_kwargs={
        "mask": mask
    }
)
```

Ako su example input-i prosleđeni, alat po default-u prvo proverava da li model može da se izvrši sa njima.

Jednostavan primer ovog načina rada nalazi se u:

```text
eksperimenti/example_input_demo.py
```

i može se pokrenuti:

```bash
python -m eksperimenti.example_input_demo
```

## Interakcija

U standardnom prikazu:

- levi klik na plavi modul otvara modul;
- desni klik unutar otvorenog modula zatvara najbliži otvoreni modul.

U composite/SAM2 prikazu dodatno postoje:

- dugme `Overview` za povratak na početni prikaz;
- `Esc` ili `Backspace` za povratak na overview.

Boje čvorova imaju sledeće značenje:

- zeleno — ulaz;
- crvenkasto — izlaz;
- plavo — modul koji može dalje da se otvori;
- sivo — terminalni PyTorch modul;
- žuto — elementarna računska operacija;
- ljubičasto — runtime ili eksterni čvor u high-level prikazu.

Legenda se automatski prikazuje ispod grafa.

## SAM2

VizTool je testiran i na zvaničnoj implementaciji modela Segment Anything 2.

Za demonstraciju je korišćena konfiguracija:

```text
configs/sam2/sam2_hiera_t.yaml
```

odnosno SAM2 model sa Hiera-Tiny image encoder-om.

Model se za vizuelizaciju instancira bez pretrained checkpoint-a, pošto vrednosti parametara nisu potrebne za analizu njegove arhitekture.

### Zašto SAM2 koristi composite prikaz?

SAM2 nije klasičan model sa jednim jednostavnim root `forward()` pozivom koji predstavlja kompletan video inference tok.

Njegove glavne komponente učestvuju u streaming obradi frejmova i koriste memoriju prethodnih frejmova. Zbog toga nije moguće samo jednim `symbolic_trace(model)` pozivom dobiti kompletan high-level SAM2 tok.

VizTool zato koristi composite režim:

1. high-level veze između glavnih SAM2 komponenti definišu se na osnovu poznate arhitekture modela;
2. svaka glavna komponenta se zatim zasebno analizira istim generičkim analyzerom koji se koristi i za ostale PyTorch modele.

Na taj način high-level pregled predstavlja funkcionalnu SAM2 arhitekturu, dok su unutrašnji computation graph-ovi komponenti automatski dobijeni analizom PyTorch modela.

### Instalacija SAM2

Potrebno je instalirati zvanični SAM2 repozitorijum u isto Python okruženje u kojem se koristi VizTool.

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```

Nakon toga se iz root foldera VizTool projekta SAM2 demonstracija može pokrenuti:

```bash
python -m eksperimenti.sam2_single_session
```

Primer se nalazi u:

```text
eksperimenti/sam2_single_session.py
```

### High-level SAM2 tok

High-level prikaz predstavlja jedan vremenski korak obrade video frejma:

```text
                              Memory Bank
                           (previous frames)
                                  |
                                  v
Input Frame -> Image Encoder -> Memory Attention -> Mask Decoder -> Mask Output
                   |                                  ^
                   |                                  |
                   |                           Prompt Encoder
                   |                                  ^
                   |                                  |
                   |                                Prompt
                   |
                   +----------------> Mask Decoder
                   |
                   +----------------> Memory Encoder
                                          ^
                                          |
                                    Mask prediction
                                          |
                                          v
                                  Memory Bank
                                     (updated)
```

Memory Bank je prikazan kao prethodna i ažurirana memorija da bi se jedan vremenski korak prikazao kao DAG.

Direktna veza iz `Image Encoder` ka `Mask Decoder` predstavlja korišćenje feature mapa visoke rezolucije.

`Memory Encoder` koristi feature-e slike i rezultat segmentacije za formiranje memorije koja se može koristiti pri obradi narednih frejmova.

Klikom na glavne blokove mogu se otvoriti njihovi computation graph-ovi:

- Image Encoder;
- Memory Attention;
- Prompt Encoder;
- Mask Decoder;
- Memory Encoder.

## Kako alat radi

Osnovni tok analize je:

```text
PyTorch model
    |
    v
torch.fx.symbolic_trace
    |
    | ako uspe
    v
ModelGraph

Ako FX ne uspe:
    |
    v
torch.export + example input
    |
    v
ModelGraph

ModelGraph
    |
    v
VisibleGraph
    |
    v
Renderer
    |
    v
Interaktivni prikaz
```

### ModelGraph

`ModelGraph` predstavlja kompletan rezultat analize modela.

Sadrži:

- operacije;
- module;
- veze između operacija;
- pripadnost operacija određenim modulima.

### VisibleGraph

`VisibleGraph` predstavlja samo deo modela koji je u tom trenutku prikazan korisniku.

Kada je modul zatvoren, njegove unutrašnje operacije predstavljene su jednim čvorom.

Kada korisnik otvori modul, prikazuju se njegovi child moduli ili njegove operacije.

Ovakav pristup omogućava da se veoma veliki modeli pregledaju postepeno, od većih arhitektonskih celina ka manjim jedinicama.

## Struktura projekta

```text
visualisation-tool/
├── model_visualizer/
│   ├── __init__.py
│   ├── analyzer.py
│   ├── graph.py
│   ├── interactive.py
│   ├── renderer.py
│   ├── runtime.py
│   ├── session.py
│   ├── view.py
│   └── visualizer.py
│
├── eksperimenti/
│   ├── torchvision_demo.py
│   ├── example_input_demo.py
│   └── sam2_single_session.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Uloga glavnih fajlova

### `analyzer.py`

Analizira PyTorch model i formira `ModelGraph`.

Prvo pokušava analizu pomoću `torch.fx.symbolic_trace`. Ako ona ne uspe i prosleđeni su example input-i, koristi `torch.export`.

### `graph.py`

Sadrži strukture podataka kojima se predstavljaju:

- operacije;
- moduli;
- veze;
- kompletan model graph;
- trenutno vidljiv graph.

### `view.py`

Određuje koji deo kompletnog grafa je trenutno vidljiv.

Ovde se nalazi logika za:

- otvaranje i zatvaranje modula;
- predstavljanje zatvorenog modula jednim čvorom;
- formiranje vidljivih veza;
- sprečavanje veštačkih ciklusa koji mogu nastati prilikom sažimanja modula.

### `renderer.py`

Zadužen je za vizuelni prikaz grafa.

Koristi NetworkX za rad sa grafom i Matplotlib za crtanje.

U njemu se nalaze:

- raspored čvorova;
- crtanje grana;
- prikaz skip veza;
- prikaz otvorenih grupa;
- boje čvorova;
- legenda.

### `interactive.py`

Obrađuje interakciju korisnika kod standardnog prikaza modela.

Levi klik otvara modul, a desni klik ga zatvara.

### `session.py`

Realizuje composite prikaz za složenije modele.

Omogućava:

- high-level overview;
- otvaranje pojedinačnih komponenti;
- povratak na overview;
- čuvanje expand/collapse stanja;
- keširanje već analiziranih komponenti.

### `runtime.py`

Sadrži pomoćne funkcije za obradu i proveru example input-a.

### `visualizer.py`

Sadrži glavni javni API alata.

Najjednostavniji način korišćenja je:

```python
visualize(model)
```

ili, kada su potrebni konkretni ulazi:

```python
visualize(
    model,
    example_inputs=x
)
```

## Ograničenja

- Nije moguće automatski odrediti validne dimenzije ulaza za proizvoljan `nn.Module`.
- `torch.fx` ne može da obradi svaku Python kontrolu toka.
- Za modele koje FX ne može da analizira potrebni su reprezentativni example input-i za `torch.export`.
- Veoma duboko otvoreni modeli mogu postati vizuelno gusti.
- Za složene sisteme bez jednog reprezentativnog root `forward()` toka, kao što je SAM2, high-level veze moraju biti poznate i deklarisane posebno.
- Alat je namenjen vizuelizaciji arhitekture i computation graph-a, a ne analizi vrednosti naučenih parametara modela.

## Korišćene tehnologije

- Python
- PyTorch
- `torch.fx`
- `torch.export`
- NetworkX
- Matplotlib

## Literatura i reference

- N. Ravi et al., *SAM 2: Segment Anything in Images and Videos*, 2024.
- Meta AI, zvanična implementacija modela Segment Anything 2.
- PyTorch dokumentacija za `torch.fx` i `torch.export`.
