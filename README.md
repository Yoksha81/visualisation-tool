# VizTool

VizTool je alat za interaktivnu vizuelizaciju arhitekture PyTorch modela. Model se prikazuje hijerarhijski: na početku se vide veći moduli, a levi klik otvara njihov sadržaj sve do PyTorch modula i elementarnih operacija.

Alat koristi `torch.fx.symbolic_trace` kao prvi način analize. Ako symbolic tracing nije dovoljan, moguće je proslediti reprezentativne ulaze i koristiti `torch.export` kao rezervni način analize.

## Mogućnosti

- prikaz hijerarhije `nn.Module` objekata;
- prikaz stvarnih dataflow veza između operacija;
- detekcija residual/skip veza;
- prikaz funkcijskih operacija kao što su `Add`, `MatMul`, `Cat`, `Transpose` i druge;
- interaktivno otvaranje i zatvaranje modula;
- podrška za modele sa jednim ili više ulaza;
- `torch.export` fallback za modele koje `torch.fx` ne može da obradi;
- composite prikaz za složene modele kao što je SAM2;
- high-level SAM2 pregled i detaljna analiza njegovih glavnih komponenti.

## Instalacija

Preporučeno je korišćenje posebnog Python okruženja.

```bash
git clone <URL_OVOG_REPOZITORIJUMA>
cd VizTool
pip install -r requirements.txt
```

Za osnovni rad alata dovoljne su biblioteke iz `requirements.txt`.

## Brzi primer

Za standardne torchvision modele nije potrebno ručno praviti ulaz ako `torch.fx.symbolic_trace` može da analizira model.

```python
from torchvision.models import resnet18, ResNet18_Weights
from model_visualizer import visualize

model = resnet18(
    weights=ResNet18_Weights.DEFAULT
)

visualize(model)
```

Pokretanje gotovog primera:

```bash
python examples/torchvision_demo.py
```

Ovaj primer može da se koristi i za demonstraciju alata na prethodno obučenom modelu iz `torchvision.models`.

## Model kome je potreban example input

Ako `symbolic_trace` ne uspe, korisnik može da prosledi konkretan ulaz:

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

## Interakcija

U običnom prikazu:

- levi klik na plavi modul: otvaranje modula;
- desni klik unutar otvorenog modula: zatvaranje najbližeg otvorenog modula.

U composite/SAM2 prikazu dodatno postoje:

- dugme `Overview`;
- `Esc` ili `Backspace` za povratak na high-level prikaz.

Boje čvorova:

- zeleno — ulaz;
- crvenkasto — izlaz;
- plavo — modul koji može da se otvori;
- sivo — terminalni PyTorch modul;
- žuto — elementarna operacija;
- ljubičasto — runtime ili eksterni čvor u high-level prikazu.

Legenda se prikazuje ispod grafa.

## SAM2

SAM2 nema jedan jednostavan root `forward()` koji predstavlja kompletan inference tok. Zbog toga VizTool koristi composite režim: high-level tok je definisan kao pregled glavnih komponenti, a svaka komponenta se zatim analizira istim analyzerom kao i običan PyTorch model.

Primer se nalazi u:

```text
examples/sam2_single_session.py
```

Za SAM2 je prvo potrebno instalirati zvanični repozitorijum:

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```

Zatim se primer pokreće iz okruženja u kojem su dostupni i SAM2 i VizTool:

```bash
python examples/sam2_single_session.py
```

Primer koristi konfiguraciju `sam2_hiera_t.yaml` i ne zahteva checkpoint za samu demonstraciju arhitekture.

High-level prikaz predstavlja jedan vremenski korak obrade video frame-a:

```text
Input Frame -> Image Encoder -> Memory Attention -> Mask Decoder -> Mask Output
                              ^                  ^
                              |                  |
                   previous Memory Bank     Prompt Encoder
                                                 ^
                                                 |
                                               Prompt

Image Encoder ----+
                  +-> Memory Encoder -> updated Memory Bank
Mask Decoder -----+
```

Memory Bank je prikazan kao prethodna i ažurirana memorija da bi jedan vremenski korak ostao DAG.

Klikom na sledeće blokove otvara se njihov stvarni computation graph:

- Image Encoder;
- Memory Attention;
- Prompt Encoder;
- Mask Decoder;
- Memory Encoder.

## Kako alat radi

Tok obrade je:

```text
PyTorch model
    |
    +-- torch.fx.symbolic_trace
    |
    +-- ako FX ne uspe:
            torch.export + example input
    |
    v
ModelGraph
    |
    v
VisibleGraph
    |
    v
renderer + interakcija
```

`ModelGraph` čuva kompletan analizirani graf. `VisibleGraph` predstavlja samo deo koji je trenutno vidljiv u zavisnosti od otvorenih i zatvorenih modula.

## Struktura projekta

```text
VizTool/
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
├── examples/
│   ├── torchvision_demo.py
│   ├── example_input_demo.py
│   └── sam2_single_session.py
├── requirements.txt
└── README.md
```

## Uloga fajlova

`analyzer.py` analizira PyTorch model i pravi `ModelGraph`.

`graph.py` sadrži strukture podataka kojima se predstavljaju operacije, moduli i veze.

`view.py` određuje koji deo grafa je trenutno vidljiv i realizuje expand/collapse logiku.

`renderer.py` računa raspored i crta čvorove, grane, grupe i legendu.

`interactive.py` obrađuje klikove za standardni model.

`session.py` realizuje composite prikaz i navigaciju između overview-a i pojedinačnih komponenti.

`runtime.py` obrađuje i proverava example input-e.

`visualizer.py` sadrži glavni javni poziv `visualize()`.

## Ograničenja

- Nije moguće automatski odrediti validne dimenzije ulaza za proizvoljan `nn.Module`.
- `torch.fx` ne može da obradi svaku Python kontrolu toka; u tim slučajevima koristi se `torch.export`.
- Veoma duboko otvoreni modeli mogu postati vizuelno gusti.
- Za modele bez jednog reprezentativnog root `forward()` high-level veze moraju biti poznate ili deklarisane posebno.

## Korišćene tehnologije

- PyTorch
- torch.fx
- torch.export
- NetworkX
- Matplotlib
