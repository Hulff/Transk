# Pipeline do Transcritor de Episódios

## Visão geral

O pipeline transforma um episódio ou vídeo completo em um roteiro
audiovisual, combinando **falas**, **informações dos personagens**,
**ações**, **ambientes**, **eventos visíveis** e uma **análise
consolidada do episódio**.

``` mermaid
flowchart TD
    A["Vídeo completo<br/>episódio / filme / anime"] --> B["Extração de áudio<br/>FFmpeg / imageio-ffmpeg"]

    B --> C["Transcrição + diarização<br/>WhisperX"]
    C --> D["Falas com timestamps<br/>+ SPEAKER_XX"]

    D --> E["Reconhecimento de personagem por voz<br/>(opcional)"]
    E --> F["Falas identificadas<br/>por personagem"]

    F --> G["Agrupamento em cenas<br/>gap_threshold_seconds"]

    A --> H["Extração de frames<br/>vários frames por cena"]

    G --> I["Análise visual por cena<br/>Qwen2.5-VL"]
    H --> I
    F --> I

    I --> J["Análise individual da cena<br/>CONTEXTO / PERSONAGENS /<br/>AMBIENTE / ACONTECIMENTOS /<br/>AÇÕES / FALAS / ESTADOS"]

    J --> K["Síntese final<br/>Qwen2.5-VL - texto"]
    F --> K

    K --> L["Análise consolidada<br/>do episódio"]

    J --> M["Validação"]
    L --> M

    M --> N["Análise final validada<br/>_analise.txt"]

    F --> O["Merge da timeline<br/>falas + ações"]
    J --> O

    O --> P["Roteiro audiovisual"]
    P --> Q[".txt"]
    P --> R[".json"]
    P --> S[".srt"]

    N --> T["Saída final"]
    Q --> T
    R --> T
    S --> T
```

## Fluxo detalhado

``` text
┌─────────────────────────────────────────────────────────────┐
│                    VÍDEO COMPLETO                           │
│                episódio / filme / anime                     │
└─────────────────────────────┬───────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
┌───────────────────────────┐   ┌─────────────────────────────┐
│ EXTRAÇÃO DE ÁUDIO         │   │ EXTRAÇÃO / AMOSTRAGEM       │
│ FFmpeg                    │   │ DE FRAMES                   │
└─────────────┬─────────────┘   └──────────────┬──────────────┘
              │                                │
              ▼                                │
┌───────────────────────────┐                   │
│ WHISPERX                  │                   │
│ Transcrição + diarização │                   │
└─────────────┬─────────────┘                   │
              │                                │
              ▼                                │
┌───────────────────────────┐                   │
│ SPEAKER_00                │                   │
│ SPEAKER_01                │                   │
│ SPEAKER_02 ...            │                   │
└─────────────┬─────────────┘                   │
              │                                │
              ▼                                │
┌───────────────────────────┐                   │
│ RECONHECIMENTO DE VOZ     │                   │
│ personagem (opcional)     │                   │
└─────────────┬─────────────┘                   │
              │                                │
              ▼                                │
┌───────────────────────────┐                   │
│ FALAS + TIMESTAMPS        │                   │
│ + PERSONAGENS             │                   │
└─────────────┬─────────────┘                   │
              │                                │
              ▼                                │
┌───────────────────────────┐                   │
│ AGRUPAMENTO EM CENAS      │                   │
│ baseado em silêncio       │                   │
└─────────────┬─────────────┘                   │
              │                                │
              └──────────────┬─────────────────┘
                             ▼
              ┌───────────────────────────────┐
              │ ANÁLISE VISUAL DA CENA        │
              │ Qwen2.5-VL                    │
              │                               │
              │ Frames + diálogo da cena      │
              └───────────────┬───────────────┘
                              ▼
              ┌───────────────────────────────┐
              │ ANÁLISE INDIVIDUAL DA CENA   │
              │                               │
              │ CONTEXTO                      │
              │ PERSONAGENS                   │
              │ AMBIENTE                      │
              │ ACONTECIMENTOS                │
              │ AÇÕES                         │
              │ FALAS IMPORTANTES             │
              │ ESTADOS / EMOÇÕES OBSERVÁVEIS │
              └───────────────┬───────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
       ┌─────────────────────┐  ┌─────────────────────┐
       │ MERGE DA TIMELINE   │  │ SÍNTESE FINAL       │
       │                     │  │                     │
       │ Falas + ações       │  │ Todas as cenas     │
       └──────────┬──────────┘  └──────────┬──────────┘
                  │                        │
                  ▼                        ▼
       ┌─────────────────────┐  ┌─────────────────────┐
       │ ROTEIRO AUDIOVISUAL │  │ ANÁLISE CONSOLIDADA │
       └──────────┬──────────┘  └──────────┬──────────┘
                  │                        │
                  │                        ▼
                  │             ┌─────────────────────┐
                  │             │ VALIDAÇÃO           │
                  │             │ contra as cenas     │
                  │             └──────────┬──────────┘
                  │                        │
                  │                        ▼
                  │             ┌─────────────────────┐
                  │             │ _analise.txt        │
                  │             │ validado             │
                  │             └─────────────────────┘
                  │
                  ├──────────────► .txt
                  ├──────────────► .json
                  └──────────────► .srt
```

## O que entra e o que sai de cada etapa

  -----------------------------------------------------------------------
  Etapa             Entrada           Processamento     Saída
  ----------------- ----------------- ----------------- -----------------
  Extração de áudio Vídeo             FFmpeg            Áudio

  Transcrição       Áudio             WhisperX          Falas +
                                                        timestamps

  Diarização        Áudio             WhisperX /        Falante por
                                      pyannote          segmento

  Reconhecimento de Falas + perfis    Speaker mapping   Personagem por
  voz                                                   fala

  Agrupamento       Falas             Gaps entre falas  Cenas

  Frames            Vídeo + cenas     Amostragem        Frames por cena
                                      temporal          

  Análise de cena   Frames + diálogo  Qwen2.5-VL        Análise factual
                                                        da cena

  Síntese           Análises das      Qwen2.5-VL texto  Análise
                    cenas + diálogo                     consolidada

  Validação         Síntese +         Segunda passada   Análise corrigida
                    análises          do modelo         
                    originais                           

  Merge             Falas + cenas     Ordenação         Timeline
                                      temporal          

  Exportação        Timeline +        Formatter         `.txt`, `.json`,
                    análise                             `.srt`,
                                                        `_analise.txt`
  -----------------------------------------------------------------------

## Princípio da análise visual

A análise visual é feita **por cena**, e não como uma sequência de
legendas independentes para cada frame.

``` text
Cena
│
├── Frame 01 ─┐
├── Frame 02  │
├── Frame 03  │
├── ...       ├──► Qwen2.5-VL
├── Frame 11  │
└── Frame 12 ─┘
       +
   diálogo completo
       │
       ▼
┌───────────────────────────┐
│ Análise da cena           │
│                           │
│ O que aconteceu?          │
│ Quem aparece?             │
│ Onde acontece?            │
│ Quais ações ocorreram?   │
│ Quais objetos importam?  │
│ Quais eventos ocorreram? │
└───────────────────────────┘
```

Isso permite capturar mudanças durante a cena, por exemplo:

``` text
Personagem está de pé
        ↓
pega um objeto
        ↓
avança em direção ao adversário
        ↓
ataca
        ↓
adversário cai
```

em vez de produzir cinco descrições independentes de imagens estáticas.

## Síntese final

A síntese não reprocessa os frames do vídeo.

``` text
CENA 1 ──► análise da cena 1 ──┐
CENA 2 ──► análise da cena 2 ──┤
CENA 3 ──► análise da cena 3 ──┤
CENA 4 ──► análise da cena 4 ──┤
...                             ├──► SÍNTESE FINAL
CENA N ──► análise da cena N ──┘
```

A síntese utiliza as informações já extraídas para construir uma visão
cronológica do episódio inteiro.

## Validação

A etapa de validação compara a análise consolidada com as análises
originais das cenas:

``` text
              ┌──────────────────────┐
              │ ANÁLISE CONSOLIDADA  │
              └──────────┬───────────┘
                         │
                         ▼
                ┌────────────────┐
                │   VALIDAÇÃO    │
                └───────┬────────┘
                        ▲
                        │
       ┌────────────────┴────────────────┐
       │                                 │
┌──────┴──────┐   ┌──────────────┐   ┌───┴──────────┐
│ Análise     │   │ Análise      │   │ Análise      │
│ Cena 1      │   │ Cena 2       │   │ Cena N       │
└─────────────┘   └──────────────┘   └──────────────┘
                        │
                        ▼
              ┌──────────────────────┐
              │ ANÁLISE FINAL        │
              │ _analise.txt         │
              └──────────────────────┘
```

A validação verifica principalmente:

-   nomes de personagens;
-   atribuição de acontecimentos;
-   consistência entre cenas;
-   informações não sustentadas;
-   erros de escrita;
-   conteúdo indevidamente inventado;
-   markdown ou formatação indesejada.

## Cache

O pipeline também pode reutilizar resultados intermediários:

``` text
Vídeo
 │
 ├──► temp/cache/<nome>_falas.json
 │
 ├──► temp/cache/<nome>_cenas.json
 │
 ├──► temp/cache/<nome>_analise_rascunho.json
 │
 └──► temp/cache/<nome>_analise_validada.json
```

Com isso, `--forcar` pode ser usado quando for necessário recalcular
todas as etapas.

## Saída final

``` text
resultados/
│
├── episodio.txt
│      └── roteiro de falas + ações
│
├── episodio.json
│      └── dados estruturados da timeline
│
├── episodio.srt
│      └── legendas / falas temporizadas
│
└── episodio_analise.txt
       └── análise audiovisual consolidada e validada
```

## Comando principal

``` bash
python app.py processar caminho/do/episodio.mp4
```

O fluxo completo é:

``` text
Vídeo
  ↓
Áudio
  ↓
WhisperX
  ↓
Falas + diarização
  ↓
Personagens por voz
  ↓
Cenas
  ↓
Frames + diálogo
  ↓
Qwen2.5-VL
  ↓
Análises individuais
  ↓
Síntese
  ↓
Validação
  ↓
┌─────────────────────────────────────┐
│ .txt                                │
│ .json                               │
│ .srt                                │
│ _analise.txt                        │
└─────────────────────────────────────┘
```
