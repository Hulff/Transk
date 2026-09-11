# Transcritor de Episódios

App em Python 100% open source para transformar episódios (vídeo) em
roteiro de texto + uma análise narrativa consolidada: falas separadas
por personagem, descrição de ação/ambiente ancorada em cada cena, e um
resumo audiovisual coerente do episódio inteiro, já revisado contra a
fonte original.

## Visão geral do pipeline

1. **Extração de áudio** (ffmpeg, via `imageio-ffmpeg` — não precisa
   instalar nada no sistema)
2. **Transcrição + diarização** (WhisperX) — identifica falas com
   timestamp e separa por locutor (`SPEAKER_00`, `SPEAKER_01`...)
3. **Reconhecimento de personagem por voz** (opcional) — troca os
   rótulos genéricos pelo nome real, se você cadastrar amostras de voz
4. **Agrupamento em cenas** — falas consecutivas são agrupadas numa
   mesma cena; um silêncio acima de `gap_threshold_seconds` começa uma
   cena nova
5. **Análise visual por cena** (Qwen2.5-VL) — pra cada cena, extrai
   vários frames espalhados pela duração inteira dela e pede pro
   modelo descrever ambiente, ações, personagens e eventos, usando
   **todo o diálogo da cena** como contexto (não frase a frase)
6. **Síntese final** — uma chamada só de texto (sem reprocessar
   frames) junta as análises de todas as cenas numa narrativa
   cronológica única do episódio inteiro
7. **Validação** — uma segunda passada confere essa síntese contra as
   análises de cena originais e corrige inconsistências: nomes que não
   aparecem no diálogo, eventos mal atribuídos, erros de escrita,
   markdown solto
8. **Saída final** — roteiro (`.txt`, `.json`, `.srt`) + a análise
   consolidada e validada (`_analise.txt`)

## Instalação

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

O ffmpeg **não precisa ser instalado manualmente no sistema** — o
projeto usa o `imageio-ffmpeg`, que baixa e gerencia um binário próprio
na primeira execução (funciona igual em Linux, Mac e Windows, sem
mexer no PATH). Algumas libs (como o WhisperX) chamam `ffmpeg`
diretamente esperando achá-lo no PATH — por isso, ao iniciar, o app
cria automaticamente um atalho `ffmpeg` dentro de `venv/bin/` apontando
pro binário baixado (veja `core/ffmpeg_utils.py`).

### Token do Hugging Face (obrigatório para diarização)

O token fica num arquivo `.env`, **nunca** direto no `config.yaml`
(pra não vazar sem querer se você commitar o repo).

1. Copie o exemplo: `cp .env.example .env`
2. Crie uma conta gratuita em huggingface.co
3. Gere um token em huggingface.co/settings/tokens (tipo "Read" evita
   problema de permissão por repositório)
4. Aceite os termos de uso destes modelos (logado com a mesma conta):
   - huggingface.co/pyannote/speaker-diarization-community-1
   - huggingface.co/pyannote/segmentation-3.0
5. Cole o token no `.env`:
   ```
   HF_TOKEN=seu_token_aqui
   ```

O `.env` já está no `.gitignore` — não sobe pro repositório.

## GPU (obrigatória para análise de cena e síntese)

O Qwen2.5-VL-7B-Instruct é um modelo pesado — a análise de cena, a
síntese final e a validação **precisam de GPU** (CPU é impraticável).
Duas formas de conseguir uma:

- **Google Colab** (recomendado, tem tier gratuito com GPU Nvidia T4).
  Se a GPU tiver menos de ~16GB de VRAM, ative no `config.yaml`:
  ```yaml
  scene_analysis:
    load_in_4bit: true
  ```
- **GPU Nvidia local** com CUDA configurado. GPUs AMD não são
  suportadas de forma confiável por essas libs (WhisperX, pyannote,
  transformers) — CUDA é exclusivo Nvidia. Em clusters SLURM, lembre
  de alocar um nó de computação (`srun --gres=gpu:1 ...`) — o nó de
  login geralmente não tem GPU.

Se você não tiver GPU disponível, ainda dá pra usar só a parte de
áudio (`python app.py audio`), que roda em CPU (mais lenta, mas
funcional), e desabilitar a análise de cena no config
(`scene_analysis.enabled: false`).

O pipeline carrega e libera o modelo (`del model` + limpeza de cache
da GPU) entre as etapas de análise de cena, síntese e validação, então
elas não competem por VRAM ao mesmo tempo — mas cada uma recarrega o
modelo do zero, o que adiciona alguns segundos de overhead por etapa.

## Uso via linha de comando

O pipeline é dividido em comandos separados, com cache automático —
você pode rodar a parte de áudio e a de cena em momentos diferentes:

```bash
# Só a parte de áudio (transcrição + diarização + personagens por voz)
python app.py audio caminho/do/episodio.mp4

# Só a parte de cena (precisa que "audio" já tenha rodado antes)
python app.py cena caminho/do/episodio.mp4

# Pipeline completo: áudio + cena + síntese + validação
python app.py processar caminho/do/episodio.mp4

# Forçar recomputar tudo, ignorando qualquer cache existente
python app.py processar caminho/do/episodio.mp4 --forcar

# Processar tudo, mas pular a análise de cena (e portanto a síntese
# e validação, que dependem dela) mesmo se estiver habilitada no config
python app.py processar caminho/do/episodio.mp4 --sem-cena
```

Os arquivos de saída vão para a pasta `resultados/`:
- `<nome>.txt`, `<nome>.json`, `<nome>.srt` — roteiro (falas + ações)
- `<nome>_analise.txt` — a síntese narrativa final, já validada

Os resultados intermediários ficam em cache dentro de `temp/cache/`,
por etapa (`falas`, `cenas`, `analise_rascunho`, `analise_validada`) —
se uma etapa já rodou, `processar` reaproveita o cache em vez de
refazer o trabalho (a menos que você use `--forcar`).

## Uso via interface web

```bash
streamlit run interface_streamlit.py
```

Abre no navegador — dá pra fazer upload do episódio, acompanhar o
progresso e renomear os personagens direto na tela antes de baixar o
resultado. (Cobre só o fluxo de falas por enquanto; a análise de cena,
síntese e validação via Qwen2.5-VL ainda são exclusivas da CLI.)

## Reconhecimento automático de personagens por voz (opcional)

Se quiser que o app já identifique os personagens pelo nome (em vez de
"SPEAKER_00", "SPEAKER_01"...), crie pastas com amostras de voz:

```
models/voice_profiles/
    joao/
        amostra1.wav
        amostra2.wav
    maria/
        amostra1.wav
```

Cada amostra deve ter só a voz daquele personagem falando, de
preferência uns 5-10 segundos limpos (sem música/efeitos por cima).

Se você não tiver essas amostras, sem problema: o app gera os rótulos
genéricos e você renomeia manualmente na interface Streamlit (tem um
campo pra isso) ou editando o dicionário no
`speaker_mapping.mapear_manual()`.

Importante: como o pipeline evita inventar nomes de personagens que
não aparecem no diálogo (veja abaixo), cadastrar essas amostras de voz
é a única forma confiável de ter nomes reais na análise final quando
os próprios personagens não se chamam pelo nome na cena.

## Análise de cena e síntese com Qwen2.5-VL

Configurável em `config.yaml`, seção `scene_analysis`:

- `frames_per_scene`: quantos frames extrair, espalhados pela duração
  inteira de cada cena
- `gap_threshold_seconds`: silêncio acima disso entre duas falas
  começa uma cena nova
- `padding_seconds`: margem de tempo antes/depois da cena
- `question`: o prompt enviado na análise de cena (usa `{dialogue}` e
  `{language}` como placeholders); vazio usa o prompt padrão
- `language`: idioma da descrição gerada
- `max_new_tokens`, `repetition_penalty`, `no_repeat_ngram_size`,
  `do_sample`: parâmetros de geração da etapa de análise de cena
- `validation_max_new_tokens`: parâmetro de geração da etapa de
  validação (processa o documento inteiro, por isso tem um limite
  maior e configurável à parte)
- `load_in_4bit`: ative se a GPU tiver VRAM limitada

### Sobre nomes de personagens

Os prompts (análise de cena, síntese e validação) são instruídos a
**só usar o nome de um personagem se ele aparecer literalmente no
diálogo transcrito** — o modelo não pode usar conhecimento externo
sobre o filme/anime pra "reconhecer" e nomear alguém. Quando o nome
não está no diálogo, ele usa um rótulo neutro e consistente ("o
personagem de cabelo espetado", "o mais alto", etc.) em vez de chutar.
Isso evita o tipo de erro visto antes dessa regra — nomes quase certos
mas errados (`Vegata`, `Veggies`, `Goham's`) ou números inventados
(`Android 7`, `Android 8`).

A etapa de validação reforça essa checagem numa segunda passada,
comparando a síntese final contra as análises de cena originais.

## Estrutura do projeto

```
transcritor_app/
├── core/
│   ├── ffmpeg_utils.py        # resolve o binário do ffmpeg (imageio-ffmpeg)
│   ├── extract_audio.py       # extração de áudio (ffmpeg)
│   ├── transcribe.py          # transcrição + diarização (WhisperX)
│   ├── speaker_mapping.py     # SPEAKER_00 -> nome do personagem
│   ├── scene_analysis.py      # agrupamento em cenas + análise visual (Qwen2.5-VL)
│   ├── context_analysis.py    # síntese narrativa final + validação
│   ├── merge.py               # junta falas + ações em uma timeline
│   └── cache.py               # cache em JSON dos resultados intermediários
├── output/
│   └── formatter.py           # gera .txt, .json, .srt
├── models/voice_profiles/     # amostras de voz por personagem
├── app.py                     # CLI (comandos: audio, cena, processar)
├── interface_streamlit.py     # interface web (fluxo de falas)
├── .env.example                # modelo do arquivo de variáveis de ambiente
└── config.yaml                 # configurações gerais
```

## Notas importantes

- **Precisão da diarização**: vozes parecidas (ex: personagens do
  mesmo gênero/timbre similar) podem ser confundidas. Revisar
  manualmente ainda é recomendado antes de considerar o roteiro final.
- **Qualidade da análise de cena**: mesmo com o Qwen2.5-VL (bem mais
  capaz que legendadores genéricos), o modelo só enxerga os frames
  extraídos — ações muito rápidas entre dois frames, ou que dependem
  de contexto de episódios anteriores, podem passar despercebidas.
- **A validação reduz, mas não elimina, erros**: é uma segunda
  passada do mesmo tipo de modelo, então pode não pegar 100% dos
  problemas — vale sempre uma revisão humana rápida no `_analise.txt`
  final antes de considerar definitivo.
- **Custo de GPU**: rodar localmente sem GPU Nvidia decente é
  impraticável pras etapas de cena, síntese e validação. O Google
  Colab (gratuito, com T4) é a opção mais acessível pra quem não tem
  uma.