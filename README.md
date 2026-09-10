# Transcritor de Episódios

App em Python 100% open source para transformar episódios (vídeo) em
roteiro de texto: falas separadas por personagem + descrição de ação/
cena, ancorada no que está acontecendo em cada linha de diálogo.

## Visão geral do pipeline

1. **Extração de áudio** (ffmpeg)
2. **Transcrição + diarização** (WhisperX) — identifica falas com
   timestamp e separa por locutor (`SPEAKER_00`, `SPEAKER_01`...)
3. **Reconhecimento de personagem por voz** (opcional) — troca os
   rótulos genéricos pelo nome real, se você cadastrar amostras de voz
4. **Análise de cena** (Qwen2.5-VL) — pra cada fala, extrai vários
   frames da janela de tempo dela e pede pro modelo descrever a ação/
   expressão/interação visível, usando o diálogo só como contexto
5. **Montagem do roteiro final** — junta falas e ações numa timeline
   única, exporta em `.txt`, `.json` e `.srt`

## Instalação

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

O ffmpeg **não precisa ser instalado manualmente no sistema** — o
projeto usa o `imageio-ffmpeg`, que baixa e gerencia um binário próprio
na primeira execução (funciona igual em Linux, Mac e Windows, sem
mexer no PATH).

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

## GPU (obrigatória para a análise de cena)

O Qwen2.5-VL-7B-Instruct é um modelo pesado — a análise de cena
**precisa de GPU** (CPU é impraticável). Duas formas de conseguir uma:

- **Google Colab** (recomendado, tem tier gratuito com GPU Nvidia T4).
  Se a GPU tiver menos de ~16GB de VRAM, ative no `config.yaml`:
  ```yaml
  scene_analysis:
    load_in_4bit: true
  ```
- **GPU Nvidia local** com CUDA configurado. GPUs AMD não são
  suportadas de forma confiável por essas libs (WhisperX, pyannote,
  transformers) — CUDA é exclusivo Nvidia.

Se você não tiver GPU disponível, ainda dá pra usar só a parte de
áudio (`python app.py audio`), que roda em CPU (mais lenta, mas
funcional), e desabilitar a análise de cena no config
(`scene_analysis.enabled: false`).

## Uso via linha de comando

O pipeline é dividido em comandos separados, com cache automático —
você pode rodar a parte de áudio e a de cena em momentos diferentes:

```bash
# Só a parte de áudio (transcrição + diarização + personagens por voz)
python app.py audio caminho/do/episodio.mp4

# Só a parte de cena (precisa que "audio" já tenha rodado antes)
python app.py cena caminho/do/episodio.mp4

# As duas juntas — reaproveita cache se "audio" ou "cena" já rodaram
python app.py processar caminho/do/episodio.mp4

# Forçar recomputar tudo, ignorando qualquer cache existente
python app.py processar caminho/do/episodio.mp4 --forcar

# Processar tudo, mas pular a análise de cena mesmo se estiver
# habilitada no config
python app.py processar caminho/do/episodio.mp4 --sem-cena
```

Os arquivos de saída (`.txt`, `.json`, `.srt`) vão para a pasta
`resultados/`. Os resultados intermediários (falas e cenas) ficam em
cache dentro de `temp/cache/`.

## Uso via interface web

```bash
streamlit run interface_streamlit.py
```

Abre no navegador — dá pra fazer upload do episódio, acompanhar o
progresso e renomear os personagens direto na tela antes de baixar o
resultado. (Cobre só o fluxo de falas por enquanto; a análise de cena
via Qwen2.5-VL ainda é exclusiva da CLI.)

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

## Análise de cena com Qwen2.5-VL

Configurável em `config.yaml`, seção `scene_analysis`:

- `frames_per_fala`: quantos frames extrair por linha de diálogo
- `padding_seconds`: margem de tempo antes/depois da fala
- `question`: o prompt enviado ao modelo (usa `{dialogue}` e
  `{language}` como placeholders)
- `language`: idioma da descrição gerada
- `max_new_tokens`, `repetition_penalty`, `no_repeat_ngram_size`,
  `do_sample`: parâmetros de geração do modelo
- `load_in_4bit`: ative se a GPU tiver VRAM limitada

O prompt padrão já vem calibrado pra descrever só o que é
**visualmente confirmável** nos frames (evita o modelo inventar nomes,
ações ou relações que não aparecem na imagem, usando o diálogo apenas
como contexto).

## Estrutura do projeto

```
transcritor_app/
├── core/
│   ├── ffmpeg_utils.py        # resolve o binário do ffmpeg (imageio-ffmpeg)
│   ├── extract_audio.py      # extração de áudio (ffmpeg)
│   ├── transcribe.py         # transcrição + diarização (WhisperX)
│   ├── speaker_mapping.py    # SPEAKER_00 -> nome do personagem
│   ├── scene_analysis.py     # ações de cena (Qwen2.5-VL)
│   ├── merge.py              # junta falas + ações em uma timeline
│   └── cache.py              # cache em JSON dos resultados intermediários
├── output/
│   └── formatter.py          # gera .txt, .json, .srt
├── models/voice_profiles/    # amostras de voz por personagem
├── app.py                    # CLI (comandos: audio, cena, processar)
├── interface_streamlit.py    # interface web (fluxo de falas)
├── .env.example               # modelo do arquivo de variáveis de ambiente
└── config.yaml                # configurações gerais
```

## Notas importantes

- **Precisão da diarização**: vozes parecidas (ex: personagens do
  mesmo gênero/timbre similar) podem ser confundidas. Revisar
  manualmente ainda é recomendado antes de considerar o roteiro final.
- **Qualidade da análise de cena**: mesmo com o Qwen2.5-VL (bem mais
  capaz que legendadores genéricos), o modelo só enxerga os frames
  extraídos — ações muito rápidas entre dois frames, ou que dependem
  de contexto de episódios anteriores, podem passar despercebidas.
- **Custo de GPU**: rodar localmente sem GPU Nvidia decente é
  impraticável para a etapa de cena. O Google Colab (gratuito, com T4)
  é a opção mais acessível pra quem não tem uma.