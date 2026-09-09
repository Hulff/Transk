# Transcritor de Episódios

App em Python 100% open source para transformar episódios (vídeo) em texto,
com falas separadas por personagem e (opcionalmente) descrições de ação/cena.

## Instalação

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Você também precisa do **ffmpeg** instalado no sistema:
- Ubuntu/Debian: `sudo apt install ffmpeg`
- Mac: `brew install ffmpeg`
- Windows: baixar em ffmpeg.org e adicionar ao PATH

### Token do Hugging Face (obrigatório para diarização)

1. Crie uma conta gratuita em huggingface.co
2. Gere um token em huggingface.co/settings/tokens
3. Aceite os termos dos modelos:
   - huggingface.co/pyannote/speaker-diarization-3.1
   - huggingface.co/pyannote/segmentation-3.0
4. Cole o token em `config.yaml` no campo `diarization.hf_token`

## Uso via linha de comando

```bash
python app.py processar caminho/do/episodio.mp4
```

Os arquivos de saída (.txt, .json, .srt) vão para a pasta `resultados/`.

## Uso via interface web (mais fácil)

```bash
streamlit run interface_streamlit.py
```

Abre no navegador — dá pra fazer upload do episódio, acompanhar o progresso
e renomear os personagens direto na tela antes de baixar o resultado.

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

Cada amostra deve ter só a voz daquele personagem falando, de preferência
uns 5-10 segundos limpos (sem música/efeitos por cima).

Se você não tiver essas amostras, sem problema: o app gera os rótulos
genéricos e você renomeia manualmente na interface Streamlit (tem um campo
pra isso) ou editando o dicionário no `speaker_mapping.mapear_manual()`.

## Análise de ações/cena (opcional, requer GPU)

Desabilitada por padrão (`scene_analysis.enabled: false` no config.yaml)
porque usa o modelo BLIP-2, que é pesado. Se tiver GPU decente, ative no
config e rode via CLI (`python app.py processar video.mp4`) — a interface
Streamlit por enquanto só cobre o fluxo de falas.

## Estrutura do projeto

```
transcritor_app/
├── core/
│   ├── extract_audio.py      # extração de áudio (ffmpeg)
│   ├── transcribe.py         # transcrição + diarização (WhisperX)
│   ├── speaker_mapping.py    # SPEAKER_00 -> nome do personagem
│   ├── scene_analysis.py     # ações de cena (BLIP-2, opcional)
│   └── merge.py              # junta falas + ações em uma timeline
├── output/
│   └── formatter.py          # gera .txt, .json, .srt
├── models/voice_profiles/    # amostras de voz por personagem
├── app.py                    # CLI
├── interface_streamlit.py    # interface web
└── config.yaml                # configurações gerais
```

## Notas importantes

- **CPU vs GPU**: sem GPU, o WhisperX (modelo large) pode demorar bastante
  em episódios longos. Pra testar rápido, use `model: "small"` ou `"base"`
  no config.yaml.
- **Precisão da diarização**: vozes parecidas (ex: personagens do mesmo
  gênero/timbre similar) podem ser confundidas. Revisar manualmente ainda
  é recomendado antes de considerar o roteiro final.
- **Ações de cena**: a qualidade das descrições do BLIP-2 é genérica
  ("uma pessoa sentada em uma sala") — não espere que ele identifique
  ações específicas da trama sem ajustes finos.
