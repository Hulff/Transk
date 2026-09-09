"""
Interface web local para o Transcritor de Episódios.

Rodar com:
    streamlit run interface_streamlit.py
"""
import streamlit as st
import yaml
import tempfile
from pathlib import Path

from core.extract_audio import extract_audio
from core.transcribe import transcribe_and_diarize
from core.speaker_mapping import map_speakers_to_characters, mapear_manual
from core.merge import merge_timeline
from output.formatter import to_txt, to_json, save_outputs

st.set_page_config(page_title="Transcritor de Episódios", layout="wide")
st.title("🎬 Transcritor de Episódios")
st.caption("Falas separadas por personagem + ações e contexto de cena")

with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

uploaded_file = st.file_uploader("Envie o episódio (mp4, mkv, etc)", type=["mp4", "mkv", "avi", "mov"])

modo_mapeamento = st.radio(
    "Como identificar os personagens?",
    ["Reconhecimento automático por voz (requer perfis cadastrados)", "Vou renomear manualmente depois"],
)

if uploaded_file and st.button("Processar episódio"):
    with tempfile.TemporaryDirectory() as tmp_dir:
        video_path = str(Path(tmp_dir) / uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

        with st.status("Processando...", expanded=True) as status:
            st.write("Extraindo áudio...")
            audio_path = extract_audio(video_path, output_dir=tmp_dir)

            st.write("Transcrevendo e identificando falantes (pode demorar bastante)...")
            segments = transcribe_and_diarize(
                audio_path,
                model_name=cfg["whisper"]["model"],
                language=cfg["whisper"]["language"],
                device=cfg["whisper"]["device"],
                compute_type=cfg["whisper"]["compute_type"],
                hf_token=cfg["diarization"]["hf_token"],
            )

            if modo_mapeamento.startswith("Reconhecimento"):
                st.write("Reconhecendo personagens por voz...")
                segments = map_speakers_to_characters(
                    segments,
                    audio_path=audio_path,
                    voice_profiles_dir=cfg["speaker_mapping"]["voice_profiles_dir"],
                    hf_token=cfg["diarization"]["hf_token"],
                    similarity_threshold=cfg["speaker_mapping"]["similarity_threshold"],
                )

            timeline = merge_timeline(segments, scenes=None)
            status.update(label="Concluído!", state="complete")

        st.session_state["timeline"] = timeline
        st.session_state["base_name"] = Path(uploaded_file.name).stem

if "timeline" in st.session_state:
    timeline = st.session_state["timeline"]

    st.subheader("Renomear personagens (opcional)")
    speakers_detectados = sorted({item["speaker"] for item in timeline if item["type"] == "fala"})
    novos_nomes = {}
    cols = st.columns(len(speakers_detectados)) if speakers_detectados else []
    for col, speaker in zip(cols, speakers_detectados):
        novos_nomes[speaker] = col.text_input(f"Nome para {speaker}", value=speaker)

    if st.button("Aplicar novos nomes"):
        timeline = [
            {**item, "speaker": novos_nomes.get(item["speaker"], item["speaker"])}
            if item["type"] == "fala" else item
            for item in timeline
        ]
        st.session_state["timeline"] = timeline

    st.subheader("Resultado")
    texto_formatado = to_txt(timeline)
    st.text_area("Roteiro gerado", texto_formatado, height=400)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Baixar como .txt",
            texto_formatado,
            file_name=f"{st.session_state['base_name']}.txt",
        )
    with col2:
        st.download_button(
            "Baixar como .json",
            to_json(timeline),
            file_name=f"{st.session_state['base_name']}.json",
        )
