from __future__ import annotations

import hashlib
import io

import cv2
import matplotlib
import numpy as np
import streamlit as st
from PIL import Image

from media_core import (
    apply_image_operation,
    apply_transform,
    create_collage,
    decode_image,
    load_audio,
    read_first_video_frame,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Multimedia Studio",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

IMAGE_OPERATIONS = [
    "Convert to Grayscale",
    "Equalize Histogram",
    "Gaussian Blur",
    "Canny Edge Detection",
    "Sharpen",
    "Resize to 150%",
    "Center Crop",
    "Adjust Brightness & Contrast",
]
# Fixed team names are defined here.
TEAM_MEMBERS: tuple[str, str] = ("Mohamed Bassem Mohamed", "Mohamed Kamel Elsayed")


st.markdown(
    """
    <style>
    :root {
        --ink: #172b45;
        --muted: #64758b;
        --blue: #2864a0;
        --line: #e4eaf2;
        --surface: #ffffff;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 10% 0%, #e8f1fc 0, transparent 35%),
            #f5f7fb;
        color: var(--ink);
        font-family: "Segoe UI", "Arial Rounded MT Bold", sans-serif;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMainBlockContainer"] {
        max-width: 1440px;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }
    .hero {
        background: linear-gradient(120deg, #17365d, #2864a0);
        border-radius: 24px;
        color: #fff;
        padding: 28px 34px;
        margin-bottom: 20px;
        box-shadow: 0 16px 38px rgba(23, 54, 93, .13);
    }
    .hero h1 {
        color: #fff;
        font-size: clamp(2rem, 3.2vw, 2.8rem);
        letter-spacing: -.04em;
        margin: 0;
    }
    .hero p {
        color: #d7e5f4;
        font-size: 1.02rem;
        margin: 8px 0 0;
    }
    .eyebrow {
        color: #bad6f2;
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .14em;
        margin-bottom: 8px;
        text-transform: uppercase;
    }
    .feature-list {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 12px 0 18px;
    }
    .feature-pill {
        background: #eaf1f9;
        border: 1px solid #dce7f3;
        border-radius: 999px;
        color: #315779;
        display: inline-block;
        font-size: .82rem;
        font-weight: 650;
        padding: 6px 12px;
    }
    .section-note {
        color: var(--muted);
        margin-top: -6px;
    }
    div[data-testid="stTabs"] button[role="tab"] {
        background: #e8eef6;
        border: 1px solid transparent;
        border-radius: 13px;
        color: #38536f;
        font-weight: 700;
        margin-right: 7px;
        padding: 9px 16px;
    }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        background: #fff;
        border-color: #dce6f1;
        color: #205b95;
        box-shadow: 0 5px 16px rgba(31, 65, 103, .08);
    }
    .stButton button, .stDownloadButton button {
        border-radius: 12px;
        font-weight: 700;
        min-height: 42px;
    }
    [data-testid="stTextInput"] label,
    [data-testid="stWidgetLabel"] p {
        color: #35506d !important;
        font-weight: 650;
    }
    [data-testid="stTextInput"] [data-baseweb="input"] {
        background: #fff !important;
        border: 1px solid #d8e2ee !important;
        border-radius: 12px !important;
    }
    [data-testid="stTextInput"] input {
        background: transparent !important;
        color: #172b45 !important;
    }
    [data-testid="stTextInput"] input::placeholder {
        color: #8292a5 !important;
        opacity: 1;
    }
    div[data-testid="stTabs"] button[role="tab"] p {
        color: #38536f !important;
    }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p {
        color: #205b95 !important;
    }
    div[data-testid="stFileUploader"] section {
        background: #fbfcfe;
        border: 1px dashed #b9c9dc;
        border-radius: 15px;
    }
    div[data-testid="stMetric"] {
        background: #fff;
        border: 1px solid var(--line);
        border-radius: 15px;
        padding: 14px 16px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255,255,255,.82);
        border-color: var(--line);
        border-radius: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def show_feature_list(features: list[str]) -> None:
    pills = "".join(f'<span class="feature-pill">{feature}</span>' for feature in features)
    st.markdown(f'<div class="feature-list">{pills}</div>', unsafe_allow_html=True)


def image_png_bytes(image: np.ndarray) -> bytes:
    if image.ndim == 2:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output = io.BytesIO()
    Image.fromarray(rgb_image).save(output, format="PNG")
    return output.getvalue()


def upload_signature(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@st.cache_data(show_spinner=False)
def decode_audio_cached(payload: bytes, filename: str) -> tuple[np.ndarray, int]:
    return load_audio(payload, filename)


@st.cache_data(show_spinner=False)
def read_video_cached(payload: bytes, filename: str) -> np.ndarray:
    return read_first_video_frame(payload, filename)


st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Creative media workspace</div>
        <h1>Multimedia Project</h1>
        <p><h2> Dr.Abd Elmounem Ali _ <h3> Dr.Abanoub Farid </h3> </h2> </p>
        <p>image processing, audio analysis, frequency transforms and video previews</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown("### Team members")
    st.caption("Team member names are fixed for this project.")
    member_columns = st.columns(len(TEAM_MEMBERS))
    for index, (column, name) in enumerate(zip(member_columns, TEAM_MEMBERS), start=1):
        with column:
            st.text_input(
                f"Member {index}",
                value=name,
                disabled=True,
                key=f"fixed_team_member_{index}",
            )

image_tab, audio_tab, transform_tab, video_tab = st.tabs(
    ["🖼️  Image Editing", "🎵  Audio Analysis", "〰️  DCT / DFT / Wavelet", "🎞️  Video & Collage"]
)


with image_tab:
    st.markdown("## Image processing")
    st.markdown(
        '<p class="section-note">Load an image, choose an operation, and adjust its properties below.</p>',
        unsafe_allow_html=True,
    )
    show_feature_list(
        [
            "Grayscale",
            "Histogram equalization",
            "Gaussian blur",
            "Canny edges",
            "Sharpen",
            "Resize & crop",
            "Brightness & contrast",
            "Download result",
        ]
    )

    image_upload = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        key="image_upload",
    )
    current_image: np.ndarray | None = None
    source_image: np.ndarray | None = None
    if image_upload is not None:
        payload = image_upload.getvalue()
        signature = upload_signature(payload)
        if signature != st.session_state.get("image_signature"):
            try:
                source_image = decode_image(payload)
                st.session_state["image_signature"] = signature
                st.session_state["image_source"] = source_image
                st.session_state["image_current"] = source_image.copy()
            except ValueError as exc:
                st.session_state["image_signature"] = signature
                st.session_state["image_source"] = None
                st.session_state["image_current"] = None
                st.error(str(exc))
        source_image = st.session_state.get("image_source")
        current_image = st.session_state.get("image_current")

    with st.container(border=True):
        st.markdown("### Image properties")
        operation_column, brightness_column, contrast_column = st.columns([1.5, 1, 1])
        with operation_column:
            operation = st.selectbox("Operation", IMAGE_OPERATIONS, key="image_operation")
        brightness = 0.0
        contrast = 1.0
        with brightness_column:
            if operation == "Adjust Brightness & Contrast":
                brightness = st.slider(
                    "Brightness",
                    min_value=-100,
                    max_value=100,
                    value=0,
                    key="image_brightness",
                )
        with contrast_column:
            if operation == "Adjust Brightness & Contrast":
                contrast = st.slider(
                    "Contrast",
                    min_value=0.5,
                    max_value=3.0,
                    value=1.0,
                    step=0.1,
                    key="image_contrast",
                )

        apply_column, reset_column = st.columns([1, 1])
        with apply_column:
            apply_clicked = st.button(
                "Apply operation",
                type="primary",
                disabled=current_image is None,
                key="apply_image_operation",
                width="stretch",
            )
        with reset_column:
            reset_clicked = st.button(
                "Reset image",
                disabled=source_image is None,
                key="reset_image_operation",
                width="stretch",
            )

        if apply_clicked and current_image is not None:
            current_image = apply_image_operation(
                current_image,
                operation,
                brightness=brightness,
                contrast=contrast,
            )
            st.session_state["image_current"] = current_image
        if reset_clicked and source_image is not None:
            current_image = source_image.copy()
            st.session_state["image_current"] = current_image

    if current_image is not None and source_image is not None:
        original_column, result_column = st.columns(2)
        with original_column:
            st.markdown("#### Original")
            st.image(source_image, channels="BGR", width="stretch")
        with result_column:
            st.markdown("#### Current result")
            st.image(current_image, channels="BGR", width="stretch")
            st.download_button(
                "Download edited image",
                data=image_png_bytes(current_image),
                file_name="edited_image.png",
                mime="image/png",
                key="download_edited_image",
                width="stretch",
            )
    else:
        st.info("Upload an image to enable the editing tools.")


with audio_tab:
    st.markdown("## Audio analysis")
    st.markdown(
        '<p class="section-note">Inspect a waveform, view a spectrogram, and measure loudness.</p>',
        unsafe_allow_html=True,
    )
    show_feature_list(["Waveform", "Spectrogram", "RMS", "dBFS", "Sample rate", "Duration"])
    st.caption("WAV works directly. MP3, FLAC, OGG, and M4A require FFmpeg to be installed.")

    audio_upload = st.file_uploader(
        "Upload an audio file",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
        key="audio_upload",
    )
    if audio_upload is None:
        st.info("Upload an audio file to see its waveform, spectrogram, and loudness.")
    else:
        audio_payload = audio_upload.getvalue()
        try:
            samples, sample_rate = decode_audio_cached(audio_payload, audio_upload.name)
        except ValueError as exc:
            st.error(str(exc))
        else:
            if sample_rate <= 0:
                st.error("The audio file has an invalid sample rate.")
            else:
                duration = len(samples) / sample_rate
                rms = float(np.sqrt(np.mean(samples**2)))
                peak = float(np.max(np.abs(samples)))
                dbfs = 20 * np.log10(rms) if rms > 0 else float("-inf")

                with st.container(border=True):
                    st.markdown("### Analysis properties")
                    metrics = st.columns(4)
                    metrics[0].metric("Duration", f"{duration:.2f} s")
                    metrics[1].metric("Sample rate", f"{sample_rate:,} Hz")
                    metrics[2].metric("RMS", f"{rms:.5f}")
                    metrics[3].metric("Approx. dBFS", f"{dbfs:.2f} dB")
                    analysis = st.radio(
                        "Choose a view",
                        ["Waveform", "Spectrogram", "Loudness details"],
                        horizontal=True,
                        key="audio_analysis_view",
                    )

                if analysis == "Waveform":
                    stride = max(1, len(samples) // 100_000)
                    times = np.arange(0, len(samples), stride) / sample_rate
                    figure, axis = plt.subplots(figsize=(11, 4))
                    axis.plot(times, samples[::stride], color="#2864A0", linewidth=0.65)
                    axis.set(title="Audio waveform", xlabel="Time (seconds)", ylabel="Amplitude")
                    axis.grid(alpha=0.2)
                    figure.tight_layout()
                    st.pyplot(figure, clear_figure=True)
                    plt.close(figure)
                elif analysis == "Spectrogram":
                    if len(samples) < 16:
                        st.warning("The audio clip is too short to create a spectrogram.")
                    else:
                        nfft = min(1024, 2 ** int(np.floor(np.log2(len(samples)))))
                        figure, axis = plt.subplots(figsize=(11, 4))
                        axis.specgram(samples, NFFT=nfft, Fs=sample_rate, cmap="viridis")
                        axis.set(title="Audio spectrogram", xlabel="Time (seconds)", ylabel="Frequency (Hz)")
                        figure.tight_layout()
                        st.pyplot(figure, clear_figure=True)
                        plt.close(figure)
                else:
                    details = st.columns(3)
                    details[0].metric("Peak amplitude", f"{peak:.5f}")
                    details[1].metric("Samples", f"{len(samples):,}")
                    details[2].metric("Channels", "Mono analysis")
                    st.audio(audio_payload, format=f"audio/{audio_upload.name.rsplit('.', 1)[-1].lower()}")


with transform_tab:
    st.markdown("## Frequency transforms")
    st.markdown(
        '<p class="section-note">Compare an input image with its frequency or wavelet representation.</p>',
        unsafe_allow_html=True,
    )
    show_feature_list(["Discrete Fourier Transform", "Discrete Cosine Transform", "Haar Wavelet"])

    transform_upload = st.file_uploader(
        "Upload an image for transform",
        type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
        key="transform_upload",
    )
    transform_source: np.ndarray | None = None
    transform_result: np.ndarray | None = None
    if transform_upload is not None:
        payload = transform_upload.getvalue()
        signature = upload_signature(payload)
        if signature != st.session_state.get("transform_signature"):
            try:
                transform_source = decode_image(payload, grayscale=True)
                st.session_state["transform_signature"] = signature
                st.session_state["transform_source"] = transform_source
                st.session_state["transform_result"] = None
            except ValueError as exc:
                st.session_state["transform_signature"] = signature
                st.session_state["transform_source"] = None
                st.session_state["transform_result"] = None
                st.error(str(exc))
        transform_source = st.session_state.get("transform_source")
        transform_result = st.session_state.get("transform_result")

    with st.container(border=True):
        st.markdown("### Transform properties")
        transform_name = st.selectbox(
            "Transform",
            ["DFT", "DCT", "Wavelet"],
            format_func=lambda value: {
                "DFT": "DFT — Discrete Fourier Transform",
                "DCT": "DCT — Discrete Cosine Transform",
                "Wavelet": "Haar Wavelet",
            }[value],
            key="transform_name",
        )
        run_transform = st.button(
            "Run transform",
            type="primary",
            disabled=transform_source is None,
            key="run_transform",
            width="stretch",
        )
        if run_transform and transform_source is not None:
            result = apply_transform(transform_source, transform_name)
            if result.dtype != np.uint8:
                result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            st.session_state["transform_result"] = result
            st.session_state["transform_result_name"] = transform_name
            transform_result = result

    if transform_source is not None:
        input_column, output_column = st.columns(2)
        with input_column:
            st.markdown("#### Grayscale input")
            st.image(transform_source, clamp=True, width="stretch")
        with output_column:
            st.markdown("#### Transform output")
            if transform_result is None:
                st.info("Choose a transform and select Run transform.")
            else:
                st.image(transform_result, clamp=True, width="stretch")
                st.download_button(
                    "Download transform",
                    data=image_png_bytes(transform_result),
                    file_name=f"{st.session_state.get('transform_result_name', 'transform').lower()}_result.png",
                    mime="image/png",
                    key="download_transform",
                    width="stretch",
                )
    else:
        st.info("Upload an image to enable DFT, DCT, and Wavelet transforms.")


with video_tab:
    st.markdown("## Video and collage")
    st.markdown(
        '<p class="section-note">Preview a video frame or combine several images into a downloadable collage.</p>',
        unsafe_allow_html=True,
    )
    show_feature_list(["First video frame", "Grayscale preview", "Up to 6 images", "Download collage"])

    video_column, collage_column = st.columns(2, gap="large")
    with video_column:
        with st.container(border=True):
            st.markdown("### Video properties")
            st.caption("Extract the first frame from a supported video file.")
            video_upload = st.file_uploader(
                "Upload video",
                type=["mp4", "avi", "mov", "mkv", "webm"],
                key="video_upload",
            )
            grayscale_preview = st.checkbox("Show the frame in grayscale", key="video_grayscale")
            frame: np.ndarray | None = None
            if video_upload is not None:
                video_payload = video_upload.getvalue()
                video_signature = upload_signature(video_payload)
                if video_signature != st.session_state.get("video_signature"):
                    st.session_state["video_signature"] = video_signature
                    st.session_state["video_frame"] = None
                if st.button("Extract first frame", key="extract_video_frame", type="primary"):
                    try:
                        st.session_state["video_frame"] = read_video_cached(
                            video_payload,
                            video_upload.name,
                        )
                    except ValueError as exc:
                        st.error(str(exc))
                frame = st.session_state.get("video_frame")
                if frame is not None and grayscale_preview:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                if frame is not None:
                    st.image(frame, channels="BGR", width="stretch")
                    st.download_button(
                        "Download frame",
                        data=image_png_bytes(frame),
                        file_name="video_frame.png",
                        mime="image/png",
                        key="download_video_frame",
                        width="stretch",
                    )
            else:
                st.info("Upload a video to extract its first frame.")

    with collage_column:
        with st.container(border=True):
            st.markdown("### Collage properties")
            st.caption("Select up to six images. Each tile is resized to fit a 2-column collage.")
            collage_uploads = st.file_uploader(
                "Upload images",
                type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
                accept_multiple_files=True,
                key="collage_uploads",
            )
            collage_payloads = [upload.getvalue() for upload in collage_uploads]
            collage_signature = (
                hashlib.sha256(b"".join(collage_payloads)).hexdigest()
                if collage_payloads
                else None
            )
            if collage_signature != st.session_state.get("collage_signature"):
                st.session_state["collage_signature"] = collage_signature
                st.session_state["collage_png"] = None
            collage_clicked = st.button(
                "Create collage",
                type="primary",
                disabled=not collage_uploads,
                key="create_collage",
                width="stretch",
            )
            if collage_clicked:
                if len(collage_uploads) > 6:
                    st.warning("Only the first six images will be included.")
                try:
                    collage = create_collage(collage_payloads[:6])
                except ValueError as exc:
                    st.error(str(exc))
                else:
                    st.session_state["collage_png"] = collage
            collage_result = st.session_state.get("collage_png")
            if collage_result is not None:
                st.image(collage_result, width="stretch")
                collage_bytes = io.BytesIO()
                collage_result.save(collage_bytes, format="PNG")
                st.download_button(
                    "Download collage",
                    data=collage_bytes.getvalue(),
                    file_name="image_collage.png",
                    mime="image/png",
                    key="download_collage",
                    width="stretch",
                )
            elif not collage_uploads:
                st.info("Upload one or more images to create a collage.")


st.markdown(
    '<p style="color:#8190a3;text-align:center;padding-top:24px;">Multimedia Studio · Image, audio and video tools</p>',
    unsafe_allow_html=True,
)
