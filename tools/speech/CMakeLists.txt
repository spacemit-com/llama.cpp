add_subdirectory(backends/qwen3_tts)

add_custom_target(speech-install
    DEPENDS q3tts-install
)
