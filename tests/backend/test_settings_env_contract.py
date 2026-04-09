"""
Focused config test to ensure Docker environment variables reach BaseSettings.
"""

import os

from backend.api.config import Settings


def _exercise_env_contract():
    original = {key: os.environ.get(key) for key in (
        "INDTEXTS_HOST",
        "INDTEXTS_USE_FP16",
        "INDTEXTS_USE_DEEPSPEED",
        "INDTEXTS_USE_CUDA_KERNEL",
        "INDTEXTS_SIMILARITY_THRESHOLD",
    )}

    try:
        os.environ["INDTEXTS_HOST"] = "127.0.0.1"
        os.environ["INDTEXTS_USE_FP16"] = "false"
        os.environ["INDTEXTS_USE_DEEPSPEED"] = "true"
        os.environ["INDTEXTS_USE_CUDA_KERNEL"] = "true"
        os.environ["INDTEXTS_SIMILARITY_THRESHOLD"] = "0.77"

        settings = Settings()

        assert settings.host == "127.0.0.1", settings.host
        assert settings.fp16_mode == "false", settings.fp16_mode
        assert settings.use_deepspeed is True, settings.use_deepspeed
        assert settings.use_cuda_kernel is True, settings.use_cuda_kernel
        assert abs(settings.similarity_threshold - 0.77) < 1e-9, settings.similarity_threshold
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if __name__ == "__main__":
    _exercise_env_contract()
    print("test_settings_env_contract: PASS")
