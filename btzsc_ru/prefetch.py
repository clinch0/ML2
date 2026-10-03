"""Фоновая загрузка весов следующей модели, пока GPU считает текущую.

Скачивание и счёт — разные ресурсы (сеть и видеокарта), и держать их по очереди
расточительно: на T4 загрузка модели занимает десятки секунд, за которые видеокарта
простаивает. Здесь один фоновый поток тянет веса следующей модели, пока текущая считается.

Поток демонический и «мягкий»: любая ошибка загрузки только логируется — модель всё равно
скачается обычным путём в момент своей очереди.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor

# Файлы, которые нам не нужны: другие форматы весов и экспорты.
IGNORE_PATTERNS = (
    "*.onnx", "*.onnx_data", "onnx/*", "openvino/*", "*.msgpack", "*.h5", "*.tflite",
    "*.ot", "coreml/*", "rust_model.ot",
)


def prefetch_model(model_id: str, revision: str | None = None, cache_dir: str | None = None) -> str:
    """Скачивает веса в кэш HF. Возвращает человекочитаемый итог."""
    from huggingface_hub import snapshot_download

    try:   # фоновая загрузка не должна печатать свои бары поверх нашего прогресса
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:  # noqa: BLE001
        pass

    snapshot_download(
        repo_id=model_id,
        revision=revision,
        cache_dir=cache_dir,
        ignore_patterns=list(IGNORE_PATTERNS),
    )
    return f"{model_id} в кэше"


class ModelPrefetcher:
    """Качает следующую модель в один фоновый поток.

    Использование:
        with ModelPrefetcher(cache_dir) as pre:
            for i, spec in enumerate(specs):
                pre.request(specs[i + 1] if i + 1 < len(specs) else None)
                ...  # считаем spec на GPU, пока качается следующая
    """

    def __init__(self, cache_dir: str | None = None, enabled: bool = True) -> None:
        self.cache_dir = cache_dir
        self.enabled = enabled
        self._pool: ThreadPoolExecutor | None = None
        self._future: Future | None = None
        self._lock = threading.Lock()
        self.current: str = ""

    def __enter__(self) -> ModelPrefetcher:
        if self.enabled:
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch")
        return self

    def __exit__(self, *exc_info) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    def request(self, model_id: str | None, revision: str | None = None) -> None:
        """Ставит модель в очередь фоновой загрузки (не более одной за раз)."""
        if not self.enabled or self._pool is None or not model_id:
            return
        with self._lock:
            if self._future is not None and not self._future.done():
                return   # предыдущая ещё качается — не плодим параллельные загрузки
            self.current = model_id.split("/")[-1]
            self._future = self._pool.submit(self._run, model_id, revision)

    def _run(self, model_id: str, revision: str | None) -> str:
        try:
            return prefetch_model(model_id, revision, self.cache_dir)
        except Exception as exc:  # noqa: BLE001
            return f"фоновая загрузка {model_id} не удалась ({type(exc).__name__}); скачается в свою очередь"
        finally:
            with self._lock:
                self.current = ""

    @property
    def status(self) -> str:
        """Короткая метка для строки прогресса."""
        with self._lock:
            return self.current
