"""EXAONE 4.0 1.2B로 한국어 명령을 로컬에서 분류한다.

가중치는 LGAI-EXAONE/EXAONE-4.0-1.2B-GGUF 의 Q4_K_M 이다.
라이선스는 EXAONE AI Model License Agreement 1.2 - NC 이며, 교육 목적 사용이 허용된다.
추론은 이 컴퓨터의 llama.cpp에서만 수행하고 외부 API를 호출하지 않는다.
"""

from pathlib import Path
import subprocess
import tempfile
import time
import zipfile
import urllib.request

from button_task.nlu import Intent, classify_command

REPO_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B-GGUF"
MODEL_FILE = "EXAONE-4.0-1.2B-Q4_K_M.gguf"
LLAMA_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download/b12651/"
    "llama-b12651-bin-win-cpu-x64.zip"
)
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
LLAMA_DIR = ROOT / "tools" / "llama-b12651-bin-win-cpu-x64"
LABELS = ("press_3", "wait", "reject", "ask")

SYSTEM_PROMPT = """너는 휠체어 로봇 팔의 명령 분류기다. 장면에는 엘리베이터 3층 버튼 하나만 있다.
사용자 문장을 보고 아래 네 라벨 중 정확히 하나만 출력한다. 설명은 쓰지 않는다.
press_3: 3층, 삼층, 세 번째 층을 누르거나 선택하라는 요청
wait: 버튼을 누르지 말거나 기다리거나 멈추라는 요청
reject: 3층이 아닌 다른 층을 누르라는 요청
ask: 몇 층인지 말하지 않았거나, 동작이 불명확하거나, 버튼 작업이 아닌 말"""


def model_path() -> Path:
    return MODEL_DIR / MODEL_FILE


def ensure_model() -> Path:
    path = model_path()
    if path.is_file() and path.stat().st_size > 0:
        return path
    from huggingface_hub import hf_hub_download

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"로컬 모델 다운로드: {REPO_ID} / {MODEL_FILE}")
    downloaded = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILE,
        local_dir=str(MODEL_DIR),
    )
    return Path(downloaded)


def ensure_llama() -> Path:
    for name in ("llama-completion.exe", "llama-cli.exe"):
        candidate = LLAMA_DIR / name
        if candidate.is_file():
            return candidate
    LLAMA_DIR.mkdir(parents=True, exist_ok=True)
    archive = LLAMA_DIR / "llama-cpu.zip"
    if not archive.is_file():
        print("llama.cpp CPU 실행 파일 다운로드")
        urllib.request.urlretrieve(LLAMA_URL, archive)
    with zipfile.ZipFile(archive) as package:
        package.extractall(LLAMA_DIR)
    for name in ("llama-completion.exe", "llama-cli.exe"):
        matches = list(LLAMA_DIR.rglob(name))
        if matches:
            return matches[0]
    raise FileNotFoundError("llama.cpp 실행 파일을 찾지 못했습니다.")


def format_prompt(text: str) -> str:
    """EXAONE 4.0 비추론 모드 대화 형식. 생각 블록은 비운다."""
    return (
        "[|system|]\n"
        f"{SYSTEM_PROMPT}\n"
        "[|endofturn|]\n"
        "[|user|]\n"
        f"{text.strip()}\n"
        "[|endofturn|]\n"
        "[|assistant|]\n"
        "<think>\n\n</think>\n\n"
    )


def mentions_floor3(text: str) -> bool:
    compact = "".join(text.split())
    return any(token in compact for token in ("3층", "3번", "삼층", "삼번", "세번째", "셋째", "세층"))


def parse_label(generated: str) -> str | None:
    compact = generated.strip().lower().replace("-", "_")
    for label in LABELS:
        if label in compact:
            return label
    return None


class ExaoneClassifier:
    """문장마다 로컬 EXAONE을 호출하고, 금지·없는 층은 규칙이 최종 차단한다."""

    def __init__(self, gguf_path: Path | None = None):
        self.gguf = Path(gguf_path) if gguf_path else ensure_model()
        self.executable = ensure_llama()

    def __call__(self, text: str) -> Intent:
        started = time.perf_counter()
        generated = self._generate(format_prompt(text))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        label = parse_label(generated)
        rule = classify_command(text)
        if label is None:
            return Intent(
                rule.action,
                f"EXAONE 출력을 라벨로 읽지 못해 규칙을 사용함: {generated.strip()!r}",
                nlu_ms=elapsed_ms,
            )
        if rule.action in {"wait", "reject"} and label == "press_3":
            return Intent(
                rule.action,
                f"EXAONE은 press_3였으나 금지·부재 규칙으로 {rule.action} 처리. {rule.reason}",
                nlu_ms=elapsed_ms,
            )
        if label == "press_3" and rule.action == "ask" and not mentions_floor3(text):
            return Intent(
                "ask",
                "EXAONE은 press_3였으나 층수가 없어 확인 질문으로 처리",
                nlu_ms=elapsed_ms,
            )
        return Intent(label, f"EXAONE 4.0 1.2B 분류: {label}", nlu_ms=elapsed_ms)

    def _generate(self, prompt: str) -> str:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", suffix=".txt", delete=False
        ) as handle:
            handle.write(prompt)
            prompt_path = handle.name
        command = [
            str(self.executable),
            "-m",
            str(self.gguf),
            "-f",
            prompt_path,
            "-c",
            "1024",
            "-n",
            "8",
            "--temp",
            "0.1",
            "--top-k",
            "1",
            "-ngl",
            "0",
            "--no-display-prompt",
            "-no-cnv",
            "--no-warmup",
            "--simple-io",
            "--reasoning",
            "off",
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        Path(prompt_path).unlink(missing_ok=True)
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(f"EXAONE 실행 실패: {detail[-800:]}")
        return completed.stdout


if __name__ == "__main__":
    classifier = ExaoneClassifier()
    samples = [
        ("3층 버튼을 눌러줘", "press_3"),
        ("엘리베이터 3층을 선택해 줘", "press_3"),
        ("삼층 좀 찍어 주라", "press_3"),
        ("버튼은 누르지 말고 대기해", "wait"),
        ("팔은 움직이지 말고 정지해", "wait"),
        ("현재 장면에 없는 7층 버튼을 눌러줘", "reject"),
        ("버튼 눌러줘", "ask"),
        ("세 번째 층 버튼을 골라줘", "press_3"),
        ("오늘 날씨 어때", "ask"),
    ]
    matched = 0
    for text, expected in samples:
        intent = classifier(text)
        ok = intent.action == expected
        matched += int(ok)
        print(f"{ok} {intent.action:8} expected={expected:8} {intent.nlu_ms:.0f}ms {text}")
        print(f"  {intent.reason}")
    print(f"{matched}/{len(samples)}")
