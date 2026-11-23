"""텍스트 명령으로 엘리베이터 3층 버튼을 누른다.

예:
  python run_button_task.py "3층 버튼을 눌러줘"
  python run_button_task.py --mode state "엘리베이터 3층을 선택해 줘"
  python run_button_task.py --eval
  python run_button_task.py --video
"""

import argparse
from pathlib import Path

from PIL import Image

from button_task.evaluate import check_language, evaluate, save_summary
from button_task.task import ButtonTask


def main() -> None:
    parser = argparse.ArgumentParser(description="엘리베이터 3층 버튼 작업")
    parser.add_argument("text", nargs="?", default="3층 버튼을 눌러줘")
    parser.add_argument("--mode", choices=("vision", "state"), default="vision")
    parser.add_argument("--eval", action="store_true", help="기본 10 + 변화 10 + 명령 5 평가")
    parser.add_argument("--video", action="store_true", help="성공 시연 영상을 저장")
    parser.add_argument("--nlu", choices=("rules", "llm"), default="llm")
    parser.add_argument("--classify-only", action="store_true", help="물리 동작 없이 의도만 분류")
    args = parser.parse_args()

    if args.eval:
        check_language()
        summary = evaluate()
        path = Path("results") / "button_eval.json"
        save_summary(summary, path)
        print(
            f"상태 기본 {summary['state_basic_sr']:.0f}% / "
            f"변화 {summary['state_variation_sr']:.0f}% / "
            f"시각 기본 {summary['vision_basic_sr']:.0f}% / "
            f"변화 {summary['vision_variation_sr']:.0f}% / "
            f"명령 {summary['command_appropriateness']:.0f}%"
        )
        print(f"저장: {path}")
        return

    if args.classify_only:
        from button_task.nlu_llm import ExaoneClassifier
        from button_task.nlu import classify_command

        classify = ExaoneClassifier() if args.nlu == "llm" else classify_command
        intent = classify(args.text)
        print(f"intent={intent.action} nlu_ms={intent.nlu_ms:.0f}")
        print(intent.reason)
        return

    task = ButtonTask(nlu=args.nlu)
    result, frames = task.run(args.text, mode=args.mode, record=args.video)
    print(
        f"intent={result.intent} nlu={task.nlu_name} success={result.success} "
        f"button={result.button_m * 1000:.1f} mm time={result.duration_s:.2f}s "
        f"nlu_ms={result.nlu_ms:.0f}"
    )
    print(result.reason)
    if result.failure:
        print(result.failure)
    if result.perception:
        print(result.perception)
    if args.video:
        if not frames:
            print("저장할 프레임이 없습니다.")
            return
        out = Path("results") / f"press_{args.mode}.gif"
        out.parent.mkdir(parents=True, exist_ok=True)
        images = [Image.fromarray(frame) for frame in frames]
        images[0].save(
            out,
            save_all=True,
            append_images=images[1:],
            duration=33,
            loop=0,
        )
        print(f"저장: {out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
