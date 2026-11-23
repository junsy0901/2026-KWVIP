# 엘리베이터 3층 버튼 누르기

전동휠체어에 달린 PiPER 팔이 MuJoCo에서 한국어 명령을 받고 엘리베이터 3층 버튼을 누른다. 문장 분류만 로컬 언어 모델이 하고, 팔 동작은 카메라로 찾은 버튼 좌표와 역기구학으로 만든다. 학습된 동작 모델은 없다.

## 사용 모델

| 역할 | 이름 | 실행 위치 |
| --- | --- | --- |
| 문장 분류 | EXAONE 4.0 1.2B, `EXAONE-4.0-1.2B-Q4_K_M.gguf` | 노트북 CPU, llama.cpp |
| 버튼 인식 | 없음. 파란 픽셀과 카메라 광선 | 같은 프로세스 |
| 팔 제어 | 없음. 감쇠 최소제곱 역기구학 | 같은 프로세스 |

EXAONE 라이선스는 EXAONE AI Model License Agreement 1.2 - NC이며, 교육 목적 사용이 허용된다. 문장은 외부 API로 보내지 않는다.

분류 결과는 네 가지다.

- `press_3`: 3층 버튼을 누른다.
- `wait`: 누르지 말고 기다린다.
- `reject`: 장면에 없는 층이라 거절한다.
- `ask`: 층수나 동작이 불명확해 확인이 필요하다.

금지 문장과 없는 층은 언어 모델 결과를 규칙이 한 번 더 막는다. 층수가 없는데 모델이 누르기라고 하면 확인 질문으로 바꾼다.

## 동작

1. 사용자 문장을 EXAONE이 위 네 라벨 중 하나로 나눈다.
2. `press_3`가 아니면 팔은 움직이지 않는다.
3. `head_rgb` 영상에서 파란 버튼의 중심 픽셀을 찾는다.
4. 카메라 위치, 방향, 화각으로 그 픽셀에 광선을 쏴 버튼 표면의 3차원 점을 구한다.
5. 팔 끝 `ee_site`가 그 점 앞에 오도록 관절 각도를 계산하고, 이어서 패널 방향(-Y)으로 민다.
6. `button_3_slide`가 10 mm 이상이면 성공이고, 팔을 뒤로 뺀다.

팔 끝 좌표는 카메라가 아니라 시뮬레이터 관절 상태에서 읽는다. 카메라가 계산하는 좌표는 버튼 표면뿐이다.

성공 예: `엘리베이터 3층을 선택해 줘`에서 버튼 표면을 보고 15.3 mm를 눌렀다. 분류 시간은 약 2.3초다.

## 실행

```powershell
python run_button_task.py "엘리베이터 3층을 선택해 줘"
python run_button_task.py --classify-only "세 번째 층 버튼을 골라줘"
python run_button_task.py --video "3층 버튼을 눌러줘"
python run_button_task.py --eval
```

`--mode state`는 카메라 대신 `button_target` 좌표를 쓴다. `--nlu rules`는 EXAONE 없이 단어 규칙만 쓴다. `--eval`의 물리 시험은 규칙 분류로 돈다.

영상은 `results/press_vision.gif`, 반복 평가 수치는 `results/button_eval.json`에 저장된다. 옵션 없는 실행은 터미널에만 출력한다.

## 파일

### `wheelchair_piper_lite.xml`

제공된 장면이다. 휠체어는 고정이고, PiPER 6축과 그리퍼, 머리 카메라 `head_rgb`, 손목 카메라 `wrist_rgb`, 엘리베이터 버튼 하나(`elevator_button_3`, `button_3_slide` 0–15 mm)가 있다. 버튼 표면에는 숫자 3이 없다. 팔 몸통 충돌은 꺼져 있다.

### `run_button_task.py`

명령행 입구다. 기본은 EXAONE 분류와 카메라 인식이다. 한 문장 실행, 의도만 출력, 영상 저장, 반복 평가를 고른다.

### `button_task/task.py`

한 번의 명령을 물리 시뮬레이션으로 수행한다. 의도 분류, 목표 좌표, 접근, 가압, 10 mm 성공 판정, 후퇴를 순서대로 호출하고 시도 결과를 모은다.

### `button_task/nlu_llm.py`

EXAONE 4.0 1.2B를 llama.cpp로 실행한다. 모델 파일은 `models/EXAONE-4.0-1.2B-Q4_K_M.gguf`, 실행 파일은 `tools/llama-b12651-bin-win-cpu-x64`에 있다. 프롬프트는 네 라벨만 출력하게 하고, 규칙으로 위험하거나 층수가 없는 경우를 보정한다.

### `button_task/nlu.py`

단어 규칙 분류기다. `3층`과 `눌러`/`선택`이면 누르기, `대기`/`누르지`이면 정지, `7층`처럼 다른 층이면 거절이다. EXAONE의 안전장치로도 쓰인다.

### `button_task/perceive.py`

`head_rgb`에서 파란 영역을 찾고, 카메라 내부·외부 파라미터로 픽셀 광선을 만들어 버튼 또는 패널에 닿는 3차원 점을 구한다.

### `button_task/ik.py`

끝점 `ee_site`와 목표 좌표의 위치 오차를 감쇠 최소제곱으로 줄여 6개 팔 관절 각도를 낸다.

### `button_task/scene.py`

XML을 불러올 때 끝점과 같은 자리(`link6`, z=0.15 m)에 반지름 12 mm 충돌 구 `ee_pusher`를 넣는다. 배포 모델은 팔이 버튼에 닿지 않아 이 구가 버튼을 민다. 시연 카메라 `eval_cam`도 여기서 추가한다. 패널 위치와 초기 관절을 시험 조건으로 바꾼다.

### `button_task/evaluate.py`

같은 성공 기준으로 기본 조건 10회, 패널 위치를 옮긴 변화 조건 10회, 명령 점검 5회를 상태 기반과 카메라 기반으로 측정한다. 성공률, 일반화 격차, 명령 적합성, 수행 시간, 계획 시간, 실패 유형을 `results/button_eval.json`에 저장한다.

### `button_task/__init__.py`

성공 기준 상수 `0.010` m를 둔다.

### `requirements.txt`

MuJoCo, NumPy, Pillow, imageio, huggingface_hub. 언어 모델 실행은 Python 바인딩 대신 llama.cpp 실행 파일을 쓴다.
