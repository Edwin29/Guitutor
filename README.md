# guitutor — MVP

비전으로 **왼손 운지**를 인식해 **잘못된 손가락**을 표시하는 튜터.  


## 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
# (옵션) 손 키포인트 추적
pip install "guitutor[vision]"
```

## 빠른 실행

```bash
PS> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
PS> .\.venv\Scripts\Activate.ps1
# 0) 캘리브레이션 (프렛보드 꼭짓점 4점 클릭)
(.venv) PS> guitutor calibrate --video 0 --save config\calib.yaml
## 또는
(.venv) PS> guitutor calibrate --video data\samples\fretboard.mp4 --save config\calib.yaml


# 1) 실시간/파일 분석
guitutor run --source 0 --config config/default.yaml --rules config/fingerings.yaml

guitutor run --source 0 --config config/default.yaml --rules config/fingerings.yaml --width 640 --height 480 --fps 30

# 또는 영상 파일
guitutor run --source data/samples/demo.mp4 --config config/default.yaml --rules config/fingerings.yaml
```

## 범위 제한 

- 고정 카메라(좌측 상단 45° 근접), **왼손만** 지원 (바레/탭 제외)
- **1~5프렛**, **6개 줄** 매핑
- 대상: 오픈 코드 5종(C/G/D/A/E)
- 프렛보드 4점 **수동 캘리브레이션** → 호모그래피로 정사영
- 손가락 끝(1,2,3,4) 중심의 **셀 충돌** 기반 매핑

추후 확장: 스케일 패턴, 바레, 다중 각도, 3D 깊이 추정 등

