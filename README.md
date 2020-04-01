# Transformer 

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en
python -m spacy download de
```


## Preprocessing

학습 데이터를 다운로드 받고, 전처리를 합니다. 

```bash
python preprocess.py
```

## Training

default 세팅 값으로 학습시키려면 간단히 다음과 같이 합니다.

```bash
python train.py
```

그외 hyper-parameter 변경의 예제는 다음과 같습니다. 

```
python3.6 train.py --batch_size=64
```

Cloud에서 학습시 nohup사용은 유용합니다.

```bash
nohup python train.py > .train.log &
```