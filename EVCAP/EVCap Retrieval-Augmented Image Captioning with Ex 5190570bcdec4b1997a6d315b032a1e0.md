# EVCap: Retrieval-Augmented Image Captioning with External Visual-Name Memory

Keywords: Lightweight Image-captioning, Multimodality, Retrieval Augmentation
학회: CVPR, Filckr30k, NoCaps, WHOOPS
Dataset: COCO, LVIS
리뷰 진행 상태: 논문리뷰 진행중
관련 연구: smallCAP
진행 일시: 2024년 7월 1일
논문 주소: https://arxiv.org/pdf/2311.15879
정리글 주소: https://junyeongson.notion.site/EVCAP-Retrieval-Augmented-Image-Captioning-with-External-Visual-Name-Memory-for-Open-World-Comprehe-51acc546f45f49269fc0556476e7223f
구현 진행 상태: 시작 전
year: 2024

# 1. Motivation

## Lightweight training in Image Captioning

- VLM 기반 image captioning은 대규모 데이터셋에 대해 학습된 대규모 모델
    - 계산 비용이 높고, open world에서의 새로운 이미지에 대해 학습 데이터를 업데이트하는 것은 거의 불가능함
- 즉 open world에 대한 지속적인 이해를 위해서는 LLM을 합리적 비용으로 Object knowledge를 유지하도록 하는 것이 중요하며, 이를 위한 training 방식으로 Lightweight training 방식이 대두됨

![Untitled](EVCap%20Retrieval-Augmented%20Image%20Captioning%20with%20Ex%205190570bcdec4b1997a6d315b032a1e0/Untitled.png)

## Retrieval-augmented image captioning

- time, data에서 모두 training cost를 줄이고 높은 성능을 보임
- large datastore를 활용하므로 LLMs이 주어진 texts를 단순히 모방하는 것으로 보이며 open world objects를 적절하게 묘사하는 능력이 떨어짐
- 많은 retrieved texts을 LLMs의 prompting에 포함하는 것은 번거로우며, 더 많은 trainable parameters를 요구
- 새로운 objects가 빈번하게 발생하기 때문에 datastore 내 sample texts를 항상 사용하는 것은 불가능하며, 사용되는 memory의 양 또한 늘리기 어려움

# 2. Methods

![Untitled](EVCap%20Retrieval-Augmented%20Image%20Captioning%20with%20Ex%205190570bcdec4b1997a6d315b032a1e0/Untitled%201.png)

- **expanded external memory**를 이용해 retrieval-augmented LLMs-based image captioning model 구축
- retrieved object names 사용해 효과적인 LLM base model 설계
    - frozen vision model + trainable LLMs
    - visual feature +input image embedding을 매칭하고 **object names(value)**을 retrieving by **external visual name memory**
    - attention fusion module을 이용해 관련없는 object name 제거,
    - attention fusion에 따라 학습된 visual feature와 ojbect name feature 결합해 prompt 형성
    - llm이 caption 생성

## 1. External visual-name memory

![Untitled](EVCap%20Retrieval-Augmented%20Image%20Captioning%20with%20Ex%205190570bcdec4b1997a6d315b032a1e0/Untitled%202.png)

- external data source를 통해 image-name pair를 수집하고 해당 image를 32번 encoding한 image embedding의 평균을 memory의 key, name을 value로 사용

**external data source**

- LVIS dataset에서 1203개의 object와 각 object에서 1~10개 랜덤 이미지 선택해 총 8581개의 object images 구성.
- synthetic image도 포함. a photo of {object name}이라는 prompt를 기반으로 stable diffusion으로 object name마다 5개의 추가 이미지 생성
- 총 8581 + 5*1203 = 14596개의 이미지로 external memory 구성위한 data source 확보

![Untitled](EVCap%20Retrieval-Augmented%20Image%20Captioning%20with%20Ex%205190570bcdec4b1997a6d315b032a1e0/Untitled%203.png)

**external memory construction**

- 각 이미지 $X^i$에 대해 frozen vision encoder $\epsilon()$이용해 (1*768) 크기의 임베딩 32개로 projection함
    - $\left\{k_1^i , k_2^i, ..., k_{32}^i\right\} = \varepsilon(X^i)$
- 32개 임베딩의 평균을 계산해 1*768 크기의 single embedding $k^i$ 생성해 M의 key로 사용, 짝을 이루는 value는 object name $v^i$
- visual name memory인 $M=\left\{(k^i, v^i)\right\}^M_{i=1}$ 생성하고 이를 FAISS를 통해 인덱싱함

## 2. Object names retrieval

![Untitled](EVCap%20Retrieval-Augmented%20Image%20Captioning%20with%20Ex%205190570bcdec4b1997a6d315b032a1e0/Untitled%204.png)

**Image encoding**

- input 이미지 X와 image query token T를 frozen vision encoder에 입력해 visual feature Q 생성
    - $Q = \varepsilon(X, T_{img})$
- frozen vision encoder는 BLIP2의 구조 ViT + Q-Former 사용
    - Vit: 257*1408 크기의 image feature를 output으로 생성
    - image feature 를 받아 Q=32개의 학습된 visual feature 생성. 각 크기는 1*768
    - $Q = \left\{q_1, q_2, ..., q_{32} \right\}$

**Retrieval**

- query $q_j \in Q$와 external image-name memory M의 key $k_i \in M$간의 cosine similarity 계산
- 주어진 각 q_j 중 가장 similarity가 높은 하나의 key 선택해 총 32개의 key-value candidates $\left\{k^{best}_j, v^{best}_j \right\}^{32}_{j=1}$ 생성
- 이 중 중복되는 object name(value)은 제거한 후, 남은 값 중 top-K value 선택
- 선택된 top-K value $v_j^{best}$는 input 이미지에 대한 retrieved top-K object names $v_l$로 재정의
    - $l \in [1,K]$
    - retrieved top-$K$ object names : $\left\{v_l \right\}^K_{l=1}$

![Untitled](EVCap%20Retrieval-Augmented%20Image%20Captioning%20with%20Ex%205190570bcdec4b1997a6d315b032a1e0/Untitled%205.png)

- K를 1~20까지 설정하여 확인했을때, K의 값이 클수록 attention module을 통해 성능이 향상됨을 확인
- 특히 K=10에서 모든 데이터셋에서의 성능이 골고루 좋은 것을 확인

![Untitled](EVCap%20Retrieval-Augmented%20Image%20Captioning%20with%20Ex%205190570bcdec4b1997a6d315b032a1e0/Untitled%206.png)

## 3. Attention Fusion

![Untitled](EVCap%20Retrieval-Augmented%20Image%20Captioning%20with%20Ex%205190570bcdec4b1997a6d315b032a1e0/Untitled%207.png)

- retrieval을 통해 얻은 object name 중 불필요한 부분을 attention fusion을 이용해 제거함으로서 object name features를 선택적으로 추출하는 과정
    - S, Q, T를 customized Q-Former F()에 입력
    - $V = F(S,Q,T_{obj})$
- $S$: Retrieval object names v_l을 구분자[SEP]로 분리하여 생성함. $S=\left\{v_1, [SEP], v_2, [SEP], … , [SEP], v_K \right\}$
- $Q$: 이전 단계에서 생성된 visual feature
- $T_{obj}$: caption 관련 object name feature를 학습하기 위해 학습동안 생성된 learnable object name query token

## 4. Caption Generation

![Untitled](EVCap%20Retrieval-Augmented%20Image%20Captioning%20with%20Ex%205190570bcdec4b1997a6d315b032a1e0/Untitled%208.png)

- LLM에 입력 전 변수들 통합. Q(visual feature)와 V(object name feature)를 concatenate하고 linear layer $\phi(\cdot)$를 통해 LLM의 latent space로 projection함
- $\phi(Q \oplus V)$
- LLM은 vicuna-13B 사용. caption generation 용도이며 opensource chatbot으로 작용하기 위해 llama를 pretraining한 모델임
- 대화 형식의 포멧으로 prompt 구성

![Untitled](EVCap%20Retrieval-Augmented%20Image%20Captioning%20with%20Ex%205190570bcdec4b1997a6d315b032a1e0/Untitled%209.png)

- 학습 단계에서 input caption tokens $\left\{c_i \right\}^L_{l=1}$이 주어지면, LLM decoder는 `embedded prompt` $\left\{w_i \right\}^N_{l=1}$와 embedded caption tokens $\left\{c_i \right\}^L_{l=1}$를 concatenate하고, `autoregressive` 방식으로 caption tokens을 예측
- end to end 방식으로 cross entrophy 최소화하는 방향으로 학습시킴
    - $L_{\theta} = -\sum_{i=1}^{L}logp_{\theta}(c_i | w_1, …, w_N, c_1, … , c_{i-1})$

# 3. Experiments

# 4. Contributions & Limitation

# 5. Ablation Study

# 6. Comment

- external visual name memory에 cultual visual name을 넣어서 top k를 구한다?
- customized q former 부분을 수정해서 cultural aspect를 더 잘 반영? attention 수정
- attention에서 value 계산시 loss 변경에 따른 value값 변화 확인
- image embedding 방식 변화? 세부적으로 보는 방향 의료쪽 VQ-GAN

## questions

- 왜 vit, q-former를 image, txt에 따라 나누는지
    - frozen인데 나눌 필요?
- attention mask 정의?

```markdown
# QFormer
        if attention_mask.dim() == 3:

```