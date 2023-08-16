# Amazon Bedrock Workshop

개발자와 솔루션 빌더를 대상으로 하는 이 실습 워크샵에서는 [Amazon Bedrock](https://aws.amazon.com/bedrock/)을 통해 파운데이션 모델(FM)을 활용하는 방법을 소개합니다.

Amazon Bedrock은 타사 제공업체 및 Amazon의 FM에 대한 액세스를 제공하는 완전 관리형 서비스로, API를 통해 이용할 수 있습니다. 베드락을 사용하면 다양한 모델 중에서 사용 사례에 가장 적합한 모델을 찾을 수 있습니다.

이 실습 시리즈에서는 Generative AI에 대해 고객들이 가장 많이 사용하는 몇 가지 사용 패턴을 살펴볼 것입니다. 텍스트와 이미지를 생성하여 생산성을 향상시킴으로써 조직의 가치를 창출하는 기술을 보여드리겠습니다. 이는 이메일 작성, 텍스트 요약, 질문에 대한 답변, 챗봇 구축, 이미지 생성에 도움이 되는 기초 모델을 활용하여 달성할 수 있습니다. 이 과정에서는 베드락 API와 SDK, 그리고 [LangChain](https://python.langchain.com/docs/get_started/introduction) 및 [FAISS](https://faiss.ai/index.html)와 같은 오픈소스 소프트웨어를 통해 이러한 패턴을 구현하는 실무 경험을 쌓을 수 있습니다.

실험실에는 다음이 포함됩니다:

- **Text Generation** \[Estimated time to complete - 30 mins\]
- **Text Summarization** \[Estimated time to complete - 30 mins\]
- **Questions Answering** \[Estimated time to complete - 45 mins\]  
- **Chatbot** \[Estimated time to complete - 45 mins\]
- **Image Generation** \[Estimated time to complete - 30 mins\]

<div align="center">

![imgs/10-overview](imgs/10-overview.png "Overview of the different labs in the workshop")

</div>

워크샵 웹사이트의 [단계별 안내 지침](https://catalog.us-east-1.prod.workshops.aws/workshops/a4bdb007-5600-4368-81c5-ff5b4154f518/en-US)을 참조할 수도 있습니다.


## 시작하기

### 노트북 환경 선택

이 워크샵은 원하는 환경에서 실행할 수 있는 일련의 **Python 노트북**으로 제공됩니다:

- 풍부한 AI/ML 기능을 갖춘 완전 관리형 환경의 경우, [세이지메이커 스튜디오](https://aws.amazon.com/sagemaker/studio/)를 사용하는 것을 권장합니다. 빠르게 시작하려면 [도메인 빠른 설정 지침](https://docs.aws.amazon.com/sagemaker/latest/dg/onboard-quick-start.html)을 참조하세요.

- 완전 관리형이지만 좀 더 기본적인 환경을 원하시면 [SageMaker Notebook 인스턴스 만들기](https://docs.aws.amazon.com/sagemaker/latest/dg/howitworks-create-ws.html)를 사용하실 수 있습니다.

- 기존(로컬 또는 기타) 노트북 환경을 사용하시려면 [AWS 호출을 위한 자격 증명](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html)이 있는지 확인하세요.

### 베드락에 대한 AWS IAM 권한 활성화

노트북 환경에서 가정하는 AWS ID(SageMaker의 [*스튜디오/노트북 실행 역할*](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html)이거나 자체 관리 노트북의 역할 또는 IAM 사용자일 수 있음)는 Amazon 베드락 서비스를 호출할 수 있는 충분한 [AWS IAM 권한](https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies.html)을 가지고 있어야 합니다.

유저의 ID에 대한 접근 권한을 부여하려면 다음과 같이 하세요:

- [AWS IAM 콘솔](https://us-east-1.console.aws.amazon.com/iam/home?#)을 엽니다.
- [역할](https://us-east-1.console.aws.amazon.com/iamv2/home?#/roles)(세이지메이커를 사용 중이거나 IAM 역할을 맡고 있는 경우) 또는 [사용자](https://us-east-1.console.aws.amazon.com/iamv2/home?#/users)를 찾습니다.
- 권한 추가 > 인라인 정책 만들기*를 선택하여 새 인라인 권한을 첨부하고, *JSON* 편집기를 열어 아래 예제 정책에 붙여넣습니다:

```
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "BedrockFullAccess",
            "Effect": "Allow",
            "Action": ["bedrock:*"],
            "Resource": "*"
        }
    ]
}
```
>⚠️ **참고:** Amazon SageMaker를 사용하면 노트북 실행 역할은 일반적으로 AWS 콘솔에 로그인하는 사용자 또는 역할과 *별개*로 설정됩니다. Amazon 베드락용 AWS 콘솔을 탐색하려면 콘솔 사용자/역할에도 권한을 부여해야 합니다.

베드락의 세분화된 작업 및 리소스 권한에 대한 자세한 내용은 베드락 개발자 가이드를 확인하세요.

### Clone and use the notebooks

> ℹ️ **참고: 세이지메이커 스튜디오에서 *파일 > 새로 만들기 > 터미널*을 클릭하여 "시스템 터미널"을 열어 이러한 명령을 실행할 수 있습니다.

노트북 환경이 설정되면 이 워크샵 리포지토리를 노트북에 복제(clone)합니다.

```sh
git clone https://github.com/dongjin-ml/amazon-bedrock-workshop-webinar-kr.git
cd amazon-bedrock-workshop-webinar-kr
```

이 서비스는 프리뷰 버전이기 때문에 Amazon Bedrock SDK는 아직 [파이썬용 AWS SDK - boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)의 표준 릴리스에 포함되어 있지 않습니다. 다음 스크립트를 실행하여 베드락 테스트를 위한 사용자 정의 SDK 휠을 다운로드하고 추출하세요:

```sh
bash ./download-dependencies.sh
```

이 스크립트는 'dependencies' 폴더를 생성하고 관련 SDK를 다운로드하지만 아직 'pip install'는 하지 않습니다.

이제 실습용 노트북을 탐색할 준비가 되었습니다! 00_Intro/bedrock_boto3_setup.ipynb](00_Intro/bedrock_boto3_setup.ipynb)부터 시작하여 Bedrock SDK를 설치하고, 클라이언트를 생성하고, Python에서 API 호출을 시작하는 방법에 대한 자세한 내용을 확인하세요.

## Content

이 리포지토리에는 베드락 아키텍처 패턴 워크숍을 위한 노트북 예제가 포함되어 있습니다. 노트북은 모듈별로 다음과 같이 구성되어 있습니다:

### Intro

- [Simple Bedrock Usage](./00_Intro/bedrock_boto3_setup.ipynb): 이 노트북은 boto3 클라이언트 설정과 베드락의 기본적인 사용법을 보여줍니다.

### Generation

- [Simple use case with boto3](./01_Generation/00_generate_w_bedrock.ipynb): 이 노트북에서는 Amazon Bedrock을 사용해 텍스트를 생성합니다. boto3를 사용하여 Amazon Titan 모델을 직접 소비하는 데모를 보여드립니다. 
- [Simple use case with LangChain](./01_Generation/01_zero_shot_generation.ipynb): 그런 다음 동일한 작업을 수행하지만 널리 사용되는 프레임워크인 LangChain을 사용합니다.
- [Generation with additional context](./01_Generation/02_contextual_generation.ipynb): 그런 다음 응답을 개선하기 위해 추가 컨텍스트가 포함된 프롬프트를 개선하여 한 단계 더 나아갑니다.

### Summarization

- [Small text summarization](./02_Summarization/01.small-text-summarization-claude.ipynb): 이 노트북에서는 베드락을 사용해 작은 텍스트를 요약하는 간단한 작업을 수행합니다. 
- [Long text summarization](./02_Summarization/02.long-text-summarization-titan.ipynb): 요약할 콘텐츠가 점점 커져 모델의 최대 토큰을 초과하면 위의 접근 방식이 작동하지 않을 수 있습니다. 이 노트북에서는 파일을 작은 덩어리로 나누고, 각 덩어리를 요약한 다음 요약을 요약하는 접근 방식을 보여드립니다.
### Question Answering

- [Simple questions with context](./03_QuestionAnswering/00_qa_w_bedrock_titan.ipynb): 이 노트북은 모델을 직접 호출하여 주어진 컨텍스트에 따라 질문에 답하는 간단한 예제를 보여줍니다. 
- [Answering questions with Retrieval Augmented Generation](./03_QuestionAnswering/01_qa_w_rag_claude.ipynb): 검색 증강 생성(RAG)이라는 아키텍처를 구현하여 위의 프로세스를 개선할 수 있습니다. RAG는 언어 모델 외부(비매개변수)에서 데이터를 검색하고 검색된 관련 데이터를 컨텍스트에 추가하여 프롬프트를 증강합니다.

### Chatbot

- [Chatbot using Claude](./04_Chatbot/00_Chatbot_Claude.ipynb): 이 노트북은 Claude를 사용하는 챗봇을 보여줍니다.
- [Chatbot using Titan](./04_Chatbot/00_Chatbot_Titan.ipynb): 이 노트북은 Titan을 사용하는 챗봇을 보여줍니다.

### Text to Image

- [Image Generation with Stable Diffusion](./05_Image/Bedrock%20Stable%20Diffusion%20XL.ipynb): 이 노트북은 안정적인 확산 모델을 사용하여 이미지를 생성하는 방법을 보여줍니다.
