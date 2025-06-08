colab에서 실행하는 방법

(0)
먼저 런타임 -> 런타임 유형을 gpu로

(1)
!git clone https://github.com/diyung0/nlp.git

(2)
import os
import subprocess

%cd nlp
print("현재 디렉토리:", os.getcwd())

(3)
!pip install torch torchvision torchaudio transformers PyPDF2 langchain langchain-text-splitters langchain-experimental faiss-cpu langchain-community bert-score pandas langchain-ollama

(4)
import subprocess
import time
import requests

# Ollama 설치
!curl -fsSL https://ollama.com/install.sh | sh

# Ollama 서버 백그라운드 시작
def start_ollama():
    process = subprocess.Popen(['ollama', 'serve'], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE)
    print("🚀 Ollama 서버를 시작하는 중...")
    time.sleep(15)  # 충분한 시간 대기
    
    # 서버 상태 확인
    try:
        response = requests.get('http://localhost:11434/api/tags', timeout=10)
        print("✅ Ollama 서버가 성공적으로 시작되었습니다!")
        return True
    except Exception as e:
        print(f"⚠️ 서버 확인 중: {e}")
        return False

start_ollama()


(5)
!ollama pull bge-m3

(6)
# 데이터 디렉토리와 PDF 파일 확인
print("📁 데이터 디렉토리 확인:")
!ls -la data/

# PDF 파일이 없다면 업로드
if not os.path.exists('data/test2.pdf'):
    print("⚠️ PDF 파일이 없습니다. 파일을 업로드해주세요.")
    os.makedirs('data', exist_ok=True)
else:
    print("✅ PDF 파일이 확인되었습니다.")


(7)
# 환경이 제대로 설정되었는지 빠른 테스트
def quick_test():
    try:
        import torch
        import transformers
        import langchain
        from langchain_ollama import OllamaEmbeddings
        
        print("✅ PyTorch:", torch.__version__)
        print("✅ Transformers:", transformers.__version__)
        print("✅ LangChain 임포트 성공")
        print("✅ Ollama 임베딩 임포트 성공")
        
        # GPU 확인
        if torch.cuda.is_available():
            print(f"✅ GPU 사용 가능: {torch.cuda.get_device_name()}")
        else:
            print("⚠️ CPU 모드로 실행됩니다")
            
        return True
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return False

quick_test()


(8)
!python final.py --pdf_path data/test_real.pdf --chunk_method recursive

