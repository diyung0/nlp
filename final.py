import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import argparse
from PyPDF2 import PdfReader
from langchain.schema import Document
from langchain_text_splitters  import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from bert_score import score

import pandas as pd
import time
import json

class QAGenerator(nn.Module):
    def __init__(self, model_name='gpt2-large'):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
    
    def get_device(self):
        return next(self.model.parameters()).device
    
    @torch.no_grad()
    def generate_answer(self, context, question, max_length=100, temperature=0.3, top_p=0.9):
        """주어진 컨텍스트와 질문으로 답변 생성 (Few-shot 프롬프트 포함)"""
        
        # 프롬프트 구성
        prompt = f"""Please answer the question below using only the information provided in the context.
        If the context does not provide enough information, say "The document does not provide enough information."
        
        Context:
        {context}
        Question:
        {question}
        Answer:"""

        # 토크나이징
        inputs = self.tokenizer.encode(prompt, return_tensors='pt', max_length=900, truncation=True)
        inputs = inputs.to(self.get_device())
        
        # 생성
        for _ in range(max_length):
            # 현재 시퀀스로 다음 토큰 예측
            logits = self.forward(inputs)
            next_token_logits = logits[0, -1, :] / temperature
            
            # Top-p 샘플링
            probs = F.softmax(next_token_logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Top-p 마스크 생성
            mask = cumsum_probs <= top_p
            mask[1:] = mask[:-1].clone()
            mask[0] = True
            
            # 필터링된 확률로 샘플링
            filtered_probs = sorted_probs * mask.float()
            filtered_probs = filtered_probs / filtered_probs.sum()
            
            # 다음 토큰 샘플링
            next_token_idx = torch.multinomial(filtered_probs, 1)
            next_token = sorted_indices[next_token_idx]
            
            # 종료 조건들
            token_text = self.tokenizer.decode(next_token.item())
            if (next_token.item() == self.tokenizer.eos_token_id or 
                token_text in ['\n\nContext:', '\nContext:', 'Question:'] or
                '\n\n' in token_text):
                break
                
            # 새 토큰을 시퀀스에 추가
            inputs = torch.cat([inputs, next_token.unsqueeze(0)], dim=-1)
        
        # 디코딩하여 텍스트로 변환
        generated_text = self.tokenizer.decode(inputs[0], skip_special_tokens=True)

        # print(f"\n\n\nGenerated text: {generated_text}\n\n\n")

        # 답변 부분만 추출 (마지막 Answer: 이후 부분)
        if "Answer:" in generated_text:
            answer_parts = generated_text.split("Answer:")
            answer = answer_parts[-1].strip()
        else:
            answer = "Could not generate an answer."

        return answer


def pdf_to_text(pdf_path):
    """Convert PDF to text."""
    reader = PdfReader(pdf_path)
    pages = [page.extract_text() for page in reader.pages]
    return "\n".join(pages)


def chunk_text(text, chunk_method, embeddings_model=None, chunk_size=128):
    """Chunk text using the specified method."""
    if chunk_method == 'character':
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False
        )
    elif chunk_method == 'recursive':
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False
        )
    elif chunk_method == 'semantic':
        if embeddings_model:
            text_splitter = SemanticChunker(
                embeddings_model,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=25  # 상위 25%에서 분할
            )
        else:
            raise ValueError("Embeddings model must be provided for semantic chunking")
    else:
        raise ValueError("Unsupported chunk method")

    documents = [Document(page_content=text, metadata={"page_number": 1})]
    return text_splitter.split_documents(documents)


def save_to_file(data, file_path):
    """Save data to a file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        if isinstance(data, (dict, list)):
            json.dump(data, f, ensure_ascii=False, indent=4)
        else:
            f.write(data)


def main(args):
    """간단한 사용 예제"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 모델 초기화
    qa_generator = QAGenerator()
    qa_generator.to(device)
    qa_generator.eval()

    # PDF 파일에서 텍스트 추출
    # print(f"Reading PDF file: {args.pdf_path}")
    # text = pdf_to_text(args.pdf_path)
    # print("PDF text extraction complete.")
    # print(f"Extracted text length: {len(text)} characters")

    # # 텍스트 저장
    # save_to_file(text, "extracted_text.txt")
    # print("📄 Extracted text saved to 'extracted_text.txt'")

    # 텍스트 읽기
    with open("extracted_text.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # 청킹 전략을 인자로 받아서 텍스트를 청크로 나누기
    print(f"Chunking text using method: {args.chunk_method} and chunk size: {args.chunk_size}")
    documents = chunk_text(
        text,
        chunk_method=args.chunk_method,
        embeddings_model=OllamaEmbeddings(model="bge-m3") if args.chunk_method == 'semantic' else None,
        chunk_size=args.chunk_size
    )
    print(f"Number of chunks created: {len(documents)}")

    # 청크 저장
    chunks_data = [doc.page_content for doc in documents]
    save_to_file(chunks_data, "chunks.json")
    print("📄 Chunks saved to 'chunks.json'")

    # (4) 청크를 임베딩 모델을 통해 vector화
    embeddings_model = OllamaEmbeddings(model="bge-m3")

    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings_model
    )
    print("Vector store created with FAISS.")

    # 벡터스토어 저장
    vector_store.save_local("vector_store")
    print("📄 Vector store saved to 'vector_store' directory")

    # (5) 질문 데이터셋을 읽어와서 vector화된 청크와 질문을 비교하여 가장 관련성 높은 청크를 찾기
    # (6) 관련 청크(context)와 질문, few-shot 예제를 gpt2모델에 입력하여 답변 생성

    questions = [
        {
            # Q1 ⭐️
            "question": "By how much can the duration of study be reduced through the Integrated Program?"
            # article 2
        },
        {
            # Q2 ⭐️ ⭐️
            "question": "What is the intended benefit of the Undergraduate-Graduate Integrated Program regarding academic timeline?"
            # article 2
        },
        {
            # Q3
            "question": "What is the purpose of the Integrated Program in terms of study duration?"
            # article 2
        },
        {
            # Q4 ⭐️ ❌
            "question": "What documents are required to apply for the Integrated Program?"
            # article 4
        },
        {
            # Q5 ⭐️
            "question": "How are applicants selected for the Integrated Program?"
            # article 5
        },
        {
            # Q6
            "question": "When are students eligible to apply for the Integrated Program based on their semester and earned credits?"
            # article 3.1
        },
        {
            # Q7
            "question": "What is the minimum GPA requirement to be eligible for the Integrated Program?"
            # article 3.2
        },
        {
            # Q8 
            "question": "Whose recommendations are required for applying to the Integrated Program?"
            # article 3.3 & 3.4
        },
        {
            # Q9
            "question": "Can graduate courses taken in the Integrated Program count toward undergraduate graduation requirements?"
            # article 7.2
        },
        {
            # Q10
            "question": "How many graduate credits can students take each semester before completing their undergraduate program?"
            # article 7.1
        }
    ]

    candidates = [] #BERT용
    results_data = []
    start_time = time.time()

    for i, question in enumerate(questions, 1):
        q_start = time.time()  # 질문 시작 시간

        # 상위 3개의 관련 청크를 검색
        # results = vector_store.similarity_search(query=question["question"], k=5)

        # 관련성 점수도 함께 확인
        results_with_scores = vector_store.similarity_search_with_score(query=question["question"], k=7)
        print(f"Retrieved {len(results_with_scores)} results for question {i}: {question['question']}")

        # 점수가 너무 낮은 것들 필터링
        filtered_results = [doc for doc, score in results_with_scores if score < 0.7]  # 임계값 조정

        combined_context = "\n".join([result.page_content for result in filtered_results[:3]])

        print(f"\n--- Example {i} ---")
        print(f"Question: {question['question']}")
        print(f"Context: {combined_context[:200]}...")  # context 요약

        # 답변 생성
        answer = qa_generator.generate_answer(combined_context, question['question'])

        q_time = time.time() - q_start  # 질문별 소요 시간

        results_data.append({
            "question": question["question"],
            "context": combined_context,
            "answer": answer,
            "elapsed_time_sec": round(q_time, 3)
        })
        candidates.append(answer)

        print(f"Answer: {answer}")
        print(f"⏱ Time for this question: {q_time:.2f} seconds")
        print("-" * 80)
    

    # 실제 답변과 예측된 답변 리스트
    references = [
        # Q1
        "The Integrated Program allows for the reduction of the duration of study by up to one year each for the bachelor's, master's, and integrated master's-doctoral degrees.",

        # Q2
        "The primary benefit of the Undergraduate-Graduate Integrated Program is that it enables students to complete their academic degrees in a shorter time by reducing the total duration of study by up to one year at each level.",

        # Q3
        "The purpose of the Integrated Program is to shorten the duration of study for bachelor's, master's, and integrated degrees by up to one year each.",

        # Q4
        "Applicants must submit an application form for the Integrated Program, a copy of their undergraduate academic transcript, a research plan, and a recommendation letter from the graduate advisor.",

        # Q5    
        "Applicants are selected through document screening, which considers undergraduate grades, research plans, recommendation letters, and other criteria set by the department.",

        # Q6
        "Students are eligible to apply for the Integrated Program after completing 4 to 7 semesters, provided they have earned at least 64, 80, 95, or 110 credits respectively, excluding seasonal session credits in the application semester.",

        # Q7
        "A cumulative GPA of 3.3 or higher is required up to the semester in which the student applies for the Integrated Program.",

        # Q8
        "Applicants need recommendations from both the head of their undergraduate department and the advisor of the intended graduate department. If applying to a different or interdisciplinary graduate department, an additional recommendation is required from the graduate department head.",

        # Q9
        "Graduate courses, except for shared undergraduate-graduate courses, are not recognized as undergraduate graduation credits. However, up to 12 credits of graduate courses can be completed before graduation, and some may be recognized as graduate credits later with approval.",

        # Q10
        "Students may take up to 6 credits of graduate courses per semester before undergraduate graduation, including shared undergraduate-graduate courses."
    ]

    # BERTScore 계산
    P, R, F1 = score(candidates, references, lang="en", verbose=True)

    # 각 점수 출력
    print(f"Precision: {P.mean():.4f}")
    print(f"Recall: {R.mean():.4f}")
    print(f"F1: {F1.mean():.4f}")
    total_time = time.time() - start_time
    print(f"\n⏱ Total Time Elapsed: {total_time:.2f} seconds")

    # DataFrame 저장
    df = pd.DataFrame(results_data)
    df.to_csv("qa_results.csv", index=False)
    print("📄 Results saved to 'qa_results.csv'")




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_path", type=str, default="data/test_real.pdf")
    parser.add_argument("--chunk_method", type=str, choices=['character', 'recursive', 'semantic'], default='recursive')
    parser.add_argument("--chunk_size", type=int, choices=[128, 256, 512, 1024], default=128) 
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)




