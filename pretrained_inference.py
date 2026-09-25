"""
사전학습된 Transformer 기반 번역 모델(Marian / opus-mt)로 실제 추론해보기.
원 논문(Attention Is All You Need)이 다룬 것과 같은 encoder-decoder
번역 구조를 그대로 쓰는 작은 모델이라 CPU 노트북에서도 가볍게 돈다.

최초 실행 시 모델 가중치(약 300MB)를 HuggingFace Hub에서 한 번 내려받는다.

실행:
    python pretrained_inference.py
"""
import torch
from transformers import MarianMTModel, MarianTokenizer

MODEL_NAME = "Helsinki-NLP/opus-mt-en-de"  # 영어 -> 독일어, 작고 빠른 모델


def main():
    print(f"모델 로드 중: {MODEL_NAME} (처음 실행이면 다운로드 발생)")
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME, attn_implementation="eager")
    model.eval()

    print(f"인코더 레이어 수: {model.config.encoder_layers}, "
          f"디코더 레이어 수: {model.config.decoder_layers}, "
          f"d_model: {model.config.d_model}, "
          f"attention heads: {model.config.encoder_attention_heads}")

    sentences = [
        "Attention is all you need.",
        "The transformer architecture relies entirely on self-attention.",
        "I study machine learning on my laptop.",
    ]

    with torch.no_grad():
        for sent in sentences:
            batch = tokenizer([sent], return_tensors="pt", padding=True)
            generated = model.generate(**batch, max_new_tokens=50)
            translation = tokenizer.decode(generated[0], skip_special_tokens=True)
            print(f"EN: {sent}")
            print(f"DE: {translation}")
            print("-" * 60)

    # 인코더 self-attention 가중치 한번 뽑아보기 (논문 핵심 개념 확인용)
    print("\n== Self-Attention 가중치 shape 확인 ==")
    sample = tokenizer(["Attention is all you need."], return_tensors="pt")
    with torch.no_grad():
        enc_out = model.get_encoder()(
            **sample, output_attentions=True
        )
    attn = enc_out.attentions[-1]  # 마지막 인코더 레이어의 attention
    print(f"마지막 인코더 레이어 attention shape: {attn.shape}")
    print("(batch, num_heads, seq_len, seq_len) - 각 헤드가 토큰 간 관계를 어떻게 보는지")


if __name__ == "__main__":
    main()
