"""
사전학습된 opus-mt-en-de 모델의 인코더 self-attention을 히트맵으로 시각화.
논문에서 말하는 "각 토큰이 다른 토큰을 얼마나 주목하는가"를 눈으로 확인하는 스크립트.

실행:
    python visualize_attention.py
결과:
    attention_heatmap.png 파일 생성
"""
import matplotlib.pyplot as plt
import torch
from transformers import MarianMTModel, MarianTokenizer

MODEL_NAME = "Helsinki-NLP/opus-mt-en-de"
SENTENCE = "The animal did not cross the street because it was tired."


def main():
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME, attn_implementation="eager")
    model.eval()

    inputs = tokenizer([SENTENCE], return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        enc_out = model.get_encoder()(**inputs, output_attentions=True)

    last_layer_attn = enc_out.attentions[-1][0]  # (num_heads, seq, seq)
    num_heads = last_layer_attn.shape[0]

    fig, axes = plt.subplots(2, num_heads // 2, figsize=(4 * num_heads // 2, 8))
    for h, ax in enumerate(axes.flat):
        ax.imshow(last_layer_attn[h].numpy(), cmap="viridis")
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=7)
        ax.set_yticklabels(tokens, fontsize=7)
        ax.set_title(f"Head {h}", fontsize=9)

    fig.suptitle(f"Encoder self-attention (last layer)\n\"{SENTENCE}\"")
    fig.tight_layout()
    fig.savefig("attention_heatmap.png", dpi=150)
    print("저장 완료: attention_heatmap.png")


if __name__ == "__main__":
    main()
