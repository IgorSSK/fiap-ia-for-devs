from transformers import pipeline


def create_summary(text: str, output_path):
    summarizer = pipeline("summarization")
    summary = summarizer(
        text,
        max_length=200,
        min_length=50,
        do_sample=False,
    )
    print(summary)
    text = summary[0]["summary_text"]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
