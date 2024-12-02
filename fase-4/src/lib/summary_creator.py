from transformers import pipeline


# Função para gerar um resumo do vídeo usando o modelo de resumo do Hugging Face
def generate_summary(activities, emotions):
    activities_summary = set(activities)
    emotions_summary = set(emotions)

    # Concatene as atividades e emoções para formar um texto que será resumido
    text_to_summarize = f"Activities: {', '.join(activities_summary)}.\nEmotions: {', '.join(emotions_summary)}."

    # Usando o pipeline de resumo do Hugging Face
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(
        text_to_summarize, max_length=150, min_length=25, do_sample=False
    )

    return summary[0]["summary_text"]
