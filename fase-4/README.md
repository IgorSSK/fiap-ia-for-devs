# TECH CHALLENGE - FASE 04 - Análise de Vídeo: Reconhecimento Facial, Expressões e Atividades

Este programa realiza **análise de vídeos** com foco em quatro tarefas principais: **reconhecimento facial**, **análise de expressões emocionais**, **detecção de atividades humanas**, e **geração de resumo automático** das principais atividades e emoções.

## Funcionalidades

1. **Reconhecimento Facial**:
   - Detecta e marca rostos no vídeo utilizando o **MediaPipe**.
2. **Análise de Expressões Emocionais**:
   - Identifica emoções (feliz, triste, neutro, etc.) nos rostos usando o modelo **FER**.
3. **Detecção de Atividades**:
   - Categoriza atividades humanas como andar, sentar, levantar os braços, etc.
4. **Geração de Resumo**:
   - Utiliza a biblioteca **Transformers** para criar um resumo textual das atividades e emoções detectadas.

---

## Como Rodar
**Execute o programa**
```bash
    cd fase-4
    pip install -r requirements.txt
    python src/main.py
```

---

## O que o Programa Faz?

1. **Processamento de Vídeo**:
   - Lê o vídeo quadro a quadro.
   - Detecta rostos e pose corporal.
2. **Análise de Emoções**:
   - Usa um modelo pré-treinado para identificar emoções em cada rosto.
3. **Classificação de Atividades**:
   - Analisa movimentos corporais e classifica atividades humanas.
4. **Resumo Automático**:
   - Gera uma descrição compacta das emoções e atividades observadas no vídeo.
