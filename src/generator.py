from typing import List

from groq import Groq


# Default model constant
DEFAULT_MODEL = "llama-3.1-8b-instant"


def configure_groq(api_key: str):
    
    client = Groq(api_key=api_key)
    return client


def create_groq_model(model_name: str = DEFAULT_MODEL):
   
    return model_name


def build_prompt(question: str, context_chunks: List[str]) -> str:

    context = "\n\n".join(context_chunks)

    prompt = f"""Use the following context from the user's documents to answer their question.
You may reason about, explain, and expand on the information found in the context.
If the question is related to topics covered in the context, use your knowledge to give a thorough and helpful answer.
Only say you don't know if the question is completely unrelated to anything in the context.
You may help the user understand summarize or expand the details of the document.
You are not allowed to make up to fill the gaps.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    return prompt


def generate_answer(
    client: Groq,
    question: str,
    context_chunks: List[str],
    model_name: str = DEFAULT_MODEL,
    temperature: float = 0.5,
    max_tokens: int = 2048
) -> str:

    prompt = build_prompt(question, context_chunks)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are a knowledgeable assistant. You answer questions "
                           "using the provided document context as your primary source, "
                           "but you can also reason, infer, and draw on related knowledge "
                           "to give thorough, helpful answers."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.9
    )

    return response.choices[0].message.content or ""


def generate_with_fallback(
    client: Groq,
    question: str,
    context_chunks: List[str],
    model_name: str = DEFAULT_MODEL,
    fallback_message: str = "I don't have enough information to answer that."
) -> str:

    if not context_chunks:
        return fallback_message

    return generate_answer(client, question, context_chunks, model_name)