"""
llm_pipeline.py
----------------
Implements the Dual Groq Model Architecture:
  Model 1 (Mixtral 8x7B)       -> Context Summarizer
  Model 2 (Llama 3.1 8B Instant) -> Final Answer Generator
"""

from groq import Groq


SUMMARIZER_MODEL = "openai/gpt-oss-20b"
ANSWER_MODEL = "openai/gpt-oss-120b"

class LLMPipelineError(Exception):
    """Custom exception for Groq API / LLM pipeline failures."""
    pass


def get_groq_client(api_key: str) -> Groq:
    if not api_key or not api_key.strip():
        raise LLMPipelineError("Groq API key is missing. Please enter it in the sidebar.")
    try:
        return Groq(api_key=api_key)
    except Exception as e:
        raise LLMPipelineError(f"Failed to initialize Groq client: {str(e)}")


def summarize_context(
    client: Groq,
    retrieved_chunks: list[dict],
    conversation_history_summary: str,
    current_question: str
) -> str:
    """
    MODEL 1 - Mixtral 8x7B - Context Summarizer
    """
    if not retrieved_chunks:
        raise LLMPipelineError("No retrieved chunks provided to summarizer.")

    chunks_text = "\n\n".join(
        f"[Source: {c['source']}, Page: {c['page']}]\n{c['chunk_text']}"
        for c in retrieved_chunks
    )

    system_prompt = (
        "You are a context summarization assistant. Your job is to condense retrieved "
        "document excerpts and prior conversation history into a compact, factual summary "
        "that preserves all key facts, figures, and source attributions needed to answer "
        "the user's current question. Eliminate redundant or irrelevant information. "
        "Do not answer the question yourself — only summarize the relevant context."
    )

    user_prompt = (
        f"Current question: {current_question}\n\n"
        f"Previous conversation summary:\n{conversation_history_summary or 'None'}\n\n"
        f"Retrieved document excerpts:\n{chunks_text}\n\n"
        "Provide a condensed context summary (key facts, figures, and source references) "
        "that will help answer the current question."
    )

    try:
        response = client.chat.completions.create(
            model=SUMMARIZER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=800
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        raise LLMPipelineError(f"Context summarization failed (Model 1 - Mixtral): {str(e)}")


def generate_answer(
    client: Groq,
    condensed_context: str,
    current_question: str
) -> str:
    """
    MODEL 2 - Llama 3.1 8B Instant - Final Answer Generator
    """
    if not condensed_context:
        raise LLMPipelineError("No condensed context provided to answer generator.")

    system_prompt = (
        "You are a research assistant that answers user questions using ONLY the provided "
        "condensed context. Always cite the source document and page number when referencing "
        "specific facts (e.g. 'according to report.pdf, page 3'). If the context does not "
        "contain enough information to answer confidently, say so clearly instead of guessing. "
        "Structure your answer clearly and explain your reasoning briefly."
    )

    user_prompt = (
        f"Condensed context:\n{condensed_context}\n\n"
        f"Question: {current_question}\n\n"
        "Provide a detailed, accurate answer with source references."
    )

    try:
        response = client.chat.completions.create(
            model=ANSWER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        raise LLMPipelineError(f"Answer generation failed (Model 2 - Llama 3.1): {str(e)}")


def run_dual_llm_pipeline(
    api_key: str,
    retrieved_chunks: list[dict],
    conversation_history_summary: str,
    current_question: str
) -> dict:
    """
    Orchestrates the full dual-LLM pipeline:
      1. Mixtral summarizes retrieved chunks + history
      2. Llama generates the final answer from that summary
    """
    client = get_groq_client(api_key)

    condensed_context = summarize_context(
        client, retrieved_chunks, conversation_history_summary, current_question
    )

    final_answer = generate_answer(client, condensed_context, current_question)

    return {
        "condensed_context": condensed_context,
        "final_answer": final_answer
    }