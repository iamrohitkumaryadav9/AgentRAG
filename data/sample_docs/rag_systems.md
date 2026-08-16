# Retrieval-Augmented Generation

Retrieval-Augmented Generation, or RAG, is a technique that combines a
retrieval system with a language model. Instead of relying purely on facts
memorized during training, a RAG system first searches a knowledge base for
passages relevant to the user's question, then asks a language model to
compose an answer using those retrieved passages as grounding context.

A typical RAG pipeline has four stages. First, documents are ingested and
split into chunks, since embedding an entire document as one vector loses
too much specificity. Second, each chunk is embedded into a numeric vector
using an embedding model, and stored in a vector store or vector database
such as FAISS, Chroma, or Pinecone. Third, when a query arrives it is
embedded with the same model and used to search the vector store for the
most similar chunks, typically using cosine similarity or dot product.
Fourth, the retrieved chunks are inserted into a prompt template and passed
to a language model, which generates the final answer.

RAG is popular because it reduces hallucination, lets a model answer
questions about private or fast-changing data without retraining, and makes
answers auditable, since each answer can be traced back to source passages.
The main risks are retrieval failure (the right passage is never found),
poor chunking (context is cut in the wrong place), and generation that
drifts away from the retrieved context despite it being provided -- which is
why production RAG systems add a groundedness check, sometimes called a
guardrail, that verifies the generated answer is actually supported by the
retrieved passages before returning it to the user.

Chunk size and overlap are important tuning parameters. Chunks that are too
large dilute the embedding with irrelevant content, while chunks that are
too small lose surrounding context. Overlap between consecutive chunks
helps avoid splitting a single idea across a hard boundary.
