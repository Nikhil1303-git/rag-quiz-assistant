"""RAG pipeline orchestrating the complete workflow."""

from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Complete Retrieval Augmented Generation pipeline."""

    def __init__(
        self,
        pdf_extractor,
        text_chunker,
        vector_store,
        llm_client,
    ):
        """Initialize RAG pipeline.

        Args:
            pdf_extractor: PDFTextExtractor instance
            text_chunker: TextChunker instance
            vector_store: ChromaVectorStore instance
            llm_client: LLMClient instance
        """
        self.pdf_extractor = pdf_extractor
        self.text_chunker = text_chunker
        self.vector_store = vector_store
        self.llm_client = llm_client
        logger.info("Initialized RAGPipeline")

    def ingest_pdf(
        self,
        pdf_path: Path,
        collection_name: Optional[str] = None,
        force_recreate: bool = False,
    ) -> Dict[str, Any]:
        """Ingest a PDF document into the vector store.

        Args:
            pdf_path: Path to PDF file
            collection_name: Optional collection name
            force_recreate: If True, recreate collection

        Returns:
            Ingestion results with metadata
        """
        pdf_path = Path(pdf_path)
        logger.info(f"Starting PDF ingestion: {pdf_path}")

        # Extract text
        logger.debug("Extracting text from PDF...")
        texts = self.pdf_extractor.extract_text(pdf_path)
        logger.info(f"Extracted {len(texts)} pages")

        # Chunk text
        logger.debug("Chunking text...")
        chunks = self.text_chunker.chunk_texts(texts)
        logger.info(f"Created {len(chunks)} chunks")

        # Create collection
        if force_recreate:
            self.vector_store.delete_collection()

        self.vector_store.create_collection()

        # Generate metadata
        metadata = [
            {
                "source": str(pdf_path),
                "source_file": pdf_path.name,
                "chunk_size": len(chunk),
            }
            for chunk in chunks
        ]

        # Add to vector store
        logger.debug("Adding chunks to vector store...")
        self.vector_store.add_documents(
            texts=chunks,
            metadata=metadata,
        )

        result = {
            "success": True,
            "pdf_path": str(pdf_path),
            "pages_extracted": len(texts),
            "chunks_created": len(chunks),
            "collection_name": self.vector_store.collection_name,
        }

        logger.info(f"Successfully ingested PDF: {result}")
        return result

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
    ) -> List[str]:
        """Retrieve relevant documents for a query.

        Args:
            query: Query text
            n_results: Number of results to retrieve

        Returns:
            List of relevant document chunks
        """
        logger.debug(f"Retrieving documents for query: {query}")

        # Ensure collection exists (load from persistence if needed)
        if self.vector_store.collection is None:
            self.vector_store.create_collection()

        results = self.vector_store.query(
            query_texts=[query],
            n_results=n_results,
        )

        documents = results["documents"][0] if results["documents"] else []
        logger.info(f"Retrieved {len(documents)} documents")

        return documents

    def generate_response(
        self,
        query: str,
        retrieved_documents: List[str],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """Generate LLM response based on retrieved documents.

        Args:
            query: Original query
            retrieved_documents: Documents retrieved from vector store
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            Generated response
        """
        logger.debug("Generating LLM response...")

        # Create context
        context = "\n\n".join(retrieved_documents)

        # Create prompt
        prompt = f"""You are a helpful expert assistant. Your users are asking questions based on provided documents.
You will be shown the user's question and relevant information from the documents.
Answer the user's question using only the provided information.
If the information is not sufficient to answer the question, say so clearly.

Question: {query}

Information from documents:
{context}

Answer:"""

        logger.debug(f"Generated prompt ({len(prompt)} chars)")

        # Generate response
        response = self.llm_client.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        logger.info("Successfully generated response")
        return response

    def rag_query(
        self,
        query: str,
        n_retrieve: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """Execute complete RAG query: retrieve + generate.

        Args:
            query: Query text
            n_retrieve: Number of documents to retrieve
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            Result dict with query, retrieved docs, and response
        """
        logger.info(f"Executing RAG query: {query}")

        # Retrieve
        documents = self.retrieve(query, n_results=n_retrieve)

        # Generate
        response = self.generate_response(
            query=query,
            retrieved_documents=documents,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        result = {
            "query": query,
            "retrieved_documents": documents,
            "response": response,
            "n_documents_retrieved": len(documents),
        }

        logger.info("RAG query completed successfully")
        return result

    def generate_quiz(
        self,
        num_questions: int = 5,
        n_context_chunks: int = 8,
        temperature: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Generate interview-style quiz questions grounded in the indexed document.

        Args:
            num_questions: How many questions to generate
            n_context_chunks: How many indexed chunks to draw the questions from
            temperature: Sampling temperature for generation

        Returns:
            List of question dicts, each with: id, question, context, expected_answer
            ("context" and "expected_answer" are the grading key — not shown to the user)
        """
        from app.utils import extract_json

        if self.vector_store.collection is None:
            self.vector_store.create_collection()

        chunks = self.vector_store.sample_documents(n=n_context_chunks)
        if not chunks:
            raise ValueError("No indexed content found. Ingest a PDF before generating a quiz.")

        context = "\n\n---\n\n".join(chunks)

        prompt = f"""You are an experienced technical interviewer preparing candidate practice questions.
Using ONLY the study material below, write {num_questions} interview-style questions that test understanding of the material.

For each question, also provide the key points a strong answer should include, based strictly on the material.

Study material:
{context}

Respond with ONLY a valid JSON array, no other text, in exactly this format:
[
  {{
    "question": "the interview question",
    "expected_answer": "2-4 key points a good answer should cover, based on the material"
  }}
]"""

        raw = self.llm_client.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=1500,
        )

        parsed = extract_json(raw)
        if not isinstance(parsed, list):
            raise ValueError("Quiz generation did not return a list of questions")

        questions = []
        for i, item in enumerate(parsed[:num_questions]):
            questions.append(
                {
                    "id": f"q{i + 1}",
                    "question": item.get("question", "").strip(),
                    "expected_answer": item.get("expected_answer", "").strip(),
                    "context": context,
                }
            )

        logger.info(f"Generated {len(questions)} quiz questions")
        return questions

    def evaluate_answer(
        self,
        question: str,
        expected_answer: str,
        context: str,
        user_answer: str,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """Grade a user's answer to a generated quiz question.

        Args:
            question: The quiz question
            expected_answer: Key points a good answer should include
            context: The source material the question was generated from
            user_answer: The candidate's typed answer
            temperature: Sampling temperature (kept low for consistent grading)

        Returns:
            Dict with: verdict, score (0-100), feedback
        """
        from app.utils import extract_json

        prompt = f"""You are grading a candidate's interview answer.

Question: {question}

Reference material (ground truth): {context}

Key points a strong answer should cover: {expected_answer}

Candidate's answer: {user_answer}

Grade the answer strictly based on the reference material and key points above.
Respond with ONLY valid JSON, no other text, in exactly this format:
{{
  "verdict": "Correct" or "Partially Correct" or "Incorrect",
  "score": <integer 0-100>,
  "feedback": "2-3 sentences: what was right, what was missing or wrong"
}}"""

        raw = self.llm_client.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=500,
        )

        parsed = extract_json(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Answer evaluation did not return a JSON object")

        return {
            "verdict": parsed.get("verdict", "Unknown"),
            "score": parsed.get("score", 0),
            "feedback": parsed.get("feedback", ""),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get pipeline status and configuration.

        Returns:
            Status information
        """
        try:
            collection_info = self.vector_store.get_collection_info()
        except Exception as e:
            collection_info = {"error": str(e)}

        model_info = self.llm_client.get_model_info()

        return {
            "llm_provider": model_info.get("provider"),
            "llm_model": model_info.get("model"),
            "vector_store": "chromadb",
            "collection_info": collection_info,
            "pdf_extractor": self.pdf_extractor.method,
            "chunk_size": self.text_chunker.chunk_size,
            "chunk_overlap": self.text_chunker.chunk_overlap,
        }
