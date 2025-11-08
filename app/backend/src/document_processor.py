import os
import logging
import traceback
from typing import List, Any
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredHTMLLoader,
    TextLoader
)
from unstructured.partition.auto import partition

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger("recipe_agent.document_processor")

# Add file handler
file_handler = logging.FileHandler('document_processor.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
))
logger.addHandler(file_handler)


class DocumentProcessor:
    """Simplified document processor for extracting text from various document formats"""

    def __init__(self):
        logger.info("Initializing DocumentProcessor")
        logger.info("DocumentProcessor initialized successfully")

    def _load_document(self, file_path: str) -> List[Any]:
        """
        Load document based on file extension using langchain loaders
        Note: For images, this method is not ideal. Use extract_text_from_document_unstructured() instead.
        """
        try:
            ext = file_path.split(".")[-1].lower()
            logger.info(f"Loading document with extension: {ext}")
            logger.debug(f"File path: {file_path}")

            # For image files, raise an exception to force using unstructured
            if ext in ["jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp", "svg"]:
                logger.warning(
                    "Image file detected - langchain loaders don't support OCR")
                raise Exception(
                    "Image files require OCR processing. Please use extract_text_from_document_unstructured() method.")

            if ext == "pdf":
                logger.debug("Using PyPDFLoader")
                try:
                    loader = PyPDFLoader(file_path)
                except Exception as pdf_error:
                    if "File has not been decrypted" in str(pdf_error) or "FileNotDecryptedError" in str(pdf_error):
                        logger.error(
                            "PDF is password-protected and cannot be processed")
                        raise Exception(
                            "The PDF is password-protected and cannot be processed. Please use an unprotected PDF.")
                    else:
                        raise pdf_error
            elif ext in ["doc", "docx"]:
                logger.debug("Using UnstructuredWordDocumentLoader")
                loader = UnstructuredWordDocumentLoader(file_path)
            elif ext in ["html", "htm"]:
                logger.debug("Using UnstructuredHTMLLoader")
                loader = UnstructuredHTMLLoader(file_path)
            elif ext in ["ppt", "pptx"]:
                logger.debug("Using TextLoader for PowerPoint (fallback)")
                loader = TextLoader(file_path)
            else:
                logger.debug("Using TextLoader")
                loader = TextLoader(file_path)

            document = loader.load()
            logger.info(
                f"Document loaded successfully: {len(document)} pages/sections")
            if document:
                logger.debug(
                    f"First page preview: {document[0].page_content[:200]}...")
            return document
        except Exception as e:
            logger.error(f"Error loading document: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def extract_text(self, file_path: str) -> str:
        """
        Extract text content from a document using langchain loaders

        Args:
            file_path: Path to the document file

        Returns:
            Extracted text content as string
        """
        try:
            logger.info(f"Extracting text from document: {file_path}")
            document = self._load_document(file_path)

            # Combine all pages/sections into a single text
            content = "\n\n".join(
                page.page_content for page in document
            )

            logger.info(f"Successfully extracted {len(content)} characters")
            logger.debug(f"First 500 chars: {content[:500]}...")
            return content

        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def extract_text_from_document_unstructured(self, file_path: str) -> str:
        """
        Extract text from various document formats using the unstructured library

        Supports:
        - Documents: PDF, Word (.doc, .docx), PowerPoint (.ppt, .pptx), HTML, RTF, ODT
        - Images: JPG, JPEG, PNG, GIF, BMP, TIFF, WEBP (uses OCR)
        - Screenshots: Any supported image format (automatically applies OCR)
        - And more...

        Note: For images and screenshots, the unstructured library uses OCR (Optical Character Recognition)
        to extract text. Make sure tesseract-ocr is installed on your system for best results.

        Args:
            file_path: Path to the document file

        Returns:
            Extracted text content as string
        """
        try:
            logger.info(
                f"Extracting text using Unstructured from: {file_path}")

            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            # Use unstructured's auto partition to handle any document type
            elements = partition(filename=file_path)

            # Extract text from all elements
            extracted_text = "\n\n".join(
                [str(element) for element in elements])

            logger.info(
                f"Successfully extracted {len(extracted_text)} characters using Unstructured")
            logger.debug(f"First 500 chars: {extracted_text[:500]}...")

            return extracted_text

        except Exception as e:
            logger.error(f"Error extracting text with Unstructured: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Fallback to original method
            logger.info("Falling back to original extraction method")
            return await self.extract_text(file_path)

    def extract_text_sync(self, file_path: str) -> str:
        """
        Synchronous version of extract_text_from_document_unstructured
        Useful for non-async contexts

        Supports:
        - Documents: PDF, Word (.doc, .docx), PowerPoint (.ppt, .pptx), HTML, RTF, ODT
        - Images: JPG, JPEG, PNG, GIF, BMP, TIFF, WEBP (uses OCR)
        - Screenshots: Any supported image format (automatically applies OCR)
        - And more...

        Args:
            file_path: Path to the document file

        Returns:
            Extracted text content as string
        """
        try:
            logger.info(
                f"Extracting text (sync) using Unstructured from: {file_path}")

            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            # Use unstructured's auto partition to handle any document type
            elements = partition(filename=file_path)

            # Extract text from all elements
            extracted_text = "\n\n".join(
                [str(element) for element in elements])

            logger.info(
                f"Successfully extracted {len(extracted_text)} characters using Unstructured")
            logger.debug(f"First 500 chars: {extracted_text[:500]}...")

            return extracted_text

        except Exception as e:
            logger.error(f"Error extracting text with Unstructured: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
