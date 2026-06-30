import unittest
from unittest.mock import MagicMock, patch

from src.chunker import create_chunks, get_chunk_statistics
from src.pdf_processor import validate_pdf
from src.vector_store import store_embeddings
import src.config as config


class TestConfig(unittest.TestCase):
    def test_config_variables_exist(self):
        self.assertEqual(config.EMBEDDING_MODEL, "all-MiniLM-L6-v2")
        self.assertEqual(config.LLM_MODEL, "llama-3.1-8b-instant")
        self.assertEqual(config.CHUNK_SIZE, 500)
        self.assertEqual(config.CHUNK_OVERLAP, 100)
        self.assertEqual(config.TOP_K, 3)
        self.assertEqual(config.DATA_DIR, "data")
        self.assertEqual(config.VECTOR_DB_PATH, "chroma_db")


class TestChunkerStatistics(unittest.TestCase):
    def test_empty_chunks_returns_consistent_keys(self):
        stats = get_chunk_statistics([])
        expected_keys = {
            'count', 'avg_length', 'min_length', 'max_length',
            'total_chars', 'chunks_by_size'
        }
        self.assertEqual(set(stats.keys()), expected_keys)
        self.assertEqual(stats['count'], 0)
        self.assertEqual(stats['total_chars'], 0)
        self.assertEqual(stats['chunks_by_size'], {'small': 0, 'medium': 0, 'large': 0})

    def test_populated_chunks_returns_consistent_keys(self):
        chunks = ['a' * 50, 'b' * 200, 'c' * 500]
        stats = get_chunk_statistics(chunks)

        self.assertEqual(stats['count'], 3)
        self.assertEqual(stats['min_length'], 50)
        self.assertEqual(stats['max_length'], 500)
        self.assertEqual(stats['total_chars'], 750)
        self.assertEqual(stats['avg_length'], 250.0)
        self.assertEqual(stats['chunks_by_size']['small'], 1)
        self.assertEqual(stats['chunks_by_size']['medium'], 1)
        self.assertEqual(stats['chunks_by_size']['large'], 1)

    def test_empty_and_populated_share_same_keys(self):
        empty = get_chunk_statistics([])
        populated = get_chunk_statistics(['hello world'])
        self.assertEqual(set(empty.keys()), set(populated.keys()))

    def test_create_chunks_empty_input(self):
        self.assertEqual(create_chunks(""), [])
        self.assertEqual(create_chunks(None), [])

    def test_create_chunks_short_text(self):
        chunks = create_chunks("Hello world.")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "Hello world.")


class TestPDFProcessor(unittest.TestCase):
    @patch('src.pdf_processor.PdfReader')
    def test_validate_pdf_valid(self, mock_pdf_reader):
        mock_reader = MagicMock()
        mock_reader.pages = [MagicMock()]
        mock_pdf_reader.return_value = mock_reader

        result = validate_pdf(b"%PDF-1.4 mock pdf data")
        self.assertTrue(result)

    def test_validate_pdf_invalid_header(self):
        self.assertFalse(validate_pdf(b"not a pdf header"))

    @patch('src.pdf_processor.PdfReader')
    def test_validate_pdf_no_pages(self, mock_pdf_reader):
        mock_reader = MagicMock()
        mock_reader.pages = []
        mock_pdf_reader.return_value = mock_reader

        self.assertFalse(validate_pdf(b"%PDF-1.4"))


class TestVectorStore(unittest.TestCase):
    def test_store_generates_unique_ids(self):
        mock_collection = MagicMock()
        store_embeddings(
            collection=mock_collection,
            documents=["doc1", "doc2"],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            ids=None
        )

        mock_collection.add.assert_called_once()
        call_kwargs = mock_collection.add.call_args[1]
        ids = call_kwargs['ids']
        self.assertEqual(len(ids), 2)
        self.assertTrue(ids[0].startswith("chunk_"))
        self.assertNotEqual(ids[0], ids[1])

    def test_store_uses_custom_ids(self):
        mock_collection = MagicMock()
        store_embeddings(
            collection=mock_collection,
            documents=["doc1"],
            embeddings=[[0.5, 0.6]],
            ids=["my_custom_id"]
        )

        call_kwargs = mock_collection.add.call_args[1]
        self.assertEqual(call_kwargs['ids'], ["my_custom_id"])


class TestHelpers(unittest.TestCase):
    def test_format_file_size(self):
        from utils.helpers import format_file_size
        self.assertEqual(format_file_size(500), "500 B")
        self.assertEqual(format_file_size(1024), "1.0 KB")
        self.assertEqual(format_file_size(1048576), "1.0 MB")

    def test_truncate_text(self):
        from utils.helpers import truncate_text
        self.assertEqual(truncate_text("hello", 10), "hello")
        self.assertTrue(truncate_text("a" * 300, 200).endswith("..."))
        self.assertEqual(len(truncate_text("a" * 300, 200)), 203)

    def test_format_elapsed_time(self):
        from utils.helpers import format_elapsed_time
        self.assertEqual(format_elapsed_time(0.5), "500ms")
        self.assertEqual(format_elapsed_time(3.2), "3.2s")
        self.assertEqual(format_elapsed_time(65.0), "1m 5s")


if __name__ == '__main__':
    unittest.main()
