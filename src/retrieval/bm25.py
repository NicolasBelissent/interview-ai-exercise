import math
from collections import Counter
from typing import List

class BM25:
    def __init__(self, documents: List[str], k1=1.5, b=0.85):
        """
        Initialize BM25 with a list of documents.
        :param documents: List of documents (as strings)
        :param k1: BM25 parameter (controls term frequency saturation) How important is term frequency
        :param b: BM25 parameter (controls length normalization) How important is document length? are docs generally the same size?
        """
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_lengths = [len(doc.split()) for doc in documents]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)
        self.term_frequencies = [Counter(doc.split()) for doc in documents]
        self.doc_frequencies = self._compute_doc_frequencies()
        self.num_documents = len(documents)

    def _compute_doc_frequencies(self):
        """
        Compute document frequency for each term across all documents.
        """
        doc_frequencies = Counter()
        for tf in self.term_frequencies:
            doc_frequencies.update(tf.keys())
        return doc_frequencies

    def _idf(self, term):
        """
        Compute the inverse document frequency (IDF) for a term.
        """
        df = self.doc_frequencies.get(term, 0)
        if df == 0:
            return 0
        return math.log((self.num_documents - df + 0.5) / (df + 0.5) + 1)

    def _bm25_score(self, query, doc_index):
        """
        Compute BM25 score for a query against a specific document.
        """
        score = 0.0
        doc_length = self.doc_lengths[doc_index]
        tf = self.term_frequencies[doc_index]
        for term in query.split():
            idf = self._idf(term)
            term_freq = tf.get(term, 0)
            numerator = term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
            score += idf * numerator / denominator
        return score

    def query(self, query, k=5):
        """
        Retrieve top-k documents for a query based on BM25 scores.
        """
        scores = [
            (index, self._bm25_score(query, index))
            for index in range(self.num_documents)
        ]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return scores[:k]
