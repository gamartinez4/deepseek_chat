import tempfile

from src.retrieval.vector_store import build_index, retrieve


def test_retrieve_returns_relevant_doc():
    docs = [
        "El producto A es de color rojo y tamaño mediano.",
        "El producto B es azul y grande.",
        "El producto C es verde y pequeño.",
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        build_index(docs, index_path=tmp_dir)
        results = retrieve("¿Qué producto es azul?", index_path=tmp_dir, top_k=2)
        assert any("azul" in doc.lower() for doc in results) 