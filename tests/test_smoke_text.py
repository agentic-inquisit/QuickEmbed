"""Reproduces the manual smoke test run during setup: loads the real local
weights and checks the text embedding path returns sane similarities.
"""

from pathlib import Path

from sentence_transformers import SentenceTransformer

MODEL_DIR = Path(__file__).parent.parent / "models" / "embeddinggemma-2"

print(MODEL_DIR)

def test_text_similarity_ranks_relevant_doc_highest():
    model = SentenceTransformer(str(MODEL_DIR), device="cpu")
    assert model.get_embedding_dimension() == 768

    query = model.encode("What causes the northern lights?", prompt_name="SearchQuery")
    docs = model.encode(
        [
            "Charged particles from the sun interact with gases in Earth's atmosphere, causing the aurora borealis.",
            "The recipe calls for two cups of flour and a teaspoon of baking soda.",
            "Photosynthesis converts sunlight into chemical energy stored in glucose.",
        ],
        prompt_name="Document",
    )

    sims = model.similarity(query, docs)[0]
    assert sims[0] > sims[1]
    assert sims[0] > sims[2]
    assert sims[0] > 0.7

    print(len(sims))

def main():
    test_text_similarity_ranks_relevant_doc_highest()
    print("This is the main loop or logic of the program.")

if __name__ == "__main__":
    # This block only executes if you run this script directly
    main()