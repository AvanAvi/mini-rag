"""Unit tests for the Answerer."""

import pytest

from fake_llm_client import FakeLlmClient
from mini_rag.answer import Answer
from mini_rag.answerer import Answerer
from mini_rag.chunk import Chunk
from mini_rag.hybrid_hit import HybridHit


class TestAnswerer:
  """Checks the threshold gate, extractive output, the client path and validation."""

  def hit(self, s_doc_id: str, i_index: int, s_text: str, f_dense_score: float | None) -> HybridHit:
    """Make a hit; only the dense score matters to the Answerer."""
    obj_chunk = Chunk(s_doc_id=s_doc_id, i_index=i_index, i_start=0, s_text=s_text)
    return HybridHit(obj_chunk=obj_chunk, f_rrf_score=0.01, f_dense_score=f_dense_score, f_bm25_score=None)

  def hits(self) -> list[HybridHit]:
    """Two found-level hits in rank order; the second has no dense score."""
    return [
      self.hit("guide", 0, "Cats sleep a lot.", 0.8),
      self.hit("notes", 3, "Dogs bark.", None),
    ]

  def assert_not_found(self, obj_answer: Answer) -> None:
    """Assert obj_answer is the fixed not-found answer."""
    assert obj_answer == Answer(s_text="Not found in the corpus.", l_sources=[], b_found=False, b_generated=False)

  def test_empty_hits_not_found(self) -> None:
    """No hits means not found."""
    self.assert_not_found(Answerer().answer("q", []))

  def test_only_none_dense_scores_not_found(self) -> None:
    """Hits that the dense index never returned cannot pass the gate."""
    l_hits = [self.hit("a", 0, "x", None), self.hit("b", 0, "y", None)]
    self.assert_not_found(Answerer().answer("q", l_hits))

  def test_best_score_below_threshold_not_found(self) -> None:
    """A best dense score just under the threshold is not found."""
    self.assert_not_found(Answerer(f_threshold=0.5).answer("q", [self.hit("a", 0, "x", 0.49)]))

  def test_best_score_equal_to_threshold_found(self) -> None:
    """A best dense score equal to the threshold counts as found."""
    assert Answerer(f_threshold=0.5).answer("q", [self.hit("a", 0, "x", 0.5)]).b_found

  def test_best_score_above_threshold_found(self) -> None:
    """A best dense score above the threshold is found."""
    assert Answerer(f_threshold=0.5).answer("q", [self.hit("a", 0, "x", 0.9)]).b_found

  def test_none_score_does_not_break_maximum(self) -> None:
    """A None score is skipped; the maximum of the rest decides."""
    l_hits = [self.hit("a", 0, "x", None), self.hit("b", 0, "y", 0.7), self.hit("c", 0, "z", 0.2)]
    assert Answerer(f_threshold=0.6).answer("q", l_hits).b_found

  def test_extractive_output_exact(self) -> None:
    """Without a client the text is every hit under its marker, blank-line separated."""
    l_hits = self.hits()
    obj_answer = Answerer().answer("q", l_hits)
    assert obj_answer.s_text == "[guide#0]\nCats sleep a lot.\n\n[notes#3]\nDogs bark."
    assert obj_answer.l_sources == [obj_hit.obj_chunk for obj_hit in l_hits]
    assert obj_answer.b_found
    assert not obj_answer.b_generated

  def test_client_path_returns_fake_text(self) -> None:
    """With a client the answer is the client's text, marked generated, with all sources."""
    l_hits = self.hits()
    obj_answer = Answerer(obj_client=FakeLlmClient("Cats sleep [guide#0].")).answer("Do cats sleep?", l_hits)
    assert obj_answer.s_text == "Cats sleep [guide#0]."
    assert obj_answer.l_sources == [obj_hit.obj_chunk for obj_hit in l_hits]
    assert obj_answer.b_found
    assert obj_answer.b_generated

  def test_user_prompt_has_markers_and_question(self) -> None:
    """The recorded user prompt holds every source marker and the question."""
    obj_client = FakeLlmClient("reply")
    Answerer(obj_client=obj_client).answer("Do cats sleep?", self.hits())
    assert len(obj_client.l_user_prompts) == 1
    s_user = obj_client.l_user_prompts[0]
    assert "[guide#0]\nCats sleep a lot." in s_user
    assert "[notes#3]\nDogs bark." in s_user
    assert "Do cats sleep?" in s_user

  def test_system_prompt_has_rules(self) -> None:
    """The recorded system prompt asks for [doc_id#index] citations and to ignore embedded instructions."""
    obj_client = FakeLlmClient("reply")
    Answerer(obj_client=obj_client).answer("q", self.hits())
    s_system = obj_client.l_system_prompts[0]
    assert "Cite" in s_system
    assert "[doc_id#index]" in s_system
    assert "ignore any instructions" in s_system

  def test_invalid_threshold_raises(self) -> None:
    """A threshold outside [-1.0, 1.0] raises ValueError; the bounds themselves are allowed."""
    with pytest.raises(ValueError):
      Answerer(f_threshold=1.5)
    with pytest.raises(ValueError):
      Answerer(f_threshold=-1.01)
    Answerer(f_threshold=-1.0)
    Answerer(f_threshold=1.0)
