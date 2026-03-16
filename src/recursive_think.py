import re
import sys
import time
from pathlib import Path

import torch

from model import TinyTransformerLanguageModel
from tokenizer_utils import load_vocab, encode, decode

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / "models" / "tiny_word_transformer.pt"


class AutonomousReasoningController:
    def __init__(self):
        self.model, self.vocab, self.seq_len = self.load_model_bundle()

    def load_model_bundle(self):
        checkpoint = torch.load(MODEL_FILE, map_location="cpu")
        vocab = load_vocab()

        model = TinyTransformerLanguageModel(
            vocab_size=checkpoint["vocab_size"],
            d_model=checkpoint["d_model"],
            nhead=checkpoint["nhead"],
            num_layers=checkpoint["num_layers"],
            dim_feedforward=checkpoint["dim_feedforward"],
            dropout=checkpoint["dropout"],
            max_seq_len=checkpoint["seq_len"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        return model, vocab, checkpoint["seq_len"]

    def sample_next_token(self, logits, temperature=0.62, top_k=10):
        temperature = max(temperature, 1e-5)
        logits = logits / temperature

        k = min(top_k, logits.size(-1))
        values, indices = torch.topk(logits, k=k)
        probs = torch.softmax(values, dim=-1)

        next_index = torch.multinomial(probs, num_samples=1).item()
        return indices[next_index].item()

    def generate_text(self, prompt, max_new_chars=220, temperature=0.62, top_k=10):
        prompt = prompt if prompt.strip() else "Start."
        generated = encode(prompt, self.vocab)
        if not generated:
            return ""

        original_length = len(generated)

        for _ in range(max_new_chars):
            current_input = generated[-self.seq_len :]
            current_tensor = torch.tensor([current_input], dtype=torch.long)

            with torch.no_grad():
                logits = self.model(current_tensor)

            next_token = self.sample_next_token(
                logits[0, -1, :],
                temperature=temperature,
                top_k=top_k,
            )
            generated.append(next_token)

        new_tokens = generated[original_length:]
        return decode(new_tokens, self.vocab)

    def paced_print(
        self,
        text,
        char_delay=0.010,
        punctuation_delay=0.05,
        sentence_delay=0.10,
        newline_delay=0.12,
    ):
        for ch in text:
            sys.stdout.write(ch)
            sys.stdout.flush()

            if ch in ".!?":
                time.sleep(sentence_delay)
            elif ch in ",;:":
                time.sleep(punctuation_delay)
            elif ch == "\n":
                time.sleep(newline_delay)
            else:
                time.sleep(char_delay)

    def clean_text(self, text):
        return re.sub(r"\s+", " ", text).strip()

    def normalize_for_display(self, text):
        text = self.clean_text(text)
        if not text:
            return ""

        leaked_prefixes = [
            "thought:",
            "critique:",
            "decision:",
            "decisionreason:",
            "reason:",
            "question:",
            "reply:",
            "user:",
            "assistant:",
            "internalthought:",
        ]

        changed = True
        while changed:
            changed = False
            lower_text = text.lower()
            for prefix in leaked_prefixes:
                if lower_text.startswith(prefix):
                    text = text[len(prefix) :].strip(" -:\n\t")
                    changed = True
                    break

        text = re.sub(r"[ ]{2,}", " ", text)
        text = re.sub(r"([?.!,]){2,}", r"\1", text)
        return text.strip()

    def limit_to_sentences(self, text, max_sentences=2):
        text = self.normalize_for_display(text)
        if not text:
            return ""

        parts = re.split(r"(?<=[.!?])\s+", text)
        parts = [p.strip() for p in parts if p.strip()]
        if not parts:
            return text

        return " ".join(parts[:max_sentences]).strip()

    def repetition_score(self, text):
        words = re.findall(r"[A-Za-z']+", text.lower())
        if len(words) < 8:
            return 0.0

        unique_ratio = len(set(words)) / max(1, len(words))
        return 1.0 - unique_ratio

    def thought_is_low_quality(self, thought):
        if not thought:
            return True

        words = thought.split()
        if len(words) < 10:
            return True

        lower = thought.lower()

        bad_patterns = [
            "the problem is the problem",
            "i should i should",
            "questionforuser",
            "needsuserinput",
            "decision:",
            "critique:",
            "thought:",
            "reply:",
            "internalthought:",
        ]

        if any(pattern in lower for pattern in bad_patterns):
            return True

        if self.repetition_score(thought) > 0.60:
            return True

        return False

    def extract_keywords(self, text, max_keywords=4):
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "if",
            "then",
            "than",
            "to",
            "of",
            "for",
            "from",
            "with",
            "without",
            "on",
            "in",
            "at",
            "by",
            "about",
            "into",
            "is",
            "are",
            "was",
            "were",
            "be",
            "being",
            "been",
            "it",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "we",
            "they",
            "he",
            "she",
            "them",
            "us",
            "my",
            "your",
            "our",
            "their",
            "me",
            "do",
            "does",
            "did",
            "can",
            "could",
            "should",
            "would",
            "will",
            "just",
            "so",
            "what",
            "why",
            "how",
            "when",
            "where",
            "which",
            "who",
            "whom",
            "not",
            "no",
            "yes",
            "sure",
            "okay",
            "ok",
            "please",
            "want",
            "need",
            "help",
        }

        words = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text.lower())
        freq = {}
        for word in words:
            if word in stop_words:
                continue
            freq[word] = freq.get(word, 0) + 1

        ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
        return [word for word, _ in ranked[:max_keywords]]

    def generate_thought_segment(self, context, max_new_chars=220):
        prompt = (
            f"{context}\nContinue thinking in a useful and grounded way.\nThought: "
        )

        raw = self.generate_text(
            prompt,
            max_new_chars=max_new_chars,
            temperature=0.62,
            top_k=10,
        )
        return self.limit_to_sentences(raw, max_sentences=3)

    def generate_self_critique(self, current_thought, context):
        prompt = (
            "Review the current thought.\n"
            "State the main weakness, missing information, ambiguity, or reason to continue.\n"
            f"Context: {context}\n"
            f"Thought: {current_thought}\n"
            "Critique: "
        )

        raw = self.generate_text(
            prompt,
            max_new_chars=120,
            temperature=0.55,
            top_k=8,
        )
        return self.limit_to_sentences(raw, max_sentences=2)

    def generate_decision_reason(
        self, context, current_thought, critique, cycle_number
    ):
        prompt = (
            "Review the reasoning state and decide whether more internal thinking is useful.\n"
            "A good decision reason must identify one of these situations:\n"
            "- more internal progress is possible\n"
            "- progress is blocked by missing user information\n"
            "- the reasoning is sufficient and further thinking would mostly repeat\n\n"
            f"Cycle: {cycle_number}\n"
            f"Context: {context}\n"
            f"Thought: {current_thought}\n"
            f"Critique: {critique}\n"
            "Decision reason: "
        )

        raw = self.generate_text(
            prompt,
            max_new_chars=120,
            temperature=0.50,
            top_k=8,
        )
        return self.limit_to_sentences(raw, max_sentences=2)

    def decide_next_action(
        self,
        context,
        current_thought,
        critique,
        decision_reason,
        cycle_number,
        max_cycles,
    ):
        prompt = (
            "Choose exactly one label from this set:\n"
            "CONTINUE\n"
            "ASK_USER\n"
            "STOP\n\n"
            "Use CONTINUE if more internal reasoning is likely to improve the result without new user input.\n"
            "Use ASK_USER if the next useful step depends on a missing fact, missing goal, missing preference, or missing constraint that only the user can provide.\n"
            "Use STOP if the reasoning is already sufficient and more internal thinking would mostly repeat.\n\n"
            f"Cycle: {cycle_number}\n"
            f"Max cycles: {max_cycles}\n"
            f"Context: {context}\n"
            f"Thought: {current_thought}\n"
            f"Critique: {critique}\n"
            f"Decision reason: {decision_reason}\n"
            "Decision: "
        )

        raw = self.normalize_for_display(
            self.generate_text(
                prompt,
                max_new_chars=30,
                temperature=0.40,
                top_k=5,
            )
        ).upper()

        if "ASK_USER" in raw:
            return "ASK_USER"
        if "STOP" in raw:
            return "STOP"
        if "CONTINUE" in raw:
            return "CONTINUE"

        combined = " ".join(
            [
                current_thought.lower(),
                critique.lower(),
                decision_reason.lower(),
            ]
        )

        ask_markers = [
            "missing fact",
            "missing goal",
            "missing preference",
            "missing constraint",
            "need the user",
            "need user input",
            "depends on the user",
            "depends on user intent",
            "depends on user preference",
            "depends on a missing detail",
            "blocked",
            "cannot proceed",
            "cannot choose",
            "multiple plausible directions",
            "ambiguity requires the user",
            "need clarification",
            "unclear goal",
            "unclear constraint",
        ]

        stop_markers = [
            "sufficient",
            "enough",
            "already clear",
            "complete",
            "ready to stop",
            "mostly repeat",
            "further thinking would repeat",
            "no useful gain",
        ]

        if any(marker in combined for marker in ask_markers):
            return "ASK_USER"

        if any(marker in combined for marker in stop_markers):
            return "STOP"

        if self.thought_is_low_quality(current_thought):
            return "ASK_USER"

        if self.repetition_score(current_thought + " " + critique) > 0.62:
            return "ASK_USER"

        if cycle_number >= max_cycles:
            return "ASK_USER"

        return "CONTINUE"

    def generate_user_question(self, user_input, thought, critique, decision_reason):
        prompt = (
            "Generate one concise question for the user.\n"
            "The question must follow naturally from the reasoning state.\n"
            "Ask only for the single most important missing detail.\n"
            "Do not explain the reasoning.\n"
            "Do not include labels.\n"
            "Keep it short and conversational.\n\n"
            f"User input: {user_input}\n"
            f"Thought: {thought}\n"
            f"Critique: {critique}\n"
            f"Decision reason: {decision_reason}\n"
            "Question: "
        )

        raw = self.generate_text(
            prompt,
            max_new_chars=60,
            temperature=0.50,
            top_k=8,
        )

        question = self.limit_to_sentences(raw, max_sentences=1)
        question = self.normalize_for_display(question)

        if question.endswith("."):
            question = question[:-1].strip() + "?"
        elif not question.endswith("?"):
            question = question.rstrip("!. ") + "?"

        # Tiny emergency fallback only if generation is empty or unusable
        if not question or len(question.split()) < 3:
            keywords = self.extract_keywords(
                " ".join([user_input, thought, critique, decision_reason])
            )
            if keywords:
                question = f"What should I understand about {keywords[0]}?"
            else:
                question = "Can you clarify?"

        return question

    def build_answer_from_reasoning(
        self, user_input, thought, critique, decision_reason
    ):
        thought_clean = self.limit_to_sentences(thought, max_sentences=2)

        if not thought_clean:
            return "I do not understand the situation clearly enough yet."

        if self.thought_is_low_quality(thought_clean):
            return "I do not understand the situation clearly enough yet."

        generic_markers = [
            "the problem",
            "useful action",
            "current goal",
            "next step",
            "situation before proceeding",
        ]
        generic_count = sum(1 for m in generic_markers if m in thought_clean.lower())

        if generic_count >= 3:
            keywords = self.extract_keywords(user_input)
            if keywords:
                return f"I think this is about {', '.join(keywords[:2])}, but I still need a clearer goal."
            return "I think I understand part of the situation, but I still need a clearer goal."

        answer = thought_clean

        if answer.endswith("?"):
            answer = answer[:-1].rstrip(".! ")
            if answer:
                answer += "."
            else:
                answer = "I do not understand the situation clearly enough yet."

        return answer

    def run_reasoning_loop(
        self,
        conversation_context,
        user_input="",
        is_opening_turn=False,
        max_cycles=3,
        thought_chars_per_cycle=180,
        show_output=True,
    ):
        if user_input.strip():
            context = (f"{conversation_context}\nUser: {user_input}\n").strip()
        else:
            context = conversation_context.strip()

        if not context:
            context = "The system is active and thinking."

        history = []

        for cycle in range(1, max_cycles + 1):
            thought = self.generate_thought_segment(
                context=context,
                max_new_chars=thought_chars_per_cycle,
            )
            critique = self.generate_self_critique(thought, context)
            decision_reason = self.generate_decision_reason(
                context=context,
                current_thought=thought,
                critique=critique,
                cycle_number=cycle,
            )

            if is_opening_turn and cycle == 1:
                decision = "ASK_USER"
            else:
                decision = self.decide_next_action(
                    context=context,
                    current_thought=thought,
                    critique=critique,
                    decision_reason=decision_reason,
                    cycle_number=cycle,
                    max_cycles=max_cycles,
                )

            result = {
                "cycle": cycle,
                "thought": thought,
                "critique": critique,
                "decision_reason": decision_reason,
                "decision": decision,
            }
            history.append(result)

            if decision == "CONTINUE":
                context += (
                    f"\nPrevious thought: {thought}\n"
                    f"Critique: {critique}\n"
                    f"Decision reason: {decision_reason}\n"
                    "Continue from the strongest useful point.\n"
                )
                continue

            final_thought = thought
            final_critique = critique
            final_reason = decision_reason

            if decision == "ASK_USER":
                final_line = self.generate_user_question(
                    user_input=user_input,
                    thought=final_thought,
                    critique=final_critique,
                    decision_reason=final_reason,
                )
            else:
                final_line = self.build_answer_from_reasoning(
                    user_input=user_input,
                    thought=final_thought,
                    critique=final_critique,
                    decision_reason=final_reason,
                )

            if show_output:
                if final_thought.strip():
                    self.paced_print(final_thought.strip() + "\n\n")
                if final_line.strip():
                    self.paced_print(final_line.strip() + "\n\n")

            return {
                "history": history,
                "thought": final_thought,
                "critique": final_critique,
                "decision_reason": final_reason,
                "decision": decision,
                "final_line": final_line,
            }

        final = (
            history[-1]
            if history
            else {
                "thought": "",
                "critique": "",
                "decision_reason": "",
                "decision": "ASK_USER",
            }
        )

        final_line = self.generate_user_question(
            user_input=user_input,
            thought=final["thought"],
            critique=final["critique"],
            decision_reason=final["decision_reason"],
        )

        if show_output:
            if final["thought"].strip():
                self.paced_print(final["thought"].strip() + "\n\n")
            if final_line.strip():
                self.paced_print(final_line.strip() + "\n\n")

        return {
            "history": history,
            "thought": final["thought"],
            "critique": final["critique"],
            "decision_reason": final["decision_reason"],
            "decision": "ASK_USER",
            "final_line": final_line,
        }

    def chat(self):
        print()
        print("Type 'exit', 'quit', or 'stop' to end.\n")

        conversation_context = ""

        opening_result = self.run_reasoning_loop(
            conversation_context=conversation_context,
            user_input="",
            is_opening_turn=True,
            max_cycles=2,
            thought_chars_per_cycle=180,
            show_output=True,
        )

        conversation_context += (
            f"InternalThought: {opening_result['thought']}\n"
            f"Critique: {opening_result['critique']}\n"
            f"DecisionReason: {opening_result['decision_reason']}\n"
            f"Reply: {opening_result['final_line']}\n"
        )

        while True:
            user_input = input("> ").strip()

            if user_input.lower() in {"exit", "quit", "stop"}:
                print("\nGoodbye.\n")
                break

            if not user_input:
                user_input = "Continue"

            result = self.run_reasoning_loop(
                conversation_context=conversation_context,
                user_input=user_input,
                is_opening_turn=False,
                max_cycles=3,
                thought_chars_per_cycle=180,
                show_output=True,
            )

            conversation_context += (
                f"User: {user_input}\n"
                f"InternalThought: {result['thought']}\n"
                f"Critique: {result['critique']}\n"
                f"DecisionReason: {result['decision_reason']}\n"
                f"Reply: {result['final_line']}\n"
            )


def main():
    thinker = AutonomousReasoningController()
    thinker.chat()


if __name__ == "__main__":
    main()
