from typing import Dict, List, Any

from .greedy import run_eval_greedy
from .tf_loss import run_eval_tf_loss


def run_eval(
    model,
    tokenizer,
    inputs: List[str],
    labels: List[List[str]],
    print_examples: int = 0,
    seed: int = 0,
    use_chat_template: bool = False,
) -> Dict[str, Any]:
    tf_loss = run_eval_tf_loss(model, tokenizer, inputs, labels,
                               seed=seed, use_chat_template=use_chat_template)

    greedy_result = run_eval_greedy(model, tokenizer, inputs, labels,
                                    print_examples=print_examples,
                                    use_chat_template=use_chat_template)

    return {
        "first_token_acc": greedy_result["first_token_acc"],
        "full_word_acc": greedy_result["full_word_acc"],
        "first_token_hits": greedy_result["first_token_hits"],
        "full_word_hits": greedy_result["full_word_hits"],
        "tf_loss": tf_loss,
        "total": greedy_result["total"],
    }
