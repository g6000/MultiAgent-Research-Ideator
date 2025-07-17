import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Tuple

import anthropic
import retry
from lit_review_tools import format_papers_for_printing
from openai import OpenAI
from utils import cache_output, call_api, shuffle_dict_and_convert_to_string

ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = ROOT / "prompts"

ANTH_KEY = os.getenv("anthropic_key")
OAI_KEY = os.getenv("api_key")
ORG_ID = os.getenv("organization_id")

REQUIRED_FIELDS = {
    "Problem",
    "Existing Methods",
    "Motivation",
    "Proposed Method",
    "Experiment Plan",
}


def check_idea_response_format(resp_str: str) -> Tuple[bool, str]:
    """
    Validate response JSON against the expected 'idea' schema.
    Returns (True, "") when valid, otherwise (False, <reason>).
    """
    try:
        data = json.loads(resp_str.strip())
    except json.JSONDecodeError as e:
        return False, f"JSON decoding failed: {e}"

    if not isinstance(data, dict):
        return False, "Top-level JSON is not a dictionary."

    if len(data.items()) < 1:
        return False, f"Expected more than or equal to 1 idea, got {len(data)}."

    for idea_name, idea_body in data.items():
        if not isinstance(idea_name, str):
            return False, f"Idea key {idea_name!r} is not a string."
        if not isinstance(idea_body, dict):
            return False, f"Value for {idea_name!r} is not a dictionary."

        missing = REQUIRED_FIELDS - set(idea_body.keys())
        if missing:
            return False, f"{idea_name!r} missing field(s): {', '.join(missing)}."

        excess = set(idea_body.keys()) - REQUIRED_FIELDS
        if excess:
            return False, f"{idea_name!r} has extra field(s): {', '.join(excess)}."

        for fld in REQUIRED_FIELDS:
            val = idea_body.get(fld, "")
            if not isinstance(val, str) or not val.strip():
                return False, f"{idea_name!r} → {fld} is empty or not a string."

    return True, ""  # all good


@retry.retry(tries=10, delay=3)
def propose_ideas(
    method,
    existing_ideas,
    paper_bank,
    grounding_k,
    examples,
    ideas_n,
    topic_description,
    openai_client,
    model,
    seed,
    temperature,
    top_p,
    max_tokens,
    prompt_role="You are an expert AI researcher.",
    RAG=True,
):
    ## retrieve top papers (with some randomization)
    top_papers = paper_bank[: int(grounding_k * 2)]
    random.shuffle(top_papers)
    grounding_papers = top_papers[:grounding_k]

    prompt = prompt_role + "\n\n"
    prompt += (
        "Now I want you to help me brainstorm some new research project ideas on the topic of: "
        + topic_description
        + ".\n\n"
    )
    if RAG:
        prompt += (
            "Here are some relevant papers on this topic just for your background knowledge:\n"
            + format_papers_for_printing(
                grounding_papers, include_score=False, include_id=False
            )
            + "\n"
        )
    prompt += f"You should generate {ideas_n} different ideas on this topic. Try to be creative and diverse in the idea generation, and do not repeat any similar ideas. "
    if RAG:
        prompt += "The above papers are only for inspiration and you should not cite them and just make some incremental modifications. Instead, you should make sure your ideas are novel and distinct from the prior literature. "
    prompt += "You should aim for projects that can potentially win best paper awards at top AI conferences like ACL and NeurIPS.\n"
    prompt += "Each idea should be described as: (1) Problem: State the problem statement, which should be closely related to the topic description and something that large language models cannot solve well yet. (2) Existing Methods: Mention some existing benchmarks and baseline methods if there are any. (3) Motivation: Explain the inspiration of the proposed method and why it would work well. (4) Proposed Method: Propose your new method and describe it in detail. The proposed method should be maximally different from all existing work and baselines, and be more advanced and effective than the baselines. You should be as creative as possible in proposing new methods, we love unhinged ideas that sound crazy. This should be the most detailed section of the proposal. (5) Experiment Plan: Specify the experiment steps, baselines, and evaluation metrics.\n"
    prompt += (
        "You can follow these examples to get a sense of how the ideas should be formatted (but don't borrow the ideas themselves):\n"
        + examples
        + "\n"
    )
    prompt += (
        "You should make sure to come up with your own novel and different ideas for the specified problem: "
        + topic_description
        + ". You should try to tackle important problems that are well recognized in the field and considered challenging for current models. For example, think of novel solutions for problems with existing benchmarks and baselines. In rare cases, you can propose to tackle a new problem, but you will have to justify why it is important and how to set up proper evaluation.\n"
    )
    # if "claude" in model:
    #    prompt += "You should make each idea standalone and not dependent on the other ideas.\n"
    if method == "prompting":
        prompt += "Focus on novel prompting ideas for now. The proposed method section should specify how to construct the prompts for all steps involved. Try to avoid large-scale pretraining experiments or human studies.\n"
    # elif method == "finetuning":
    #    prompt += "Focus on novel finetuning ideas for now. The proposed method section should specify how to get the finetuning data and what's the training objective.\n"
    # else:
    #    prompt += "Focus on proposing novel empirical methods, which can include prompting, finetuning, inference-time interventions, etc. The proposed method section should specify all the details involved, such as how to get the data, what's the training objective, how to construct the prompts, etc.\n"
    if existing_ideas:
        prompt += (
            "You should avoid repeating the following existing ideas and try to be different and diverse: "
            + existing_ideas
            + "\n"
        )
    prompt += f'Please write down your {ideas_n} ideas (each idea should be described as one paragraph. Output the ideas in json format as a dictionary, where you should generate a short idea name (e.g., "Non-Linear Story Understanding", or "Multi-Agent Negotiation") as the key and the actual idea description as the value (following the above format). Do not repeat idea names or contents.'

    response, cost = call_api(
        openai_client,
        model,
        [{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
        json_output=True,
    )

    valid, err = check_idea_response_format(response)
    if not valid:
        print("Format check failed:", err)
        # Throwing an exception triggers @retry for up to 3 automatic retries
        raise ValueError(err)

    return response, cost


@retry.retry(tries=10, delay=3)
def critique_ideas(
    current_ideas_json_str,
    topic_description,
    openai_client,
    model,
    seed,
    temperature,
    top_p,
    max_tokens,
    prompt_role,
    critique_prompt_template,
):
    prompt_critic = prompt_role + "\n\n"
    prompt_critic += f"You need to provide some constructive feedback to the given project proposal on the topic of: {topic_description}.\n\n"
    prompt_critic += "The project proposal (containing multiple ideas) is" + "\n\n"
    prompt_critic += current_ideas_json_str + "\n\n"
    prompt_critic += critique_prompt_template

    response, cost = call_api(
        openai_client,
        model,
        [{"role": "user", "content": prompt_critic}],
        temperature,
        top_p,
        max_tokens,
        seed,
        json_output=False,
    )

    return response, cost


@retry.retry(tries=10, delay=3)
def revise_ideas(
    current_ideas_json_str,
    response_critic,
    topic_description,
    openai_client,
    model,
    seed,
    temperature,
    top_p,
    max_tokens,
    prompt_role,
    revise_prompt_template,
):
    prompt_revise = prompt_role + "\n\n"
    prompt_revise += f"You previously proposed the following research ideas on the topic of: {topic_description}.\n\n"
    prompt_revise += (
        "The original project proposal (containing multiple ideas) is" + "\n\n"
    )
    prompt_revise += current_ideas_json_str + "\n\n"
    prompt_revise += (
        "However, the following criticisms were raised by expert reviewers: \n\n"
    )
    prompt_revise += response_critic.strip() + "\n\n"
    prompt_revise += revise_prompt_template + "\n\n"

    response, cost = call_api(
        openai_client,
        model,
        [{"role": "user", "content": prompt_revise}],
        temperature,
        top_p,
        max_tokens,
        seed,
        json_output=True,
    )

    valid, err = check_idea_response_format(response)
    if not valid:
        print("Format check failed:", err)
        # Throwing an exception triggers @retry for up to 3 automatic retries
        raise ValueError(err)

    return response, cost


def idea_generation_diverse_personas(
    method,
    existing_ideas,
    paper_bank,
    grounding_k,
    examples,
    ideas_n,
    topic_description,
    openai_client,
    model,
    seed,
    temperature,
    top_p,
    max_tokens,
    diverse_role,
    RAG=True,
):
    """
    Generates initial ideas, then has different 'persona' critics (using gpt-3.5-turbo)
    critique each idea from a specific domain perspective. Finally, regenerates ideas
    based on the collected critiques using the original generator model.
    """
    total_cost = 0

    # ========== STEP 1: Initial idea generation by persona (Generator LLM) ==========
    with open(PROMPTS_DIR / "prompts_persona.json", "r") as f:
        personas = json.load(f)

    if diverse_role == "proposer/reviser":
        prompt_role_proposer = personas[random.choice(list(personas.keys()))]
        prompt_role_critic = (
            "You are a critical reviewer for top AI conferences like NeurIPS or ACL."
        )
        prompt_role_reviser = prompt_role_proposer
    elif diverse_role == "critic":
        prompt_role_proposer = "You are an expert AI researcher."
        prompt_role_critic = personas[random.choice(list(personas.keys()))]
        prompt_role_reviser = "You are an expert AI researcher."
    else:
        raise ValueError(f"Invalid diverse_role: {diverse_role}")

    response_proposer, cost_proposer = propose_ideas(
        method=method,
        existing_ideas=existing_ideas,
        paper_bank=paper_bank,
        grounding_k=grounding_k,
        examples=examples,
        ideas_n=ideas_n,
        topic_description=topic_description,
        openai_client=openai_client,
        model=model,
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        prompt_role=prompt_role_proposer,
        RAG=RAG,
    )
    current_seed = seed + 1
    total_cost = cost_proposer

    # ========== STEP 2: Regular critique (Critic LLM) ==========
    response_critic, cost_critique = critique_ideas(
        current_ideas_json_str=response_proposer,
        topic_description=topic_description,
        openai_client=openai_client,
        model=model,
        seed=current_seed,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        prompt_role=prompt_role_critic,
        critique_prompt_template=critique_prompt_template,
    )
    total_cost += cost_critique

    # ========== STEP 3: Regeneration reflecting critique (Generator LLM) ==========
    response_revise, cost_revise = revise_ideas(
        current_ideas_json_str=response_proposer,
        response_critic=response_critic,
        topic_description=topic_description,
        openai_client=openai_client,
        model=model,
        seed=current_seed,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        prompt_role=prompt_role_reviser,
        revise_prompt_template=revise_prompt_template,
    )
    total_cost += cost_revise

    return response_revise, total_cost


def idea_generation_iterative_self_critique(
    method,
    existing_ideas,
    paper_bank,
    grounding_k,
    examples,
    ideas_n: int,
    topic_description,
    openai_client,
    model,
    seed,
    temperature,
    top_p,
    max_tokens,
    iterations,
    critique_prompt_template,
    revise_prompt_template,
    RAG=True,
):
    """
    Performs iterative self-critique for a specified number of iterations.
    Uses the same LLM for generation, critique, and regeneration.
    """
    total_cost = 0

    # ========== Initial Idea Generation (Iteration 0) ==========
    prompt_role_proposer = "You are an expert AI researcher."
    response_proposer, cost_proposer = propose_ideas(
        method=method,
        existing_ideas=existing_ideas,
        paper_bank=paper_bank,
        grounding_k=grounding_k,
        examples=examples,
        ideas_n=ideas_n,
        topic_description=topic_description,
        openai_client=openai_client,
        model=model,
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        prompt_role=prompt_role_proposer,
        RAG=RAG,
    )
    current_seed = seed + 1
    total_cost += cost_proposer

    # ========== Iterative Critique and Regeneration ==========
    for i in range(iterations):
        print(f"--- Iteration {i + 1} of {iterations} ---")

        # --- Step 2: Critique ---
        prompt_role_critic = "You are an expert AI researcher."
        response_critic, cost_critique = critique_ideas(
            current_ideas_json_str=response_proposer,
            topic_description=topic_description,
            openai_client=openai_client,
            model=model,
            seed=current_seed,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            prompt_role=prompt_role_critic,
            critique_prompt_template=critique_prompt_template,
        )
        total_cost += cost_critique

        # --- Step 3: Regenerate based on critique ---
        prompt_role_reviser = "You are an expert AI researcher."
        response_revise, cost_revise = revise_ideas(
            current_ideas_json_str=response_proposer,
            response_critic=response_critic,
            topic_description=topic_description,
            openai_client=openai_client,
            model=model,
            seed=current_seed,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            prompt_role=prompt_role_reviser,
            revise_prompt_template=revise_prompt_template,
        )
        total_cost += cost_revise
        response_proposer = response_revise

        current_seed += 1

    # Return the initial prompt, the final response after all iterations, and total cost
    return response_proposer, total_cost


def idea_generation_parallel_self_critique(
    method,
    existing_ideas,
    paper_bank,
    grounding_k,
    examples,
    ideas_n: int,
    topic_description,
    openai_client,
    model,
    seed,
    temperature,
    top_p,
    max_tokens,
    n_critics,
    critique_prompt_template,
    revise_prompt_template,
    RAG=True,
):
    """
    Performs parallel self-critique for a specified number of iterations.
    Uses different LLMs for generation, critique, and regeneration.
    """
    total_cost = 0

    # ========== Initial Idea Generation (Iteration 0) ==========
    prompt_role_proposer = "You are an expert AI researcher."
    response_proposer, cost_proposer = propose_ideas(
        method=method,
        existing_ideas=existing_ideas,
        paper_bank=paper_bank,
        grounding_k=grounding_k,
        examples=examples,
        ideas_n=ideas_n,
        topic_description=topic_description,
        openai_client=openai_client,
        model=model,
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        prompt_role=prompt_role_proposer,
        RAG=RAG,
    )
    current_seed = seed + 1
    total_cost += cost_proposer

    # ========== Parallel Self-Critique ==========
    list_response_critic = []
    cost_critic_batch = 0
    for i in range(n_critics):
        print(f"--- Critic {i + 1} of {n_critics} ---")
        # --- Step 2: Critique ---
        prompt_role_critic = "You are an expert AI researcher."
        response_critic_batch, cost_critique_batch = critique_ideas(
            current_ideas_json_str=response_proposer,
            topic_description=topic_description,
            openai_client=openai_client,
            model=model,
            seed=current_seed,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            prompt_role=prompt_role_critic,
            critique_prompt_template=critique_prompt_template,
        )
        list_response_critic.append(response_critic_batch)
        cost_critic_batch += cost_critique_batch
        current_seed += 1

    response_critic = "\n\n".join(list_response_critic)
    total_cost += cost_critic_batch

    # --- Step 3: Regenerate based on critique ---
    prompt_role_reviser = "You are an expert AI researcher."
    response_revise, cost_revise = revise_ideas(
        current_ideas_json_str=response_proposer,
        response_critic=response_critic,
        topic_description=topic_description,
        openai_client=openai_client,
        model=model,
        seed=current_seed,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        prompt_role=prompt_role_reviser,
        revise_prompt_template=revise_prompt_template,
    )
    response_proposer = response_revise
    total_cost += cost_revise

    # Return the initial prompt, the final response after all iterations, and total cost
    return response_proposer, total_cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine",
        type=str,
        default="claude-3-opus-20240229",
        help="api engine; https://openai.com/api/",
    )
    parser.add_argument(
        "--paper_cache",
        type=str,
        default=None,
        required=True,
        help="cache file name for the retrieved papers",
    )
    parser.add_argument(
        "--idea_cache",
        type=str,
        default=None,
        required=True,
        help="where to store the generated ideas",
    )
    parser.add_argument(
        "--RAG",
        type=str,
        default="True",
        required=True,
        help="whether to do RAG for idea generation",
    )
    parser.add_argument(
        "--method", type=str, default="prompting", help="either prompting or finetuning"
    )
    parser.add_argument(
        "--grounding_k",
        type=int,
        default=10,
        help="how many papers to use for grounding",
    )
    parser.add_argument(
        "--append_existing_ideas",
        type=str,
        default="True",
        help="whether to append existing ideas to the idea cache",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=30000, help="max tokens in the output"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="temperature in sampling"
    )
    parser.add_argument("--top_p", type=float, default=1.0, help="top p in sampling")
    parser.add_argument(
        "--ideas_n", type=int, default=5, help="how many ideas to generate"
    )
    parser.add_argument(
        "--seed", type=int, default=2024, help="seed for GPT-4 generation"
    )
    parser.add_argument("--debug", action="store_true", help="enable debug mode")
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help="Number of critique/regeneration iterations (for iterative_self_critique)",
    )
    parser.add_argument(
        "--n_critics",
        type=int,
        default=0,
        help="Number of critics (for parallel_self_critique)",
    )
    parser.add_argument(
        "--discussion_type",
        type=str,
        default="single",
        help="discussion type for idea generation",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    if "claude" in args.engine:
        client = anthropic.Anthropic(
            api_key=ANTH_KEY,
        )
    elif "o1" in args.engine or "gpt" in args.engine:
        client = OpenAI(
            # organization=ORG_ID,
            api_key=OAI_KEY
        )
    else:
        raise ValueError(f"Unsupported engine: {args.engine}")

    with open(args.paper_cache, "r") as f:
        lit_review = json.load(f)

    topic_description = lit_review["topic_description"]
    paper_bank = lit_review["paper_bank"]

    ## cache dir and file
    if args.RAG == "True":
        print("RAG is enabled for idea generation")
    else:
        print("RAG is disabled for idea generation")
    ideas_file = args.idea_cache

    # extract existing ideas
    existing_ideas = None
    if os.path.exists(ideas_file) and args.append_existing_ideas == "True":
        with open(ideas_file, "r") as f:
            ideas_cache = json.load(f)
        if "ideas" in ideas_cache:
            existing_ideas = [
                key for idea in ideas_cache["ideas"] for key in idea.keys()
            ]
            existing_ideas = list(set(existing_ideas))
            existing_ideas = "; ".join(existing_ideas)
            print("Appending previous ideas.")
    else:
        print("Not appending previous ideas.")

    if args.method == "prompting":
        with open(PROMPTS_DIR / "idea_examples_prompting_method.json", "r") as f:
            method_idea_examples = json.load(f)
            method_idea_examples = shuffle_dict_and_convert_to_string(
                method_idea_examples
            )
    elif args.method == "finetuning":
        with open(PROMPTS_DIR / "idea_examples_finetuning_method.json", "r") as f:
            method_idea_examples = json.load(f)
            method_idea_examples = shuffle_dict_and_convert_to_string(
                method_idea_examples
            )
    else:
        with open(PROMPTS_DIR / "idea_examples_method.json", "r") as f:
            method_idea_examples = json.load(f)
            method_idea_examples = shuffle_dict_and_convert_to_string(
                method_idea_examples, n=4
            )

    try:
        with open(PROMPTS_DIR / "self_critique_prompt.txt", "r") as f:
            critique_prompt_template = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{PROMPTS_DIR}/self_critique_prompt.txt not found. Please make sure the file exists in the prompts directory."
        )

    try:
        with open(PROMPTS_DIR / "self_revise_prompt.txt", "r") as f:
            revise_prompt_template = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{PROMPTS_DIR}/self_revise_prompt.txt not found. Please make sure the file exists in the prompts directory."
        )

    print("topic: ", topic_description)
    print("existing ideas: ", existing_ideas)
    print("\n")
    print("generating {} ideas...".format(str(args.ideas_n)))

    # Jump to a generation method based on specified discussion type.
    if args.discussion_type == "single":
        response, cost = idea_generation_iterative_self_critique(
            args.method,
            existing_ideas,
            paper_bank,
            args.grounding_k,
            method_idea_examples,
            args.ideas_n,
            topic_description,
            client,
            args.engine,
            args.seed,
            args.temperature,
            args.top_p,
            args.max_tokens,
            iterations=0,
            critique_prompt_template=critique_prompt_template,
            revise_prompt_template=revise_prompt_template,
            RAG=args.RAG,
        )
    elif args.discussion_type == "baseline":
        response, cost = idea_generation_iterative_self_critique(
            args.method,
            existing_ideas,
            paper_bank,
            args.grounding_k,
            method_idea_examples,
            args.ideas_n,
            topic_description,
            client,
            args.engine,
            args.seed,
            args.temperature,
            args.top_p,
            args.max_tokens,
            iterations=1,
            critique_prompt_template=critique_prompt_template,
            revise_prompt_template=revise_prompt_template,
            RAG=args.RAG,
        )
    elif args.discussion_type == "diff_personas_proposer_reviser":
        response, cost = idea_generation_diverse_personas(
            args.method,
            existing_ideas,
            paper_bank,
            args.grounding_k,
            method_idea_examples,
            args.ideas_n,
            topic_description,
            client,
            args.engine,
            args.seed,
            args.temperature,
            args.top_p,
            args.max_tokens,
            diverse_role="proposer/reviser",
            RAG=args.RAG,
        )
    elif args.discussion_type == "diff_personas_critic":
        response, cost = idea_generation_diverse_personas(
            args.method,
            existing_ideas,
            paper_bank,
            args.grounding_k,
            method_idea_examples,
            args.ideas_n,
            topic_description,
            client,
            args.engine,
            args.seed,
            args.temperature,
            args.top_p,
            args.max_tokens,
            diverse_role="critic",
            RAG=args.RAG,
        )
    elif "iterative_self_critique" in args.discussion_type:
        response, cost = idea_generation_iterative_self_critique(
            args.method,
            existing_ideas,
            paper_bank,
            args.grounding_k,
            method_idea_examples,
            args.ideas_n,
            topic_description,
            client,
            args.engine,
            args.seed,
            args.temperature,
            args.top_p,
            args.max_tokens,
            args.iterations,
            critique_prompt_template,
            revise_prompt_template,
            RAG=args.RAG,
        )
    elif "parallel_self_critique" in args.discussion_type:
        response, cost = idea_generation_parallel_self_critique(
            args.method,
            existing_ideas,
            paper_bank,
            args.grounding_k,
            method_idea_examples,
            args.ideas_n,
            topic_description,
            client,
            args.engine,
            args.seed,
            args.temperature,
            args.top_p,
            args.max_tokens,
            args.n_critics,
            critique_prompt_template,
            revise_prompt_template,
            RAG=args.RAG,
        )
    else:
        print(f"Unsupported discussion type: {args.discussion_type}")
        sys.exit(1)

    print("idea generation cost: ", cost)

    response = json.loads(response.strip())
    ideas = {"topic_description": topic_description, "ideas": [response]}

    ## if the idea_cache already exists, directly add to the current list
    if os.path.exists(ideas_file):
        with open(ideas_file, "r") as f:
            ideas_cache = json.load(f)
        ideas_cache["ideas"].append(response)
        ideas = ideas_cache

    print("#ideas generated so far: ", sum(len(d) for d in ideas["ideas"]))

    ## save the cache
    cache_dir = os.path.dirname(ideas_file)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cache_output(ideas, ideas_file)
