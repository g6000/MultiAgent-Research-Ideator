##########################################################
# Setup params

MAX_PARALLEL_JOBS=7

# grounded idea generation
topic_names=(
    "bias_prompting_method"
    #"coding_prompting_method"
    #"safety_prompting_method"
    #"multilingual_prompting_method"
    #"factuality_prompting_method"
    #"math_prompting_method"
    #"uncertainty_prompting_method"
)

topic_descriptions=(
    "novel prompting methods to reduce social biases and stereotypes of large language models"
    #"novel prompting methods for large language models to improve code generation"
    #"novel prompting methods to improve large language models' robustness against adversarial attacks or improve their security or privacy"
    #"novel prompting methods to improve large language models' performance on multilingual tasks or low-resource languages and vernacular languages"
    #"novel prompting methods that can improve factuality and reduce hallucination of large language models"
    #"novel prompting methods for large language models to improve mathematical problem solving"
    #"novel prompting methods that can better quantify uncertainty or calibrate the confidence of large language models"
)

discussion_types=(
    "single"
    #"baseline"
    #"diff_personas_proposer_reviser"
    #"diff_personas_critic"
    #"parallel_self_critique-2"
    #"parallel_self_critique-3"
    #"parallel_self_critique-4"
    #"iterative_self_critique-2"
    #"iterative_self_critique-3"
    #"iterative_self_critique-4"
) 

# cache dir ( "logs/log_yyyy_mm_dd/")
cache_dir="logs/log_2025_07_07"

llm_engine="gpt-4o-mini-2024-07-18"


##########################################################
# Run Literature Review 
# After we run this, we get lit_review review in the cache dir.

for i in "${!topic_names[@]}"; do
    topic=${topic_names[$i]}
    topic_description=${topic_descriptions[$i]}
    uv run multiagent_research_ideator/src/lit_review.py \
        --engine $llm_engine \
        --mode "topic" \
        --topic_description "${topic_description}" \
        --cache_name "${cache_dir}/lit_review/${topic}.json" \
        --max_paper_bank_size 100 \
        --print_all &
done
wait
