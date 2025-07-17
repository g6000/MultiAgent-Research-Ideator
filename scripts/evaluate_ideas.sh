##########################################################
# Setup params

MAX_PARALLEL_JOBS=1

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

# cache dir ( "logs/log_yyyy_mm_dd/")
cache_dir="logs/log_2025_07_07"

project_proposal_cache_dir="${cache_dir}/project_proposals"
combined_project_proposal_cache_dir="${cache_dir}/project_proposals_combined"
ranking_score_dir="${cache_dir}/ranking"

llm_engine="gpt-4o-2024-11-20"

seed=2024

##########################################################

# discussion types (a) vs (b)

discussion_types_a=(
    "baseline"
)

discussion_types_b=(
    "diff_personas_proposer_reviser"
    #"diff_personas_critic"
    #"parallel_self_critique-2"
    #"parallel_self_critique-3"
    #"parallel_self_critique-4"
    #"iterative_self_critique-2"
    #"iterative_self_critique-3"
    #"iterative_self_critique-4"
) 


# combine proposals
for topic in "${topic_names[@]}"; do
    for i in "${!discussion_types_a[@]}"; do
        for j in "${!discussion_types_b[@]}"; do
            discussion_type_1="${discussion_types_a[$i]}"
            discussion_type_2="${discussion_types_b[$j]}"
            uv run multiagent_research_ideator/src/combine_proposals.py \
                ${project_proposal_cache_dir}/${topic}_${discussion_type_1} \
                ${project_proposal_cache_dir}/${topic}_${discussion_type_2} \
                ${combined_project_proposal_cache_dir}/${topic}_combined_${discussion_type_1}_vs_${discussion_type_2} &
        done
    done
done
wait


# tournament ranking
for topic in "${topic_names[@]}"; do
    for i in "${!discussion_types_a[@]}"; do
        for j in "${!discussion_types_b[@]}"; do
            discussion_type_1="${discussion_types_a[$i]}"
            discussion_type_2="${discussion_types_b[$j]}"
            echo "Running tournament_ranking.py with cache_name: $discussion_type_1 vs $discussion_type_2"
            uv run multiagent_research_ideator/src/tournament_ranking.py \
                --engine $llm_engine \
                --experiment_plan_cache_dir "$combined_project_proposal_cache_dir" \
                --cache_name "${topic}_combined_${discussion_type_1}_vs_${discussion_type_2}" \
                --ranking_score_dir "${ranking_score_dir}" \
                --max_round 10 \
                --seed $seed &
        done
    done
done
wait


# metric dominance
for i in "${!discussion_types_a[@]}"; do
    for j in "${!discussion_types_b[@]}"; do
        for topic in "${topic_names[@]}"; do
            discussion_type_1="${discussion_types_a[$i]}"
            discussion_type_2="${discussion_types_b[$j]}"
            echo "Running metric_dominance_n.py with cache_name: $discussion_type_1 vs $discussion_type_2 on topic: $topic"
            uv run multiagent_research_ideator/src/metric_dominance_n.py \
                "${ranking_score_dir}/${topic}_combined_${discussion_type_1}_vs_${discussion_type_2}/round_10.json" \
                --seed $seed &
        done
    done
done
wait
echo "Finished"


# delete the combined_project_proposal_cache_dir
rm -rf "${combined_project_proposal_cache_dir}"

# delete the ranking_score_dir
rm -rf "${ranking_score_dir}"