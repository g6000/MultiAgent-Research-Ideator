##########################################################
# Setup params

MAX_PARALLEL_JOBS=32

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
    #"single"
    "baseline"
    "diff_personas_proposer_reviser"
    #"diff_personas_critic"
    #"parallel_self_critique-2"
    #"parallel_self_critique-3"
    #"parallel_self_critique-4"
    #"iterative_self_critique-2"
    #"iterative_self_critique-3"
    #"iterative_self_critique-4"
) 

ideas_n=5
method="prompting"
rag_value="True"

# cache dir ( "logs/log_yyyy_mm_dd/")
cache_dir="logs/log_2025_07_07"

seed_ideas_cache_dir="${cache_dir}/ideas_seed"
idea_dedup_cache_dir="${cache_dir}/ideas_dedup"
project_proposal_cache_dir="${cache_dir}/project_proposals"

llm_engine="gpt-4o-mini-2024-07-18"

###########################################################

# Run grounded idea generation

# Iterate over each topic name 
for topic in "${topic_names[@]}"; do
    for discussion_type in "${discussion_types[@]}"; do
        for seed in {1..2}; do
            # Limit concurrent jobs - improved job control
            while (( $(jobs -r | wc -l) >= MAX_PARALLEL_JOBS )); do
                wait -n 2>/dev/null || sleep 1
            done

            echo "Running grounded idea generation with seed $seed"

            if [ -f "${seed_ideas_cache_dir}/${topic}_${discussion_type}_seed${seed}.json" ]; then
                echo "File already exists: ${seed_ideas_cache_dir}/${topic}_${discussion_type}_seed${seed}.json"
                echo "Skipping...\n"
                continue
            fi

            echo "Running grounded_idea_gen.py on: $topic with seed $seed and RAG=$rag_value with discussion_type=$discussion_type"

            # if discussion_type contains "iterative", split it to get iterations
            if [[ $discussion_type == *"iterative"* && $discussion_type == *"-"* ]]; then
                iterations=$(echo $discussion_type | cut -d'-' -f2)
                echo "Iterations: $iterations"
                uv run multiagent_research_ideator/src/grounded_idea_gen.py \
                    --engine $llm_engine \
                    --max_tokens 16384 \
                    --paper_cache "${cache_dir}/lit_review/$topic.json" \
                    --idea_cache "${seed_ideas_cache_dir}/${topic}_${discussion_type}_seed${seed}.json" \
                    --grounding_k 10 \
                    --method "$method" \
                    --ideas_n $ideas_n \
                    --seed $seed \
                    --discussion_type $discussion_type \
                    --RAG $rag_value \
                    --iterations $iterations \
                    --debug &
            elif [[ $discussion_type == *"parallel_self_critique"* ]]; then
                n_critics=$(echo $discussion_type | cut -d'-' -f2)
                echo "Critics: $n_critics"
                uv run multiagent_research_ideator/src/grounded_idea_gen.py \
                    --engine $llm_engine \
                    --max_tokens 16384 \
                    --paper_cache "${cache_dir}/lit_review/$topic.json" \
                    --idea_cache "${seed_ideas_cache_dir}/${topic}_${discussion_type}_seed${seed}.json" \
                    --grounding_k 10 \
                    --method "$method" \
                    --ideas_n $ideas_n \
                    --seed $seed \
                    --discussion_type $discussion_type \
                    --RAG $rag_value \
                    --n_critics $n_critics \
                    --debug &
            else
                uv run multiagent_research_ideator/src/grounded_idea_gen.py \
                    --engine $llm_engine \
                    --max_tokens 16384 \
                    --paper_cache "${cache_dir}/lit_review/$topic.json" \
                    --idea_cache "${seed_ideas_cache_dir}/${topic}_${discussion_type}_seed${seed}.json" \
                    --grounding_k 10 \
                    --method "$method" \
                    --ideas_n $ideas_n \
                    --seed $seed \
                    --discussion_type $discussion_type \
                    --RAG $rag_value \
                    --debug &
            fi
        done
    done
done

wait 


echo "Finished all grounded idea generation processes."


###########################################################

# Merge temporary idea files per discussion type using Python script
echo "Merging temporary idea files using Python script..."
# Define cache directory (ensure it's consistent)

for topic in "${topic_names[@]}"; do
    for discussion_type in "${discussion_types[@]}"; do
        final_cache_file="${seed_ideas_cache_dir}/${topic}_${discussion_type}.json"
        echo "Running merge script for ${topic} - ${discussion_type}"
        # Call the Python script to merge files for this combination
        # It will find _seed*.json files inside the cache_dir
        uv run multiagent_research_ideator/src/merge_seed_ideas.py \
            --cache_dir "${seed_ideas_cache_dir}" \
            --topic "${topic}" \
            --discussion_type "${discussion_type}" \
            --output_file "${final_cache_file}" 
            # Add --no_delete here if you want to keep temporary files
            # --no_delete 
        
        if [ $? -ne 0 ]; then
             echo "Error during merge script for ${topic} - ${discussion_type}"
        fi
    done
done
echo "Finished merging temporary idea files."

###########################################################

# Deduplication

for topic in "${topic_names[@]}"; do
    for discussion_type in "${discussion_types[@]}"; do
        # Limit concurrent jobs - improved job control
        while (( $(jobs -r | wc -l) >= MAX_PARALLEL_JOBS )); do
            wait -n 2>/dev/null || sleep 1
        done

        echo "Running analyze_ideas_semantic_similarity.py with cache_name: $discussion_type"
        uv run multiagent_research_ideator/src/analyze_ideas_semantic_similarity.py \
            --cache_dir "$seed_ideas_cache_dir" \
            --cache_name "${topic}_${discussion_type}" \
            --save_similarity_matrix &
    done
done

wait

for topic in "${topic_names[@]}"; do
    for discussion_type in "${discussion_types[@]}"; do
        # Limit concurrent jobs - improved job control
        while (( $(jobs -r | wc -l) >= MAX_PARALLEL_JOBS )); do
            wait -n 2>/dev/null || sleep 1
        done
        
        echo "Running dedup_ideas.py with cache_name: $discussion_type"
        uv run multiagent_research_ideator/src/dedup_ideas.py \
            --cache_dir "$seed_ideas_cache_dir" \
            --cache_name "${topic}_${discussion_type}" \
            --dedup_cache_dir "${idea_dedup_cache_dir}" \
            --similarity_threshold 0.8 &
    done
done

wait


###########################################################

# Proposal Generation

for topic in "${topic_names[@]}"; do    
    for discussion_type in "${discussion_types[@]}"; do
        # Limit concurrent jobs - improved job control
        while (( $(jobs -r | wc -l) >= MAX_PARALLEL_JOBS )); do
            wait -n 2>/dev/null || sleep 1
        done

        # delete all files in the cache_dir
        rm -rf "${project_proposal_cache_dir}/${topic}_${discussion_type}/*"
        
        echo "Running experiment_plan_gen.py with cache_name: $discussion_type"
        uv run multiagent_research_ideator/src/experiment_plan_gen.py \
            --engine $llm_engine \
            --idea_dedup_cache_dir "${idea_dedup_cache_dir}" \
            --cache_name "${topic}_${discussion_type}" \
            --experiment_plan_cache_dir "${project_proposal_cache_dir}" \
            --idea_name "all" \
            --seed $seed \
            --method "prompting" &
    done
done

wait