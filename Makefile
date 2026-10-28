.PHONY: install run debug clean lint data_search_doc moulinette_doc data_search_code moulinette_code index answer answer_code answer_doc evaluate_code

Q ?= How do I configure the KV cache in vLLM?

install:
	uv sync

run:
	uv run python -m src

index:
	uv run python -m src index --max_chunk_size 2000

data_search_code:
	uv run python -m src search_dataset --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json --k 5 --save_directory data/output/search_results/UnansweredQuestions

data_search_doc:
	uv run python -m src search_dataset --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json --k 5 --save_directory data/output/search_results/UnansweredQuestions

moulinette_doc:
	uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 5 --save_directory data/output/search_results/UnansweredQuestions 
	./moulinette/moulinette-ubuntu evaluate_student_search_results data/output/search_results/UnansweredQuestions/dataset_docs_public.json data/datasets/AnsweredQuestions/dataset_docs_public.json --k 5 --max_context_length 2000

moulinette_code:
	uv run python -m src search_dataset --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json --k 5 --save_directory data/output/search_results/UnansweredQuestions
	./moulinette/moulinette-ubuntu evaluate_student_search_results data/output/search_results/UnansweredQuestions/dataset_code_public.json data/datasets/AnsweredQuestions/dataset_code_public.json --k 5 --max_context_length 2000

answer:
	uv run python -m src answer --query "$(Q)" --k 5

answer_code: data_search_code
	uv run python -m src answer_dataset --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json --save_directory data/output/search_results_and_answer

answer_doc: data_search_doc
	uv run python -m src answer_dataset --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json --save_directory data/output/search_results_and_answer

evaluate_code:
	uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json

evaluate_doc:
	uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
