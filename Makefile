.PHONY: install run debug clean lint data_search

install:
	uv sync

run:
	uv run python -m src

data_search:
	uv run python -m src search_dataset --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json --k 5 --save_directory data/output/search_results/UnansweredQuestions