.PHONY: install run debug clean lint data_search_doc moulinette_doc

install:
	uv sync

run:
	uv run python -m src

data_search_doc:
	uv run python -m src search_dataset --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json --k 5 --save_directory data/output/search_results/UnansweredQuestions

moulinette_doc:
	uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json --k 5 --save_directory data/output/search_results/UnansweredQuestions 
	./moulinette/moulinette-ubuntu evaluate_student_search_results data/output/search_results/UnansweredQuestions/dataset_docs_public.json data/datasets/AnsweredQuestions/dataset_docs_public.json --k 5 --max_context_length 2000
