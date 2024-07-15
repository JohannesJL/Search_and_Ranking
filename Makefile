SHELL 			    := /bin/bash -o pipefail
.DEFAULT_GOAL 	    := help


##@ setup
.PHONY: setup
setup:  # install dependencies
	pip install -r requirements.txt -r requirements_dev.txt

##@ format code
.PHONY: format-code
format-code:  # perform code formatting
	isort src
	black src
