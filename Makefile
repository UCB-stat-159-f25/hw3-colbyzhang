.ONESHELL:
SHELL = /bin/bash

## env: Create or update the conda environment
.PHONY: env
env:
	@echo "Creating or updating environment..."
	if conda env list | grep -q "ligotools-env"; then \
		echo "Updating existing environment..."; \
		conda env update -f environment.yml --prune; \
	else \
		echo "Creating new environment..."; \
		conda env create -f environment.yml; \
	fi
	@echo "Done! Activate with: conda activate ligotools-env"

## html: Build the MyST HTML site
.PHONY: html
html:
	@echo "Building MyST HTML site..."
	myst build --html
	@echo "Build complete! You can view your site in _build/html/"

## clean: Remove generated figures, audio, and build directories
.PHONY: clean
clean:
	@echo "Cleaning up generated files..."
	rm -rf figures audio _build
	@echo "Cleanup complete!"