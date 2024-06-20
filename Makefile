.PHONY: test run

# use the new docker compose command if available or the legacy docker-compose command
DOCKER_COMPOSE := $(shell \
		docker compose version > /dev/null 2>&1; \
		if [ $$? -eq 0 ]; then \
				echo "docker compose"; \
		else \
				docker-compose version > /dev/null 2>&1; \
				if [ $$? -eq 0 ]; then \
						echo "docker-compose"; \
				fi; \
		fi; \
)

build:
				$(DOCKER_COMPOSE) build
run:
				$(DOCKER_COMPOSE) up