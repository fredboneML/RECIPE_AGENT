from functools import wraps
import time
import logging
from typing import Any, Callable, TypeVar, Dict
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http import exceptions as qdrant_exceptions

# Configure logging
logger = logging.getLogger(__name__)

# Circuit breaker implementation


class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: int = 3, recovery_timeout: int = 30):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF-OPEN

    def execute(self, func, *args, **kwargs):
        if self.state == "OPEN":
            # Check if recovery timeout has elapsed
            if time.time() - self.last_failure_time > self.recovery_timeout:
                logger.info(
                    f"Circuit {self.name} transitioning from OPEN to HALF-OPEN")
                self.state = "HALF-OPEN"
            else:
                logger.warning(f"Circuit {self.name} is OPEN - fast failing")
                raise Exception(f"Circuit {self.name} is open")

        try:
            result = func(*args, **kwargs)

            # If successful and in HALF-OPEN, close the circuit
            if self.state == "HALF-OPEN":
                logger.info(
                    f"Circuit {self.name} recovered - transitioning to CLOSED")
                self.reset()

            return result

        except Exception as e:
            self.record_failure()

            # Check if we should open the circuit
            if self.state == "CLOSED" and self.failure_count >= self.failure_threshold:
                logger.warning(
                    f"Circuit {self.name} transitioning to OPEN after {self.failure_count} failures")
                self.state = "OPEN"
                self.last_failure_time = time.time()

            raise e

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()

    def reset(self):
        self.failure_count = 0
        self.state = "CLOSED"


# Create circuit breakers for different operations
qdrant_search_breaker = CircuitBreaker("qdrant_search")
qdrant_connection_breaker = CircuitBreaker("qdrant_connection")

# Function decorators for resilience
F = TypeVar('F', bound=Callable[..., Any])


def with_circuit_breaker(breaker: CircuitBreaker):
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return breaker.execute(func, *args, **kwargs)
        return wrapper
    return decorator


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    # Catch all Qdrant-related exceptions
    retry=retry_if_exception_type((UnexpectedResponse, Exception))
)
@with_circuit_breaker(qdrant_search_breaker)
def search_qdrant_safely(client, collection_name, query_vector, **kwargs):
    """Execute Qdrant search with circuit breaker and retry logic"""
    # Ensure timeout is set and is an integer
    if 'timeout' not in kwargs:
        kwargs['timeout'] = 30  # Increased from 5s to 30s for large databases (600K recipes)
    else:
        kwargs['timeout'] = int(kwargs['timeout'])

    try:
        # Get collection info to determine vector name
        collection_info = client.get_collection(collection_name)
        vector_names = list(collection_info.config.params.vectors.keys())
        if not vector_names:
            raise ValueError(
                f"No vector names found in collection {collection_name}")
        vector_name = vector_names[0]  # Use the first vector name

        return client.search(
            collection_name=collection_name,
            query_vector=(vector_name, query_vector),  # Specify vector name
            **kwargs
        )
    except Exception as e:
        logger.error(f"Qdrant search error: {e}")
        raise
