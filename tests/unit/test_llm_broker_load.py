import asyncio
import time


async def test_high_load_parallel_requests(service, mock_llm_response):
    """Many parallel generate() calls complete successfully."""
    num_requests = 200
    prompts = [f"request_{i}" for i in range(num_requests)]
    for prompt in prompts:
        mock_llm_response[prompt] = f"Response for {prompt}"

    start_time = time.perf_counter()
    tasks = [service.generate(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    end_time = time.perf_counter()

    assert len(results) == num_requests
    success_count = sum(1 for r in results if not isinstance(r, Exception))
    assert success_count == num_requests

    health = service.pool.get_health_status()
    assert health.status == "healthy"
    assert health.alive_actors == health.total_actors

    duration = end_time - start_time
    rps = num_requests / duration if duration > 0 else 0
    print(f"\n[PERF] Parallel requests: {num_requests} requests in {duration:.2f}s = {rps:.2f} RPS")


async def test_high_load_batch_requests(service, mock_llm_response):
    """Large generate_batch run completes successfully."""
    num_requests = 150
    requests = [(f"request_{i}", None) for i in range(num_requests)]
    for prompt, _ in requests:
        mock_llm_response[prompt] = f"Response for {prompt}"

    start_time = time.perf_counter()
    results = await service.generate_batch(requests)
    end_time = time.perf_counter()

    assert len(results) == num_requests
    success_count = sum(1 for r in results if not isinstance(r, Exception))
    assert success_count == num_requests

    health = service.pool.get_health_status()
    assert health.status == "healthy"
    assert health.alive_actors == health.total_actors

    duration = end_time - start_time
    rps = num_requests / duration if duration > 0 else 0
    print(f"\n[PERF] Batch requests: {num_requests} requests in {duration:.2f}s = {rps:.2f} RPS")


async def test_high_load_mixed_with_errors(service, mock_llm_response):
    """High load with some failing prompts returns mixed results."""
    num_requests = 100
    prompts = [f"request_{i}" for i in range(num_requests)]
    error_indices = [10, 25, 50, 75]

    for i, prompt in enumerate(prompts):
        if i in error_indices:
            mock_llm_response[prompt] = Exception(f"Error for {prompt}")
        else:
            mock_llm_response[prompt] = f"Response for {prompt}"

    start_time = time.perf_counter()
    tasks = [service.generate(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    end_time = time.perf_counter()

    assert len(results) == num_requests
    success_count = sum(1 for r in results if not isinstance(r, Exception))
    error_count = sum(1 for r in results if isinstance(r, Exception))
    assert success_count == num_requests - len(error_indices)
    assert error_count == len(error_indices)

    health = service.pool.get_health_status()
    assert health.status in ("healthy", "degraded")

    duration = end_time - start_time
    rps = num_requests / duration if duration > 0 else 0
    print(
        f"\n[PERF] Mixed with errors: {num_requests} requests ({success_count} success, {error_count} errors) in {duration:.2f}s = {rps:.2f} RPS"
    )
