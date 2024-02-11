import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import local
from typing import Any, Dict, List

import openai
from loguru import logger
from tqdm import tqdm


class GPTProcessor:
    """Class of GPT processor."""

    def __init__(
        self,
        model: str,
        secondary_model: str | None,
        timeout_retry_sleep: int = 10,
        rate_limit_retry_sleep: int = 30,
        max_retry: int = 5,
        **kwargs: Any,
    ) -> None:
        """Initialize processor."""
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.model = model
        self.secondary_model = secondary_model
        self.timeout_retry_sleep = timeout_retry_sleep
        self.rate_limit_retry_sleep = rate_limit_retry_sleep
        self.max_retry = max_retry
        # to keep track of total tries for each thread
        self.thread_local_data = local()
        self.kwargs = kwargs
        # update kwargs
        if "seed" not in self.kwargs:
            self.kwargs["seed"] = 0
        if "request_timeout" not in self.kwargs:
            self.kwargs["request_timeout"] = 40

    def get_response(self, message: List[Dict[str, str]], request_id: str = "") -> Dict[str, Any]:
        """
        Attempt to get a response from ChatGPT.

        Tries with primary model first and if the context is too large moves to secondary_model.
        Handles Timeout and RateLimit exceptions with user defined parameters.
        :param message: message to be passed to ChatGPT
        :param request_id: optional request id for logging and debugging
        """
        if not hasattr(self.thread_local_data, "retry_count"):
            self.thread_local_data.retry_count = 0
        try:
            out: Dict[str, Any] = openai.ChatCompletion.create(
                model=self.model, messages=message, **self.kwargs
            ).to_dict()
            self.thread_local_data.retry_count = 0
            return out
        except openai.error.InvalidRequestError as e:
            if self.secondary_model is not None:
                logger.info(f"Failed with {self.model} trying with {self.secondary_model}: {request_id}")
                out = openai.ChatCompletion.create(
                    model=self.secondary_model, messages=message, **self.kwargs
                ).to_dict()
                self.thread_local_data.retry_count = 0
                return out
            else:
                raise Exception(f"{e}: {request_id}")
        except openai.error.Timeout as e:
            return self._submit_retry(message, request_id, sleep=self.timeout_retry_sleep, error=e)
        except (openai.error.RateLimitError, openai.error.APIError) as e:
            return self._submit_retry(message, request_id, sleep=self.rate_limit_retry_sleep, error=e)
        except Exception as e:
            raise Exception(f"{e}: {request_id}")

    def _submit_retry(
        self, message: List[Dict[str, str]], request_id: str, sleep: int, error: Exception
    ) -> Dict[str, Any]:
        """Retry while keeping count and raising info."""
        if self.thread_local_data.retry_count < self.max_retry:
            logger.info(
                f"OpenAI: {error} || Re-trying after {sleep} seconds "
                f"{self.thread_local_data.retry_count} of {self.max_retry}: {request_id}"
            )
            time.sleep(sleep)
            self.thread_local_data.retry_count += 1
            return self.get_response(message, request_id)
        else:
            raise Exception(f"{error} Max retry count reached: {request_id}")

    def bulk_process(
        self,
        message_list: List[List[Dict[str, str]]],
        request_names: str | List[str],
        num_workers: int,
        return_dict: bool = False,
        save_dir: str | None = "gpt_results",
        overwrite_results: bool = False,
    ) -> Dict[str, Dict[str, Any]] | None:
        """
        Process a list of messages with multi-threading (when num_workers > 0).

        :param message_list: List of messages/tasks to be passed to ChatGPT
        :param request_names: List of unique names for each task (required for output)
        :param num_workers: number of multi-thread workers. Set to 0 to not use ThreadPoolExecutor
        :param return_dict: whether to return a dict as output where request_names be the keys
        :param save_dir: path to save folder where request_names will form the file_name.pkl
                        (if None no files will be saved)
        :param overwrite_results: whether to overwrite existing result files.
        :return: Dict[response] or None
        """
        assert len(message_list) == len(set(request_names)), "each message must be associated with a unique name."
        assert not (not return_dict and save_dir is None), "at least a save_dir or return_dict==True is required."

        if save_dir is not None:
            assert "." not in save_dir, "save_dir must be a valid dir path without '.' in it"
            assert (
                not any([os.path.isfile(f"{save_dir}/{i}.pkl") for i in request_names]) or overwrite_results
            ), f"some expected files are already present in {save_dir}. Change save_dir or set overwrite_results==True"
            os.makedirs(save_dir, exist_ok=True)

        out = {}
        tasks = zip(message_list, request_names)
        if num_workers > 0:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                logger.info(f"submitting {len(message_list)} tasks to {num_workers} workers.")
                futures = {executor.submit(self.get_response, m, f"{n}_{i}"): n for i, (m, n) in enumerate(tasks)}
                for future in tqdm(as_completed(futures), total=len(futures)):
                    result_name = futures[future]
                    try:
                        result = future.result()
                        if return_dict:
                            out[result_name] = result
                        if save_dir is not None:
                            with open(f"{save_dir}/{result_name}.pkl", "wb") as f:
                                pickle.dump(result, f)
                    except Exception as exc:
                        for f in futures:  # type: ignore[assignment]
                            f.cancel()  # type: ignore[attr-defined]
                        self._dump_partial_results(out, save_dir)
                        raise Exception from exc
        else:
            for m, n in tqdm(tasks, total=len(message_list)):
                try:
                    res = self.get_response(m, n)
                    if return_dict:
                        out[n] = res
                    if save_dir is not None:
                        with open(f"{save_dir}/{n}.pkl", "wb") as f:
                            pickle.dump(res, f)
                except Exception as exc:
                    self._dump_partial_results(out, save_dir)
                    raise Exception from exc
        if return_dict:
            return out
        else:
            return None

    @staticmethod
    def _dump_partial_results(out: Dict[str, Dict[str, Any]], save_dir: str | None) -> None:
        """Save all partial results in case of exception."""
        if len(out) > 0:
            dump_file = "result_dump.pkl" if save_dir is None else f"{save_dir}/result_dump.pkl"
            with open(dump_file, "wb") as f:
                pickle.dump(out, f)
            logger.info(f"partial results dumped in {dump_file}")
