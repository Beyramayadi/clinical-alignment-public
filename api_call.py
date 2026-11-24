import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request


def load_dotenv_file() -> bool:
	"""Load key=value pairs from a local .env into os.environ if not already set."""
	candidates = [Path.cwd() / ".env", Path(__file__).resolve().parent / ".env"]
	dotenv_path = next((p for p in candidates if p.is_file()), None)
	if not dotenv_path:
		return False

	for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
		line = raw_line.strip()
		if not line or line.startswith("#"):
			continue
		if line.startswith("export "):
			line = line[len("export "):].strip()
		if "=" not in line:
			continue

		key, value = line.split("=", 1)
		key = key.strip()
		value = value.strip()
		if not key:
			continue
		if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
			value = value[1:-1]

		os.environ.setdefault(key, value)

	return True


def choose_small_model_ids(text_models):
	"""Return text-capable model IDs sorted from smallest/cheapest-looking first."""
	ids = [str(m.get("id", "")) for m in text_models if m.get("id")]

	def score(model_id: str):
		name = model_id.lower()
		if "nano" in name:
			return (0, name)
		if "mini" in name:
			return (1, name)
		if "gpt-35-turbo" in name:
			return (2, name)
		if "turbo" in name:
			return (3, name)
		return (9, name)

	return sorted(ids, key=score)


def choose_large_gpt5_model_ids(text_models):
	"""Return GPT-5 family model IDs sorted to prefer larger non-mini/non-nano variants first."""
	ids = [
		str(m.get("id", ""))
		for m in text_models
		if str(m.get("id", "")).startswith("gpt-5")
	]

	def score(model_id: str):
		name = model_id.lower()
		if "nano" in name:
			return (4, name)
		if "mini" in name:
			return (3, name)
		if "lite" in name:
			return (2, name)
		if "chat" in name:
			return (1, name)
		return (0, name)

	return sorted(ids, key=score)


def extract_text_from_response(payload):
	"""Extract plain text from Responses API or Chat Completions API payloads."""
	output_text = payload.get("output_text")
	if isinstance(output_text, str) and output_text.strip():
		return output_text.strip()

	if isinstance(payload.get("output"), list):
		parts = []
		for item in payload["output"]:
			if not isinstance(item, dict):
				continue
			for content_item in item.get("content", []):
				if not isinstance(content_item, dict):
					continue
				if content_item.get("type") in ("output_text", "text") and content_item.get("text"):
					parts.append(str(content_item["text"]))
		if parts:
			return "\n".join(parts).strip()

	choices = payload.get("choices")
	if isinstance(choices, list) and choices:
		message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
		content = message.get("content") if isinstance(message, dict) else None
		if isinstance(content, str) and content.strip():
			return content.strip()
		if isinstance(content, list):
			parts = []
			for item in content:
				if isinstance(item, dict) and item.get("text"):
					parts.append(str(item["text"]))
			if parts:
				return "\n".join(parts).strip()

	return None


	def describe_missing_text(payload):
		"""Return a short diagnostic message when a response succeeds without extractable text."""
		status = payload.get("status")
		incomplete = payload.get("incomplete_details")
		output = payload.get("output")
		output_types = []
		if isinstance(output, list):
			output_types = [item.get("type", "<unknown>") for item in output if isinstance(item, dict)]

		parts = []
		if status:
			parts.append(f"status={status}")
		if incomplete:
			parts.append(f"incomplete_details={json.dumps(incomplete, ensure_ascii=False)}")
		if output_types:
			parts.append(f"output_types={output_types}")

		return "; ".join(parts) if parts else "No extractable text fields were present in the payload."


def send_test_prompt(endpoint: str, api_key: str, model_id: str) -> bool:
	"""Send one short test prompt and print the model response."""
	test_prompt = "Say hello in one short sentence."
	url = f"{endpoint}/openai/v1/responses"
	payload = {
		"model": model_id,
		"input": test_prompt,
		"max_output_tokens": 300,
		"reasoning": {"effort": "minimal"},
		"text": {
			"format": {"type": "text"},
			"verbosity": "low",
		},
	}

	request = urllib.request.Request(
		url,
		headers={
			"api-key": api_key,
			"Content-Type": "application/json",
		},
		data=json.dumps(payload).encode("utf-8"),
		method="POST",
	)

	try:
		with urllib.request.urlopen(request, timeout=45) as response:
			body = response.read().decode("utf-8")
			parsed = json.loads(body)
			text = extract_text_from_response(parsed)
			print(f"Test prompt model: {model_id}")
			print(f"Test prompt: {test_prompt}")
			if text:
				print("Test response:", text)
			else:
				print("Test response text not found in payload.")
				print(describe_missing_text(parsed))
				return False
			return True
	except urllib.error.HTTPError as exc:
		error_body = exc.read().decode("utf-8", errors="replace")
		print(f"Test prompt failed for model {model_id} with HTTP {exc.code}: {exc.reason}")
		print(error_body[:800])
		return False
	except Exception as exc:
		print(f"Test prompt failed for model {model_id}: {exc}")
		return False


def test_models(endpoint: str, api_key: str, model_ids, label: str, max_attempts: int):
	"""Try a sequence of model IDs and return the first one that succeeds."""
	if not model_ids:
		print(f"No {label} model IDs found for running a test prompt.")
		return None

	print(f"Running a simple test prompt on {label} models...")
	for model_id in model_ids[:max_attempts]:
		if send_test_prompt(endpoint, api_key, model_id):
			print(f"First working {label} model: {model_id}")
			return model_id

	print(f"Could not complete test prompt on the first {max_attempts} {label} model candidates.")
	return None


def main():
	load_dotenv_file()

	api_key = (
		os.environ.get("AZURE_OPENAI_API_KEY")
		or os.environ.get("API_KEY")
	)
	endpoint = (
		os.environ.get("AZURE_OPENAI_ENDPOINT")
		or os.environ.get("API_ENDPOINT")
	)

	if not api_key or not endpoint:
		print("Missing AZURE_OPENAI_API_KEY/API_KEY or AZURE_OPENAI_ENDPOINT/API_ENDPOINT.")
		return 1

	endpoint = endpoint.rstrip("/")
	url = f"{endpoint}/openai/v1/models"

	request = urllib.request.Request(
		url,
		headers={
			"api-key": api_key,
			"Content-Type": "application/json",
		},
		method="GET",
	)

	try:
		with urllib.request.urlopen(request, timeout=30) as response:
			body = response.read().decode("utf-8")
			print(f"Request succeeded with HTTP {response.status}.")
			try:
				payload = json.loads(body)
				print("Response keys:", list(payload.keys()))
				if isinstance(payload.get("data"), list):
					models = payload["data"]
					text_models = [
						m for m in models
						if m.get("capabilities", {}).get("inference")
						and (
							m.get("capabilities", {}).get("chat_completion")
							or m.get("capabilities", {}).get("completion")
						)
					]
					gpt5_models = [m for m in text_models if str(m.get("id", "")).startswith("gpt-5")]

					print(f"Total models returned: {len(models)}")
					print(f"Text-capable models: {len(text_models)}")
					print("Text-capable model IDs:")
					for item in sorted(text_models, key=lambda x: str(x.get("id", ""))):
						print(f"- {item.get('id', '<unknown>')}")

					print(f"GPT-5 family models found: {len(gpt5_models)}")
					for item in sorted(gpt5_models, key=lambda x: str(x.get("id", ""))):
						print(f"- {item.get('id', '<unknown>')}")

					large_gpt5_candidates = choose_large_gpt5_model_ids(text_models)
					working_gpt5 = test_models(endpoint, api_key, large_gpt5_candidates, "GPT-5", 10)

					if not working_gpt5:
						small_candidates = choose_small_model_ids(text_models)
						test_models(endpoint, api_key, small_candidates, "small", 5)
				else:
					print(body[:1000])
			except json.JSONDecodeError:
				print(body[:1000])
		return 0
	except urllib.error.HTTPError as exc:
		error_body = exc.read().decode("utf-8", errors="replace")
		print(f"Request failed with HTTP {exc.code}: {exc.reason}")
		print(error_body)
		return 1
	except Exception as exc:
		print(f"Request failed: {exc}")
		return 1


if __name__ == "__main__":
	sys.exit(main())
