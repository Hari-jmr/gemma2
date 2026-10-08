"""List official Kaggle variants or download an explicitly selected version."""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("KAGGLEHUB_CACHE", str(ROOT / "models" / "kaggle-cache"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--handle", help="google/embeddinggemma-2/framework/variant/version")
    args = parser.parse_args()
    if args.handle:
        from kagglehub import model_download
        from kagglehub.handle import parse_model_handle
        handle = parse_model_handle(args.handle)
        if handle.owner != "google" or handle.model != "embeddinggemma-2" or not handle.is_versioned():
            parser.error("Select a versioned variant of google/embeddinggemma-2.")
        path = model_download(args.handle)
        candidates = [p.parent for p in Path(path).rglob("modules.json")]
        if len(candidates) != 1:
            raise SystemExit("Downloaded variant does not contain one sentence-transformers model; inspect its documented format.")
        print("EMBEDDING_MODEL_PATH=" + str(candidates[0]))
        return
    from kagglehub.clients import build_kaggle_client
    from kagglesdk.models.types.model_api_service import ApiListModelInstancesRequest
    request = ApiListModelInstancesRequest()
    request.owner_slug = "google"
    request.model_slug = "embeddinggemma-2"
    with build_kaggle_client() as client:
        while True:
            response = client.models.model_api_client.list_model_instances(request)
            for instance in response.instances:
                print(instance.framework.name, instance.slug, "version", instance.version_number)
            if not response.next_page_token:
                break
            request.page_token = response.next_page_token


if __name__ == "__main__":
    main()
