from datetime import datetime


def generate_run_name(base_name: str) -> str:
    return f"{base_name}_{datetime.now():%Y%m%d_%H%M%S}"


def sanitize(value: str) -> str:
    return value.replace(" ", "_").replace("/", "-")
