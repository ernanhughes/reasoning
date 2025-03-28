from datetime import datetime, timezone
from reasoning.logs import PromptLog


def test_prompt_log():
    PromptLog(
        prompt_id="abc123",
        model="tinyllama",
        dataset="openorca",
        text="What is reasoning?",
        tokens=[1, 2, 3],
        created_at=datetime.now(timezone.utc).isoformat()
    ).save()

    pl = PromptLog(
        prompt_id="abc123",
        model="tinyllama",
        dataset="openorca",
        text="What is reasoning?",
        tokens=[1, 2, 3],
        created_at=datetime.now(timezone.utc).isoformat()
    )
    pl.use_db = True
    pl.save()


if __name__ == "__main__":
    test_prompt_log()
