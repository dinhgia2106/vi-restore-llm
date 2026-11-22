"""Prompt format shared by fine-tuning and inference, so they never drift apart."""

SYSTEM = (
    "Bạn là bộ gõ tiếng Việt. Thêm dấu và sửa lỗi gõ cho câu người dùng nhập. "
    "Chỉ trả về câu đã sửa, giữ nguyên ý và dấu câu."
)


def messages(source: str, target: str | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": source}]
    if target is not None:
        msgs.append({"role": "assistant", "content": target})
    return msgs
