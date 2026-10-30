"""#344 反思积压：同一次后台任务内按切分片段连续消费。"""

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from astrbot.api.platform import MessageType

from tests.event_handler_support import _make_real_handler


def _day(day: int, hour: int, minute: int = 0) -> float:
    return datetime(2026, 10, day, hour, minute).timestamp()


async def _seed_messages(manager, session_id: str, timestamps: list[float]) -> None:
    for index, timestamp in enumerate(timestamps):
        await manager.add_message(
            session_id=session_id,
            role="user" if index % 2 == 0 else "assistant",
            content=f"message-{index}",
            sender_id="u1" if index % 2 == 0 else "bot1",
            group_id="group-1",
            timestamp=timestamp,
        )


def _configure_processor(handler, *, side_effect=None):
    processor = handler.memory_processor

    async def _summary(*args, **kwargs):
        return "summary", {"topics": ["t1"]}, 0.6

    processor.process_conversation = AsyncMock(
        side_effect=side_effect or _summary
    )
    processor.classify_atoms_from_metadata = Mock(return_value=[])


async def _build_plan(handler, session_id: str, total_messages: int):
    return await handler._memory_reflection._build_reflection_plan_for_cursor(
        session_id,
        total_messages,
        0,
        None,
        None,
        "persona-1",
        session_id,
    )


async def _run_scheduled(handler, plan) -> None:
    await handler._memory_reflection._schedule_reflection(plan)
    await asyncio.gather(*list(handler._storage_tasks))


def _passive_group_event(session_id: str, message_id: str):
    from astrbot.core.message.components import Plain

    event = Mock()
    event.unified_msg_origin = session_id
    event.message_obj = SimpleNamespace(
        message_id=message_id,
        timestamp=_day(26, 0, 20),
        sender=SimpleNamespace(user_id="user-2", nickname="Member 2"),
    )
    event.get_message_type.return_value = MessageType.GROUP_MESSAGE
    event.get_sender_id.return_value = "user-2"
    event.get_self_id.return_value = "bot-1"
    event.get_sender_name.return_value = "Member 2"
    event.get_platform_name.return_value = "test"
    event.get_group_id.return_value = session_id
    event.get_messages.return_value = [Plain("a passive group message")]
    event.get_message_str.return_value = "a passive group message"
    event.get_extra.side_effect = lambda key, default=None: default
    return event


@pytest.mark.asyncio
async def test_backlog_consumes_all_day_segments_in_one_run(tmp_path):
    handler, manager, store = await _make_real_handler(tmp_path)
    session_id = "test:group:backlog"
    timestamps = [
        _day(25, 22, 0),
        _day(25, 22, 5),
        _day(25, 22, 10),
        _day(25, 22, 15),
        _day(26, 0, 1),
        _day(26, 0, 5),
        _day(26, 0, 10),
        _day(26, 0, 15),
    ]
    await _seed_messages(manager, session_id, timestamps)
    _configure_processor(handler)
    handler.memory_engine.find_similar_memory = AsyncMock(return_value=None)
    handler.memory_engine.add_memory = AsyncMock(return_value=1)

    try:
        plan = await _build_plan(handler, session_id, len(timestamps))
        assert (plan.start_index, plan.end_index) == (0, 4)

        await _run_scheduled(handler, plan)

        assert handler.memory_processor.process_conversation.await_count == 2
        assert await manager.get_session_metadata(
            session_id, "last_summarized_index", 0
        ) == 8
        assert await manager.get_session_metadata(session_id, "pending_summary", None) is None
        windows = [
            call.kwargs["metadata"]["source_window"]
            for call in handler.memory_engine.add_memory.await_args_list
        ]
        assert [
            (window["start_index"], window["end_index"]) for window in windows
        ] == [(0, 4), (4, 8)]
    finally:
        await handler.shutdown()
        await store.close()


@pytest.mark.asyncio
async def test_backlog_respects_segment_budget(tmp_path):
    handler, manager, store = await _make_real_handler(tmp_path)
    session_id = "test:group:budget"
    await _seed_messages(
        manager,
        session_id,
        [
            _day(25, 22, 0),
            _day(25, 22, 5),
            _day(25, 22, 10),
            _day(25, 22, 15),
            _day(26, 0, 1),
            _day(26, 0, 5),
            _day(26, 0, 10),
            _day(26, 0, 15),
        ],
    )
    handler.config_manager._config["reflection_engine"]["max_segments_per_run"] = 1
    _configure_processor(handler)
    handler.memory_engine.find_similar_memory = AsyncMock(return_value=None)
    handler.memory_engine.add_memory = AsyncMock(return_value=1)

    try:
        plan = await _build_plan(handler, session_id, 8)
        await _run_scheduled(handler, plan)

        assert handler.memory_processor.process_conversation.await_count == 1
        assert await manager.get_session_metadata(
            session_id, "last_summarized_index", 0
        ) == 4
    finally:
        await handler.shutdown()
        await store.close()


@pytest.mark.asyncio
async def test_backlog_stops_when_segment_fails(tmp_path):
    handler, manager, store = await _make_real_handler(tmp_path)
    session_id = "test:group:failure"
    await _seed_messages(
        manager,
        session_id,
        [
            _day(25, 22, 0),
            _day(25, 22, 5),
            _day(25, 22, 10),
            _day(25, 22, 15),
            _day(26, 0, 1),
            _day(26, 0, 5),
            _day(26, 0, 10),
            _day(26, 0, 15),
        ],
    )
    _configure_processor(handler, side_effect=RuntimeError("llm failed"))
    handler.memory_engine.add_memory = AsyncMock(return_value=1)

    try:
        plan = await _build_plan(handler, session_id, 8)
        await _run_scheduled(handler, plan)

        assert handler.memory_processor.process_conversation.await_count == 1
        assert await manager.get_session_metadata(
            session_id, "last_summarized_index", 0
        ) == 0
        assert await manager.get_session_metadata(
            session_id, "pending_summary", None
        ) == {"start_index": 0, "end_index": 4, "retry_count": 1}
        handler.memory_engine.add_memory.assert_not_awaited()
    finally:
        await handler.shutdown()
        await store.close()


@pytest.mark.asyncio
async def test_passive_group_capture_summarizes_without_bot_reply(tmp_path):
    handler, manager, store = await _make_real_handler(tmp_path)
    session_id = "test:group:passive-summary"
    handler.config_manager._config["reflection_engine"][
        "summary_trigger_rounds"
    ] = 1
    await _seed_messages(
        manager,
        session_id,
        [_day(26, 0, 0), _day(26, 0, 5)],
    )
    _configure_processor(handler)
    handler.memory_engine.find_similar_memory = AsyncMock(return_value=None)
    handler.memory_engine.add_memory = AsyncMock(return_value=1)

    try:
        await handler.handle_all_group_messages(
            _passive_group_event(session_id, "passive-1")
        )
        await asyncio.gather(*list(handler._storage_tasks))

        assert handler.memory_processor.process_conversation.await_count == 1
        assert await manager.get_session_metadata(
            session_id, "last_summarized_index", 0
        ) == 3
    finally:
        await handler.shutdown()
        await store.close()


@pytest.mark.asyncio
async def test_group_summary_threshold_counts_member_messages(tmp_path):
    handler, manager, store = await _make_real_handler(tmp_path)
    session_id = "test:group:member-message-threshold"
    handler.config_manager._config["reflection_engine"][
        "summary_trigger_rounds"
    ] = 2
    _configure_processor(handler)
    handler.memory_engine.find_similar_memory = AsyncMock(return_value=None)
    handler.memory_engine.add_memory = AsyncMock(return_value=1)

    try:
        await handler.handle_all_group_messages(
            _passive_group_event(session_id, "passive-threshold-1")
        )
        assert not handler._storage_tasks

        await handler.handle_all_group_messages(
            _passive_group_event(session_id, "passive-threshold-2")
        )
        await asyncio.gather(*list(handler._storage_tasks))

        assert handler.memory_processor.process_conversation.await_count == 1
        assert await manager.get_session_metadata(
            session_id, "last_summarized_index", 0
        ) == 2
    finally:
        await handler.shutdown()
        await store.close()


@pytest.mark.asyncio
async def test_failed_group_summary_keeps_messages_and_cursor(tmp_path):
    handler, manager, store = await _make_real_handler(tmp_path)
    session_id = "test:group:summary-retry-cooldown"
    handler.config_manager._config["reflection_engine"][
        "summary_trigger_rounds"
    ] = 1
    await _seed_messages(
        manager,
        session_id,
        [_day(26, 0, 0), _day(26, 0, 5)],
    )
    pending = {
        "start_index": 0,
        "end_index": 2,
        "retry_count": 3,
        "retry_after": 4_000_000_000,
    }
    await manager.update_session_metadata(session_id, "pending_summary", pending)

    try:
        await handler.handle_all_group_messages(
            _passive_group_event(session_id, "passive-2")
        )

        assert not handler._storage_tasks
        assert await manager.get_session_metadata(
            session_id, "last_summarized_index", 0
        ) == 0
        assert await manager.get_session_metadata(
            session_id, "pending_summary", None
        ) == pending
        assert await manager.store.get_message_count(session_id) == 3
    finally:
        await handler.shutdown()
        await store.close()
