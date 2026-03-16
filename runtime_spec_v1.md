# Runtime Spec V1

## Purpose
Create a continuously running cognitive runtime that maintains identity, memory, and goals.

## Always-On Components
- Identity
- WorkingMemory
- EpisodicMemory
- SemanticMemory
- GoalRegistry
- AttentionState
- SelfModel
- EventQueue
- RuntimeState

## Event Types
- user_message
- timer_tick
- tool_result
- internal_reflection_trigger
- goal_deadline
- error_event
- memory_consolidation_trigger

## One Cognition Cycle
1. Ingest events
2. Update salience
3. Select focus
4. Run reasoning cycle
5. Decide action
6. Commit updates
7. Schedule next cycle