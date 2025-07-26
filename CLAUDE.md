# Development Guidelines

## Error Handling Philosophy

**Prefer early failures over silent incorrect behavior.**

When writing code, especially utility functions that handle data transformations:
- Always try to perform the expected operation directly
- Let exceptions propagate rather than silently falling back to potentially incorrect behavior
- Avoid "helpful" fallback cases that mask bugs
- Clear error messages are better than working-but-wrong code

This principle helps catch bugs early and makes debugging much easier than tracking down silent data corruption or incorrect transformations.