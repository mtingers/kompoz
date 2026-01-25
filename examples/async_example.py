"""
Example: Async Rules and Transforms

This example shows how to use async combinators for rules
that need to hit databases, APIs, or other async services.
"""

import asyncio
from dataclasses import dataclass, field

from kompoz import (
    AsyncRetry,
    async_pipe,
    async_pipe_args,
    async_rule,
    async_rule_args,
)


@dataclass
class User:
    id: int
    name: str
    roles: list[str] = field(default_factory=list)
    profile: dict = field(default_factory=dict)
    permissions: list[str] = field(default_factory=list)


# =============================================================================
# Simulated Async Services
# =============================================================================


class MockDatabase:
    """Simulated async database."""

    PERMISSIONS = {
        1: ["read", "write"],
        2: ["read"],
        3: ["read", "write", "admin"],
    }

    PROFILES = {
        1: {"tier": "premium", "verified": True},
        2: {"tier": "basic", "verified": False},
        3: {"tier": "enterprise", "verified": True},
    }

    @classmethod
    async def check_permission(cls, user_id: int, permission: str) -> bool:
        await asyncio.sleep(0.01)  # Simulate network delay
        perms = cls.PERMISSIONS.get(user_id, [])
        return permission in perms

    @classmethod
    async def get_profile(cls, user_id: int) -> dict:
        await asyncio.sleep(0.01)
        return cls.PROFILES.get(user_id, {})

    @classmethod
    async def get_roles(cls, user_id: int) -> list[str]:
        await asyncio.sleep(0.01)
        if user_id == 3:
            return ["admin", "user"]
        return ["user"]


# =============================================================================
# Async Rules (Predicates)
# =============================================================================


@async_rule
async def is_verified(user: User) -> bool:
    """Check if user is verified in database."""
    profile = await MockDatabase.get_profile(user.id)
    return profile.get("verified", False)


@async_rule
async def is_premium(user: User) -> bool:
    """Check if user has premium tier."""
    profile = await MockDatabase.get_profile(user.id)
    return profile.get("tier") in ("premium", "enterprise")


@async_rule_args
async def has_permission(user: User, permission: str) -> bool:
    """Check if user has specific permission."""
    return await MockDatabase.check_permission(user.id, permission)


@async_rule_args
async def has_role(user: User, role: str) -> bool:
    """Check if user has specific role."""
    roles = await MockDatabase.get_roles(user.id)
    return role in roles


# =============================================================================
# Async Transforms (Enrichment)
# =============================================================================


@async_pipe
async def load_profile(user: User) -> User:
    """Enrich user with profile data."""
    user.profile = await MockDatabase.get_profile(user.id)
    return user


@async_pipe
async def load_roles(user: User) -> User:
    """Enrich user with roles."""
    user.roles = await MockDatabase.get_roles(user.id)
    return user


@async_pipe_args
async def load_permissions(user: User, scope: str = "all") -> User:
    """Enrich user with permissions."""
    # Simplified - in real code would filter by scope
    user.permissions = MockDatabase.PERMISSIONS.get(user.id, [])
    return user


# =============================================================================
# Compose Async Rules
# =============================================================================

# Can write if verified AND has write permission
can_write = is_verified & has_permission("write")

# Can admin if premium AND has admin role
can_admin = is_premium & has_role("admin")

# Full enrichment pipeline
enrich_user = load_profile & load_roles & load_permissions("default")


# =============================================================================
# Async Retry Example
# =============================================================================


class FlakyService:
    """Simulates a service that sometimes fails."""

    call_count = 0

    @classmethod
    async def check(cls, user: User) -> bool:
        cls.call_count += 1
        await asyncio.sleep(0.01)
        # Fail first 2 attempts, succeed on 3rd
        if cls.call_count < 3:
            raise ConnectionError("Service unavailable")
        return True


@async_rule
async def flaky_check(user: User) -> bool:
    return await FlakyService.check(user)


# Retry with exponential backoff
resilient_check = AsyncRetry(
    flaky_check, max_attempts=5, backoff=0.01, exponential=True, jitter=0.005
)


# =============================================================================
# Run Examples
# =============================================================================


async def main():
    users = [
        User(1, "Alice"),
        User(2, "Bob"),
        User(3, "Charlie"),
    ]

    print("=" * 60)
    print("ASYNC PERMISSION CHECKS")
    print("=" * 60)
    print()

    for user in users:
        ok, _ = await can_write.run(user)
        status = "✓" if ok else "✗"
        print(f"{user.name}: {status} can_write")

    print()

    for user in users:
        ok, _ = await can_admin.run(user)
        status = "✓" if ok else "✗"
        print(f"{user.name}: {status} can_admin")

    print()
    print("=" * 60)
    print("ASYNC ENRICHMENT PIPELINE")
    print("=" * 60)
    print()

    for user in users:
        ok, enriched = await enrich_user.run(user)
        print(f"{enriched.name}:")
        print(f"  Profile: {enriched.profile}")
        print(f"  Roles: {enriched.roles}")
        print(f"  Permissions: {enriched.permissions}")
        print()

    print("=" * 60)
    print("ASYNC RETRY")
    print("=" * 60)
    print()

    FlakyService.call_count = 0
    user = User(1, "TestUser")

    try:
        ok, _ = await resilient_check.run(user)
        print(f"Result: {'✓ Success' if ok else '✗ Failed'}")
        print(f"Attempts: {FlakyService.call_count}")
    except Exception as e:
        print(f"Failed after retries: {e}")

    print()
    print("=" * 60)
    print("PARALLEL ASYNC CHECKS")
    print("=" * 60)
    print()

    # Run multiple checks in parallel
    user = User(1, "Alice")
    results = await asyncio.gather(
        is_verified.run(user),
        is_premium.run(user),
        has_permission("write").run(user),
        has_role("admin").run(user),
    )

    checks = ["is_verified", "is_premium", "has_permission(write)", "has_role(admin)"]
    for name, (ok, _) in zip(checks, results):
        status = "✓" if ok else "✗"
        print(f"  {status} {name}")


if __name__ == "__main__":
    asyncio.run(main())
