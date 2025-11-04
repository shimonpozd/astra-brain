from __future__ import annotations

import asyncio
import getpass
from argparse import ArgumentParser, Namespace

from brain_service.core.database import create_engine, create_session_factory
from brain_service.core.settings import Settings
from brain_service.models.db import Base
from brain_service.services.user_service import (
    UserAlreadyExistsError,
    UserNotFoundError,
    UserService,
)


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description=(
            "Create or update a user record in the brain_service database. "
            "Run this inside the virtualenv: "
            "python -m brain_service.scripts.create_user --username admin --role admin"
        )
    )
    parser.add_argument("--username", required=True, help="Логин пользователя")
    parser.add_argument("--password", help="Пароль (если не указан, спросим интерактивно)")
    parser.add_argument(
        "--role",
        choices=("admin", "member"),
        default="member",
        help="Роль пользователя (admin или member)",
    )
    parser.add_argument(
        "--inactive",
        action="store_true",
        help="Создать пользователя неактивным (по умолчанию активный).",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Если пользователь существует — обновить пароль/роль/статус.",
    )
    return parser.parse_args()


async def prepare_user_service(
    username: str,
    password: str,
    role: str,
    is_active: bool,
) -> tuple:
    settings = Settings()
    engine = create_engine(settings.DATABASE_URL)

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = create_session_factory(engine)
    service = UserService(session_factory, encryption_secret=settings.API_KEY_SECRET)

    return engine, service


def main() -> None:
    args = parse_args()
    password = args.password
    if not password:
        password = getpass.getpass("Введите пароль: ").strip()
        if not password:
            raise SystemExit("Пароль обязателен.")

    async def runner() -> None:
        engine, service = await prepare_user_service(
            username=args.username,
            password=password,
            role=args.role,
            is_active=not args.inactive,
        )

        try:
            user = await service.create_user(
                username=args.username,
                password=password,
                role=args.role,
                is_active=not args.inactive,
                created_manually=True,
            )
            print(f"✅ Пользователь '{user.username}' (роль: {user.role}) создан.")
        except UserAlreadyExistsError:
            if not args.update:
                print(
                    f"⚠️  Пользователь '{args.username}' уже существует — используйте --update, "
                    "если хотите заменить пароль или параметры."
                )
            else:
                try:
                    user = await service.update_user(
                        args.username,
                        password=password,
                        role=args.role,
                        is_active=not args.inactive,
                    )
                    print(f"🔁 Пользователь '{user.username}' обновлён.")
                except UserNotFoundError:
                    print(f"❌ Не удалось найти пользователя '{args.username}' для обновления.")
        finally:
            await engine.dispose()

    asyncio.run(runner())


if __name__ == "__main__":
    main()
